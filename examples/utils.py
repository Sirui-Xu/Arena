import sys
import os
import itertools
import random
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import *

models_name = ['EdgeConv', 'PointConv', 'DeeperGCN', 'EdgeConvTongzhou']

def load_model_info(dataset):
    info = {"node_dim": dataset[0].x[0].shape[0],
            "pos_dim": dataset[0].pos[0].shape[0],
            "edge_dim": dataset[0].edge_attr[0].shape[0],
            "output_dim": dataset[0].y.shape[1]}
    return info

def load_model(model, info):
    node_dim=info["node_dim"] 
    pos_dim=info["pos_dim"]
    edge_dim=info["edge_dim"]
    output_dim = info["output_dim"]
    lower2upper = {name.lower():name for name in models_name}
    return globals()[lower2upper[model.lower()]](input_dim=node_dim, pos_dim=pos_dim, edge_dim=edge_dim, output_dim=output_dim)
 
class Logger(object):
    def __init__(self, filename="log.txt", mode='a'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        self.log.flush()

class IterationBasedBatchSampler(Sampler):
    """
    Wraps a BatchSampler, resampling from it until a specified number of iterations have been sampled

    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


class RepeatSampler(Sampler):
    def __init__(self, data_source, repeats=1):
        self.data_source = data_source
        self.repeats = repeats

    def __iter__(self):
        return iter(itertools.chain(*[range(len(self.data_source))] * self.repeats))

    def __len__(self):
        return len(self.data_source) * self.repeats


def wrap_dataloader(dataloader: DataLoader, num_iters, start_iter=0):
    batch_sampler = dataloader.batch_sampler
    dataloader.batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return dataloader


def test_IterationBasedBatchSampler():
    from torch.utils.data.sampler import SequentialSampler, BatchSampler
    sampler = SequentialSampler([i for i in range(10)])
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, 5)

    # check __len__
    assert len(batch_sampler) == 5
    for i, index in enumerate(batch_sampler):
        assert [i * 2, i * 2 + 1] == index

    # check start iter
    batch_sampler.start_iter = 2
    assert len(batch_sampler) == 3


def test_RepeatSampler():
    data_source = [1, 2, 5, 3, 4]
    repeats = 5
    sampler = RepeatSampler(data_source, repeats=repeats)
    assert len(sampler) == repeats * len(data_source)
    sampled_indices = list(iter(sampler))
    assert sampled_indices == list(range(len(data_source))) * repeats

def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


def compute_grad_norm(parameters, norm_type=2):
    """Refer to torch.nn.utils.clip_grad_norm_"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm



class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)