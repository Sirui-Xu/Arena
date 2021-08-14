# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import Logger, set_random_seed, worker_init_fn, compute_grad_norm, GradualWarmupScheduler, wrap_dataloader, \
    load_model, load_model_info
from dataset import GamePatch

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True,
                    help='input the dataset name')
parser.add_argument('--model_path', type=str)
parser.add_argument('--model', type=str, default="pointconv",
                    help='model name')
parser.add_argument('--aggr', type=str, default='max',
                    help='aggregation function')
args = parser.parse_args()
BATCH_SIZE = 128
CROSS_ENTROPY = True
LOSS_BALANCE = True

data_path = args.data_path
with open(data_path, 'r') as f:
    data = json.load(f)

dataset = GamePatch(data["data"])
model_info = load_model_info(dataset, aggr=args.aggr)
policy_net = load_model(model=args.model, info=model_info)
policy_net.load_state_dict(torch.load(args.model_path)['policy_net'])
policy_net = policy_net.to(device)


train_dataset=dataset
train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              drop_last=False,
                              num_workers=8,
                              worker_init_fn=worker_init_fn,
                              )

print("==> data size: {}\n".format(len(dataset)))


policy_net.eval()
true_sample, all_sample = 0, 0
eval_result = []
for i, data_batch in tqdm(enumerate(train_dataloader)):
    # copy data from cpu to gpu
    data_batch = data_batch.to('cuda')

    # forward
    with torch.no_grad():
        outputs = policy_net(data_batch)
        loss = 0
        q = outputs['q']

        actions = q.cpu().argmax(-1).numpy()
        gt_actions = data_batch.y.cpu().argmax(-1).numpy()
        true_flag = actions == gt_actions
        true_sample+=np.sum(true_flag)
        all_sample += actions.shape[0]
        #print(true_flag)
        eval_result.append(true_flag)
print('acc: ',true_sample/all_sample)
eval_result = np.concatenate(eval_result)
np.save(os.path.join(os.path.dirname(args.model_path), "eval_result.npy"), eval_result)