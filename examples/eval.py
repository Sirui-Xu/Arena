# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import Logger, set_random_seed, worker_init_fn, compute_grad_norm, GradualWarmupScheduler, wrap_dataloader, load_model, load_model_info
from dataset import GamePatch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints_path', type=str,
                    help='checkpoint path')
parser.add_argument('--dataset', type=str, required=True,
                    help='input the dataset name')
parser.add_argument('--resume_epoch', type=int, default=0,
                    help='resume epoch for the saved model')
args = parser.parse_args()

with open(os.path.join(args.checkpoints_path, 'info.json'), 'r') as f:
    info = json.load(f)

model = info["model"]
model_info = info["model_info"]
CROSS_ENTROPY = True
LOSS_BALANCE = False

data_path = args.dataset
with open(data_path, 'r') as f:
    data = json.load(f)

dataset = GamePatch(data["data"])
policy_net = load_model(model=model, info=model_info).to(device)
resume_epoch = args.resume_epoch
save_path = os.path.join(args.checkpoints_path, 'epoch_{}'.format(resume_epoch))
save_state = torch.load(save_path)
policy_net.load_state_dict(save_state['policy_net'])

val_dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            # shuffle=False,
                            shuffle=True,
                            num_workers=8,
                            worker_init_fn=worker_init_fn,
                           )

print("==> data size: {}\n".format(len(dataset)))
if LOSS_BALANCE:
    class_sample_count = sum([data_batch.y for data_batch in dataset])
    weight = torch.sum(class_sample_count) / class_sample_count
    weight /= torch.sum(weight)
    weight = weight.squeeze_(0).to('cuda')
    print("loss weight:{}".format(weight))
else:
    weight = torch.ones(dataset[0].y.shape[1]).to('cuda')

policy_net.eval()
losses = 0
true_sample, all_sample = 0, 0
for i, data_batch in tqdm(enumerate(val_dataloader)):
    # copy data from cpu to gpu
    data_batch = data_batch.to('cuda')

    # forward
    with torch.no_grad():
        outputs = policy_net(data_batch)
        loss = 0
        q = outputs['q']
        if CROSS_ENTROPY:
            loss = F.cross_entropy(q, data_batch.y.argmax(-1), reduction='mean', weight=weight)  # (b,)               
        else:
            loss = F.mse_loss(q, data_batch.y, reduction='mean')

        losses += loss

        actions = q.cpu().argmax(-1).numpy()
        gt_actions = data_batch.y.cpu().argmax(-1).numpy()
        true_sample += np.sum(actions == gt_actions)
        all_sample += actions.shape[0]
        
print("evaluation [{}]  loss:{}  accuracy:{}".
    format((i_epochs+1), losses / len(val_dataloader), true_sample / all_sample))
      
print('Complete')
