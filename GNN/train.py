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
from torch_geometric.data import DataLoader
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import Logger, set_random_seed, worker_init_fn, compute_grad_norm, GradualWarmupScheduler, wrap_dataloader, load_model
from dataset import GamePatch
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True,
                    help='input the dataset name')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='the number of epochs for training')
parser.add_argument('--checkpoints_path', type=str,
                    help='model to continue to train')
parser.add_argument('--resume_epoch', type=int,
                    help='resume epoch for the saved model')
parser.add_argument('--model', type=str, default="pointconv",
                    help='model name')
args = parser.parse_args()
BATCH_SIZE = 32
SAVE_EPOCH = 10
PRINT_EPOCH = 1
CROSS_ENTROPY = True
LOSS_BALANCE = True
    
dataset = GamePatch(args)
policy_net = load_model(model=args.model,
                        node_dim=dataset[0].x[0].shape[0], 
                        pos_dim=dataset[0].pos[0].shape[0], 
                        edge_dim=dataset[0].edge_attr[0].shape[0],
                        output_dim = dataset[0].y.shape[1])

optimizer = optim.Adam([{'params': policy_net.parameters(), 'initial_lr': 1e-3}], 1e-3)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0, last_epoch=args.num_epochs)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosine)
optimizer.zero_grad()
optimizer.step()
# set logger
if args.continue_train:
    logger = Logger(filename=os.path.join(args.checkpoints_path, 'log.txt'), mode='a')
    resume_epoch = args.resume_epoch
    save_path = os.path.join(args.checkpoints_path, 'epoch_{}'.format(resume_epoch))
    save_state = torch.load(save_path)
    policy_net.load_state_dict(save_state['policy_net'])
    optimizer.load_state_dict(save_state['optim'])
else:
    os.makedirs(args.checkpoints_path, exist_ok=True)
    logger = Logger(filename=os.path.join(args.checkpoints_path, 'log.txt'), mode='w')
    resume_epoch = 0

sys.stdout = logger
os.makedirs(os.path.join(args.checkpoints_path, 'runs'), exist_ok=True)
writer = SummaryWriter(os.path.join(args.checkpoints_path, 'runs', time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))

# save options
with open(os.path.join(args.checkpoints_path, 'opt.txt'), 'w') as outfile:
    outfile.write(json.dumps(vars(args), indent=2))

# save cmdline
with open(os.path.join(args.checkpoints_path, 'cmdline.txt'), 'w') as outfile:
    outfile.write(' '.join(sys.argv))

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=8,
                              worker_init_fn=worker_init_fn,
                             )
val_dataloader = DataLoader(test_dataset,
                            batch_size=BATCH_SIZE,
                            # shuffle=False,
                            shuffle=True,
                            num_workers=8,
                            worker_init_fn=worker_init_fn,
                           )

if LOSS_BALANCE:
    class_sample_count = sum([data_batch.y for data_batch in dataset])
    weight = torch.sum(class_sample_count) / class_sample_count
    weight /= torch.sum(weight)
    weight = weight.squeeze_(0).to('cuda')
    print("loss weight:{}".format(weight))
else:
    weight = torch.ones(output_dim).to('cuda')

num_epochs = args.num_epochs
for i_epochs in range(resume_epoch, num_epochs):
    # Initialize the environment and state
    scheduler_warmup.step(i_epochs + 1)
    policy_net.train()
    print("[{} epoch] lr = {}".format(i_epochs, optimizer.param_groups[0]['lr']))
    losses = 0
    for i, data_batch in tqdm(enumerate(train_dataloader)):
        # copy data from cpu to gpu
        data_batch = data_batch.to('cuda')
        # print(data_batch.y)
        # forward
        outputs = policy_net(data_batch)
        loss = 0
        q = outputs['q']
        if CROSS_ENTROPY:
            loss = F.cross_entropy(q, data_batch.y.argmax(-1), reduction='mean', weight=weight)  # (b,)
        else:
            loss = F.mse_loss(q, data_batch.y, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss
        # Update the target network, copying all weights and biases in DQN
        if (i+1) % PRINT_EPOCH == 0:
            print("[{}] loss:{}, average loss:{}".
                format((i+1), loss, losses / (i+1)))
            writer.add_scalar('loss', loss, global_step=(i+1)+i_epochs*len(train_dataloader))
            
    if (i_epochs+1) % SAVE_EPOCH == 0:
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
        writer.add_scalar('val_loss', losses / len(val_dataloader), global_step=i_epochs+1)
        writer.add_scalar('accuracy', true_sample / all_sample, global_step=i_epochs+1)

        save_state = {'policy_net': policy_net.state_dict(), 'optim': optimizer.state_dict()}
        save_path = os.path.join(args.checkpoints_path, 'epoch_{}'.format((i_epochs+1)))
        torch.save(save_state, save_path)
        
        policy_net.train()
      
print('Complete')
