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
import matplotlib.pyplot as plt

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return super(NpEncoder, self).default(obj)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True,
                    help='input the dataset name')
parser.add_argument('--model_dir', type=str)
parser.add_argument('--model', type=str, default="pointconv",
                    help='model name')
parser.add_argument('--aggr', type=str, default='max',
                    help='aggregation function')
args = parser.parse_args()

eval_results=[]
for i in range(10):
    eval_result_path = args.model_dir+f'/run{i}/eval_result.npy'
    eval_results.append(np.load(eval_result_path))
eval_results = np.array(eval_results)
correct_cnt = np.sum(eval_results, axis=0)
x=np.arange(0,11)
y=np.zeros([11])
for i in range(correct_cnt.shape[0]):
    y[correct_cnt[i]]+=1
#y/=y.sum()
#print(y)

def plot():
    plt.xticks(x)
    plt.yticks(0.1*np.arange(11))
    plt.ylim(0.0,1.0)
    plt.plot(x,y)
    plt.show()

def plot_cumulative(x,y):
    x=np.flip(x)
    y=np.flip(y)
    sum=0
    for i in range(x.shape[0]):
        y[i] += sum
        print('sum=',sum)
        sum+=y[i]-sum
    print(y)
    plt.xlim(max(x), min(x))
    plt.ylim(0.0,1.0)
    plt.plot(x,y)
    plt.show()
#plot_cumulative(x,y)
#exit()

data_path = args.data_path
with open(data_path, 'r') as f:
    data = json.load(f)
data_sa = data['data']
data_ks = [None] * 10
for i in range(1,11):
    indices = correct_cnt>=i
    print(indices)
    data_sa_k = []
    for j, flag in enumerate(indices):
        if flag:
            data_sa_k.append(data_sa[j])
    data['data'] = data_sa_k
    print(f'>={i} count: {len(data_sa_k)}')
    with open(f'/home/yiran/pc_mapping/arena-v2/examples/bc_filtered_data/{i}of10.json', 'w') as f:
        #print(type(data), type(f))
        json.dump(data, f, cls=NpEncoder)
#print('data len:', len(data['data']))