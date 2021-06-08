import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import argparse
from torch import nn
from functools import partial
import copy

from pgle import PGLE
from pgle.games import ARENA

from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--model_path', type=str, help='Q network path to save/load, for train/eval mode')
parser.add_argument('--num_episode', type=int, default=2000)
parser.add_argument('--num_rewards', type=int, default=1)
parser.add_argument('--fix_num_rewards', type=bool, default=False)
parser.add_argument('--model_id', type=str, default="")
args= parser.parse_args()

num_episodes=args.num_episode
is_train=args.train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The configuration AX0
# changed to 64x64 for faster training
h,w=64,64
game = ARENA(
    width=w,
    height=h,
    object_size=8,
    num_rewards=1,
    num_enemies=0,
    num_bombs=0,
    num_projectiles=3,
    num_obstacles=0,
    num_obstacles_groups=1,
    agent_speed=0.25,
    enemy_speed=0.25, # Since there is no enemy, the speed does not matter.
    projectile_speed=0.25,
    bomb_life=100,
    bomb_range=4,
    duration=200
)
env = PGLE(game)
env.init()
env.reset()
init_state=copy.deepcopy(env.game.getGameState())

def change_num_rewards(env):
    if args.fix_num_rewards:
        env.game.N_REWARDS = args.num_rewards
    else:
        n_rewards = random.choice(np.arange(1,6))
        env.game.N_REWARDS = n_rewards


import torch_geometric.nn as gnn
from torch_scatter import scatter
from torch_geometric.nn import GENConv, DeepGCNLayer, global_max_pool, global_add_pool

class MyPointConv(gnn.PointConv):
    def __init__(self, aggr, local_nn=None, global_nn=None, **kwargs):
        super(gnn.PointConv, self).__init__(aggr=aggr, **kwargs)
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.reset_parameters()
        self.add_self_loops = True

class PointConv(nn.Module):

    def __init__(self, aggr='max', input_dim=8, pos_dim=4, edge_dim=None, output_dim=6): # edge_dim is not used!
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.encoder_pos = nn.Sequential(
            nn.Linear(pos_dim, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )
        local_nn = nn.Sequential(
            nn.Linear(32 + 64 + pos_dim, 512, bias=False), nn.GroupNorm(8, 512), nn.ReLU(),
            nn.Linear(512, 512, bias=False), nn.GroupNorm(8, 512), nn.ReLU(),
            nn.Linear(512, 512),
        )
        global_nn = nn.Sequential(
            nn.Linear(512, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256),
        )
        gate_nn = nn.Sequential(
            nn.Linear(256, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 256), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        self.gnn = MyPointConv(aggr='add', local_nn=local_nn, global_nn=global_nn)
        self.aggr = aggr
        self.attention = gnn.GlobalAttention(gate_nn)

        self.reset_parameters()

    def forward(self, data, batch_size=None, **kwargs):
        assert batch_size is not None
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        pos = data.pos

        x = self.encoder(x)
        pos_feature = self.encoder_pos(pos)
        task_feature = x
        feature = torch.cat([task_feature, pos_feature], dim=1)
        x = self.gnn(x=feature, pos=pos, edge_index=edge_index)
        x = self.attention(x, batch, size=batch_size) 
        x = F.dropout(x, p=0.1, training=self.training)
        q = self.decoder(x)
        return q

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def process_state(state):
    local_state = state['local']
    num_objs = len(local_state)
    edges = []
    for i in range(num_objs):
        for j in range(num_objs):
            if i != j:
                edges.append((i,j))
    edges = (np.array(edges).T).astype(np.int64)
    edge_index = torch.from_numpy(edges)

    x, pos = [], []
    for node in local_state:
        if node['type'] == 'agent':
            p = node['position']
    for node in local_state:
        rel_pos = np.array(node['position']) - np.array(p) / 8.0 # Hard code the size of an object.
        x.append(node['type_index'] + node['velocity'] + rel_pos.tolist())
        pos.append(node['position'] + rel_pos.tolist())
    x = torch.tensor(x)
    pos = torch.tensor(pos)
    return Data(x=x, edge_index=edge_index, pos=pos)

from dqgnn_agent import DQGNN_agent

qnet_local = PointConv()
qnet_target = PointConv()
qnet_target.load_state_dict(qnet_local.state_dict())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQGNN_agent(qnet_local, qnet_target, device=device, seed=0) 



def dqn(n_episodes=4000, max_t=500, eps_start=0.9, eps_end=0.05, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        change_num_rewards(env)
        state = env.reset()
        state = process_state(state)
        score = 0
        for t in range(max_t):
            action, action_type = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('Episode {}\t Score: {:.2f}'.format(i_episode, score))
        if i_episode % 100 == 0:
            np.save(args.model_path+f'/score.npy', np.array(scores))
            torch.save(agent.qnetwork_local.state_dict(), args.model_path+f'/ep{i_episode}.pth')

    return scores


if __name__=='__main__':

    model_path=args.model_path
    os.makedirs(model_path, exist_ok=True)
    if is_train:
        scores = dqn(n_episodes=num_episodes)
        torch.save(agent.qnetwork_local.state_dict(), os.path.join(model_path, 'final.pth'))
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    else:
        agent.qnetwork_local.load_state_dict(torch.load(model_path))

        num_test_episodes=1000
        for i in range(num_test_episodes):
            state = env.reset()
            print('test episode %d'%i)
            score=0.0
            for t in range(1000):
                action, action_type = agent.act(state, 0.0)
                #print('final %s action:'%action_type, action)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                score += reward
                if done:
                    break
            print('final score:', score)
