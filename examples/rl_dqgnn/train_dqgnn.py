import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import argparse
from torch import nn
from functools import partial
import copy
import os, sys

dqgnn_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(os.path.dirname(dqgnn_path))
sys.path.append(root_path)

from arena import Arena, Wrapper
from examples.rl_dqgnn.nn_utils import MyPointConv, PointConv, process_state
from dqgnn_agent import DQGNN_agent

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--model_path', type=str, help='Q network path to save/load, for train/eval mode')
parser.add_argument('--num_episodes', type=int, default=2000)
parser.add_argument('--num_rewards', type=int, default=1)
parser.add_argument('--num_enemies', type=int, default=0)
parser.add_argument('--num_bombs', type=int, default=0)
parser.add_argument('--fix_num_rewards', type=bool, default=False)
parser.add_argument('--model_id', type=str, default="")
args= parser.parse_args()

num_episodes=args.num_episodes
is_train=args.train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The configuration AX0
# changed to 64x64 for faster training
h,w=256,256
game = Arena(
    width=w,
    height=h,
    object_size=32,
    num_coins=1,
    num_enemies=0,
    num_bombs=0,
    num_projectiles=0,
    num_obstacles=0,
    agent_speed=8,
    enemy_speed=8, # Since there is no enemy, the speed does not matter.
    projectile_speed=8,
    explosion_max_step=100,
    explosion_radius=128,
    reward_decay=1.0,
    max_step=200
)
env=Wrapper(game)
env.init()
state=env.reset()

def change_num_rewards(env):
    if args.fix_num_rewards:
        env.game.N_REWARDS = args.num_rewards
    else:
        n_rewards = random.choice(np.arange(1,6))
        env.game.N_REWARDS = n_rewards

#input_dim, pos_dim = 8,4
input_dim, pos_dim = 8,4


qnet_local = PointConv(input_dim=input_dim, pos_dim=pos_dim)
qnet_target = PointConv(input_dim=input_dim, pos_dim=pos_dim)
qnet_target.load_state_dict(qnet_local.state_dict())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQGNN_agent(qnet_local, qnet_target, device=device, seed=0)



def dqn(n_episodes=4000, max_t=500, save_freq=10, eps_start=0.9, eps_end=0.05, eps_decay=0.995):
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
        if i_episode % save_freq == 0:
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