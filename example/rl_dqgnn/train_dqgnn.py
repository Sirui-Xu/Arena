import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import argparse

import ENV
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--model_path', type=str, help='Q network path to save/load, for train/eval mode')
parser.add_argument('--num_episode', type=int, default=5000)
args= parser.parse_args()

#env = gym.make('LunarLander-v2')
num_episodes=5000
is_train=args.train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = ENV(TBD)
env_name=env.name
env.seed(0)

print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

model=NETWORK
from dqgnn_agent import DQGNN_agent

agent = DQGNN_agent(state_input_dim=env.observation_space.shape, state_output_dim=env.action_space.n,
                    device=device, seed=0)

# watch an untrained agent
'''
state = env.reset()
for j in range(200):
    action = agent.act(state)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break

env.close()
'''


def dqn(n_episodes=4000, max_t=1000, eps_start=1.0, eps_end=0.1, eps_decay=0.995):
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
        state = env.reset()
        score = 0
        for t in range(max_t):
            action, action_type = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        #if np.mean(scores_window) >= 200.0:
        #    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
        #                                                                                 np.mean(scores_window)))
        #    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        #    break

    return scores

model_path=args.model_path
if is_train:
    scores = dqn(n_episodes=num_episodes)
    torch.save(agent.qnetwork_local.state_dict(), model_path)
    # plot the scores
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.plot(np.arange(len(scores)), scores)
    #plt.ylabel('Score')
    #plt.xlabel('Episode #')
    #plt.show()

else:
    agent.qnetwork_local.load_state_dict(torch.load(model_path))

    env = ENV(TBD)
    num_test_episodes=1000
    num_correct=0
    num_trivial=0
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