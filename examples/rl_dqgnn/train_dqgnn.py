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
import pickle
from statistics import mean

dqgnn_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(os.path.dirname(dqgnn_path))
sys.path.append(root_path)

from arena import Arena, Wrapper
from examples.rl_dqgnn.nn_utils import *
from examples.env_setting_kwargs import get_env_kwargs_dict
from dqgnn_agent import DQGNN_agent

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--model_path', type=str, help='Q network path to save')
parser.add_argument('--num_episodes', type=int, default=2000)
parser.add_argument('--fix_num_rewards', type=bool, default=False)
parser.add_argument('--model_id', type=str, default="")
parser.add_argument('--env_setting', type=str, default='AX0')
parser.add_argument('--gnn_aggr', type=str, default='max')
parser.add_argument('--nn_name', type=str, default='PointConv')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--double_q', type=bool, default=True)
parser.add_argument('--PER', action='store_true')
parser.add_argument('--save_experience', action='store_true')
parser.add_argument('--hard_update', action='store_true')
parser.add_argument('--target_update_freq', type=int, default=500)
args= parser.parse_args()
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

num_episodes=args.num_episodes
is_train=args.train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_kwargs = get_env_kwargs_dict(args.env_setting)
#env=Wrapper(Arena(**kwargs_dict))
env = GraphObservationEnvWrapper(Arena, env_kwargs)
env.reset()

nn_func = get_nn_func(args.nn_name)
input_dim, pos_dim = 8,4
network_kwargs_dict = {
    'aggr': args.gnn_aggr,
    'input_dim':input_dim,
    'pos_dim':pos_dim,
    'dropout':False
}

qnet_local = nn_func(**network_kwargs_dict)
qnet_target = nn_func(**network_kwargs_dict)
qnet_target.load_state_dict(qnet_local.state_dict())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQGNN_agent(qnet_local, qnet_target, lr=args.lr,
                    hard_update = args.hard_update,
                    target_update_freq=args.target_update_freq, double_q=args.double_q,
                    PER = args.PER, replay_eps = 0.01, replay_alpha = 0.5,
                    replay_beta = LinearSchedule(0.4, 1.0, args.num_episodes * 100),
                    device=device, seed=0)

def dqn(n_episodes=4000, max_t=500, save_freq=200, eps_start=0.9, eps_end=0.05, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    with open(args.model_path + '/env_kwargs.pkl', 'wb') as f:
        pickle.dump(env_kwargs, f, pickle.HIGHEST_PROTOCOL)
    with open(args.model_path + '/network_kwargs.pkl', 'wb') as f:
        pickle.dump(network_kwargs_dict, f, pickle.HIGHEST_PROTOCOL)
    os.system('cp /home/yiran/pc_mapping/arena-v2/examples/rl_dqgnn/*.py '+args.model_path)
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    #state_processor = EnvStateProcessor(env_kwargs)
    experience_saver = ExperienceSaver(args.model_path + '/experiences')
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        state_raw = env._state_raw
        score = 0
        for t in range(max_t):
            action, action_type = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state_raw = env._state_raw
            agent.step(state, action, reward, next_state, done)
            if args.save_experience:
                experience_saver.store(state_raw, action, reward)
            state_raw = next_state_raw
            state = next_state
            score += reward
            if done:
                break
        if args.save_experience:
            experience_saver.close_traj()
        scores_window.append(score)  # save most recent score
        scores.append(mean(scores_window))  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('Episode {}\t Score: {:.2f}'.format(i_episode, mean(scores_window)))
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
        plt.savefig(args.model_path+"/score_plot.png")
        #plt.show()

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