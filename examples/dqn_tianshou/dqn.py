import numpy as np
import copy
import os, sys
import pickle
from collections import deque
from functools import partial
from statistics import mean
import argparse

import torch
from torch import nn
import tianshou as ts
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pprint

dqn_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(os.path.dirname(dqn_path))
sys.path.append(root_path)

from arena import Arena, Wrapper
from examples.rl_dqgnn.nn_utils import *
from examples.env_setting_kwargs import get_env_kwargs
from examples.dqn_tianshou.venvs import GraphSubprocVectorEnv, GraphDummyVectorEnv
from examples.dqn_tianshou.graph_collector import GraphCollector as Collector
from examples.dqn_tianshou.dqn_policy import DQGNNPolicy
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from examples.dqn_tianshou.graph_vector_buffer import VectorReplayBuffer, PrioritizedVectorReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument('--env_setting', type=str, default='AX0')
parser.add_argument('--nn_name', type=str, default='TSPointConv')
parser.add_argument('--gnn_aggr', type=str, default='add')
parser.add_argument('--eps-test', type=float, default=0.005)
parser.add_argument('--eps-train', type=float, default=1.)
parser.add_argument('--eps-train-final', type=float, default=0.05)
parser.add_argument('--buffer-size', type=int, default=100000)
parser.add_argument('--prioritized_replay', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--target-update-freq', type=int, default=2000)
# need to check whether dividing grad_step is needed
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--step-per-epoch', type=int, default=100000)
parser.add_argument('--step-per-collect', type=int, default=4)
parser.add_argument('--n_step', type=int, default=1)
parser.add_argument('--update-per-step', type=float, default=0.1)
parser.add_argument('--batch-size', type=int, default=32)
# dummy setting, should change when finish debugging
parser.add_argument('--training-num', type=int, default=4)
parser.add_argument('--test-num', type=int, default=4)
# parallel sampling env num
parser.add_argument('--logdir', type=str, default='log')
parser.add_argument('--device', type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--resume-path', type=str, default=None)
parser.add_argument('--watch', default=False, action='store_true',
                    help='watch the play of pre-trained policy only')
parser.add_argument('--save-buffer-name', type=str, default=None)
parser.add_argument('--seed', type=int, default=233)
args = parser.parse_args()

env_kwargs = get_env_kwargs(args.env_setting)
game_fn = lambda kwargs: Wrapper(Arena(**kwargs))
env_fn = lambda: GraphObservationEnv(game_fn, env_kwargs)
# for parallel sampling, would require to set seed
#subprocenv = GraphSubprocVectorEnv([env_fn, env_fn, env_fn, env_fn])
#env=subprocenv.reset()
#obs = subprocenv.step(np.array([0,0,1,1]),[0,1,2,3])
#print(type(obs))
#exit()
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (32768, rlimit[1]))

def test_dqn(args):
    train_envs = GraphSubprocVectorEnv([env_fn for _ in range(args.training_num)])
    test_envs = GraphSubprocVectorEnv([env_fn for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # Q_param = V_param = {"hidden_sizes": [128]}
    # model
    nn_func = get_nn_func(args.nn_name)
    input_dim, pos_dim = 8, 4
    network_kwargs_dict = {
        'aggr': args.gnn_aggr,
        'input_dim': input_dim,
        'pos_dim': pos_dim,
        'dropout': False,
        'device': args.device
    }

    net = nn_func(**network_kwargs_dict).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQGNNPolicy(
        net, optim, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)
    # whether buffer can hold graph data?
    # buffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size, buffer_num=len(train_envs),
            alpha=args.alpha, beta=args.beta)
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    #print('collector data:', train_collector.data)
    #exit()
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, args.env_setting+'_dqn')
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        #return mean_rewards >= env.spec.reward_threshold
        return False

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.step_per_collect, args.test_num,
        args.batch_size, update_per_step=args.update_per_step, train_fn=train_fn,
        test_fn=test_fn, stop_fn=stop_fn, save_fn=save_fn, logger=logger)
    assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

    # save buffer in pickle format, for imitation learning unittest
    buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(test_envs))
    policy.set_eps(0.2)
    collector = Collector(policy, test_envs, buf, exploration_noise=True)
    result = collector.collect(n_step=args.buffer_size)
    pickle.dump(buf, open(args.save_buffer_name, "wb"))
    print(result["rews"].mean())


def test_pdqn(args):
    args.prioritized_replay = True
    args.gamma = .95
    args.seed = 1
    test_dqn(args)


test_dqn(args)