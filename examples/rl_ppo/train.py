from functools import partial
import argparse
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from arena import Arena, Wrapper
#from examples.rl_ppo.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from examples.rl_ppo.utils.run_utils import setup_logger_kwargs

from examples.rl_ppo.ppo import ppo
from examples.rl_ppo.ppo_core import GNNActorCritic
from examples.env_setting_kwargs import get_env_kwargs_dict
from examples.rl_dqgnn.nn_utils import PointConv, EnvStateProcessor
from examples.env_setting_kwargs import get_env_kwargs_dict

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--steps', type=int, default=2000)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--exp_name', type=str, default='ppo')
parser.add_argument('--env_setting', type=str, default='AY0')
parser.add_argument('--gnn_aggr', type=str, default='max')
parser.add_argument('--lr', type=float, default=1e-3)


args = parser.parse_args()

data_path=osp.dirname(osp.abspath(__file__))
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=data_path)

env_kwargs_dict = get_env_kwargs_dict(args.env_setting)
env_func = lambda : Wrapper(Arena(**env_kwargs_dict))

state_processor = EnvStateProcessor(env_kwargs_dict)
input_dim, pos_dim = 8,4
network_kwargs_dict = {
    'aggr': args.gnn_aggr,
    'input_dim':input_dim,
    'pos_dim':pos_dim,
}

device_name = 'cuda:0'

ppo(env_func, PointConv, network_kwargs_dict, state_processor, device_name,
    actor_critic=GNNActorCritic, gamma=args.gamma, vf_lr=args.lr,
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    logger_kwargs=logger_kwargs, save_freq=100)