from functools import partial
import argparse
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from arena import Arena, Wrapper
#from examples.rl_ppo.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from examples.rl_ppo.utils.run_utils import setup_logger_kwargs

from examples.rl_ppo.ppo import ppo
from examples.rl_ppo.ppo_core import *
from examples.rl_dqgnn.nn_utils import *
from examples.env_setting_kwargs import get_env_kwargs

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=233)
parser.add_argument('--steps', type=int, default=600)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--exp_name', type=str, default='ppo')
parser.add_argument('--env_setting', type=str, default='AX0')
#parser.add_argument('--gnn_aggr', type=str, default='max')
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--v_lr', type=float, default=1e-3)
parser.add_argument('--pi_lr', type=float, default=1e-5)
parser.add_argument('--save_freq', type=int, default=100)

args = parser.parse_args()

data_path=osp.dirname(osp.abspath(__file__))
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=data_path)
output_dir = args.save_path
logger_kwargs['output_dir'] = args.save_path
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
os.system(f'cp {data_path}/*.py {output_dir}/')

env_kwargs = get_env_kwargs(args.env_setting)
env_fn = lambda: GraphObservationEnvWrapper(Arena, env_kwargs)
'''
env = env_fn()
for i in range(100):
    env.reset()
    done=False
    actions = [4]*25 + [5]*25
    i=0
    while not done:
        next_s, r, done, _ = env.step(actions[i])
        i+=1
    print('score:', env.score())
exit()
'''
#img=env_fn().render()
#import matplotlib.pyplot as plt
#plt.imshow(img)
#plt.show()
#exit()
print('caution: for debugging, set env setting to extremely simple and fixed initialization')
input_dim, pos_dim = 8,4
pi_kwargs = {
    'aggr': 'max',
    'input_dim':input_dim,
    'pos_dim':pos_dim,
    'dropout':False
}
v_kwargs = {
    'aggr': 'add',
    'input_dim':input_dim,
    'pos_dim':pos_dim,
    'output_dim': 1,
    'dropout':False
}

device_name = 'cuda:0'

ppo(env_fn, PointConv, pi_kwargs, v_kwargs, device_name,
    actor_critic=GNNActorCritic, gamma=args.gamma, vf_lr=args.v_lr, pi_lr=args.pi_lr,
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    logger_kwargs=logger_kwargs, save_freq=args.save_freq)