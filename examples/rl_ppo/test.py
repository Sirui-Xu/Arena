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
from examples.rl_dqgnn.nn_utils import *
from examples.env_setting_kwargs import get_env_kwargs_dict
from examples.rl_ppo.ppo_core import GNNActorCritic

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=233)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--exp_name', type=str, default='ppo')
parser.add_argument('--env_setting', type=str, default='sanity_check')
#parser.add_argument('--gnn_aggr', type=str, default='max')
parser.add_argument('--v_lr', type=float, default=1e-4)
parser.add_argument('--pi_lr', type=float, default=1e-4)

args = parser.parse_args()

data_path=osp.dirname(osp.abspath(__file__))
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=data_path)

env_kwargs = get_env_kwargs_dict(args.env_setting)
env_fn = lambda: GraphObservationEnvWrapper(Arena, env_kwargs)
print('caution: for debugging, set env setting to extremely simple and fixed initialization')
input_dim, pos_dim = 8,4
pi_kwargs = {
    'aggr': 'max',
    'input_dim':input_dim,
    'pos_dim':pos_dim,
}
v_kwargs = {
    'aggr': 'add',
    'input_dim':input_dim,
    'pos_dim':pos_dim,
    'output_dim': 1
}

device_name = 'cuda:0'

# Create actor-critic module
ac = GNNActorCritic(PointConv, pi_kwargs, v_kwargs, device_name)
ac.load_state_dict(torch.load('/home/yiran/pc_mapping/arena-v2/examples/rl_ppo/ppo/ppo_s237/pyt_save/model20.pt'))

env = env_fn()
#print(env.getGameState()['local'][0]['position'])
#env.step(5)
#print(env.getGameState()['local'][0]['position'])
#exit()
for i in range(100):
    state=env.reset()
    done=False
    while not done:
        a, v, logp = ac.step(state)
        next_state, r, done, _ = env.step(a.item())
        print('action:', a)
        state=next_state
    print('score:', env.score())
exit()
#img=env_fn().render()
#import matplotlib.pyplot as plt
#plt.imshow(img)
#plt.show()
#exit()