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

result_1 = [ 1.  ,  3.  ,  5. ,   7.  ,  9.  , 11. ,  11.3 , 13.63 ,13.79, 13.43]
result_2 = [ 1.  ,  3.  ,  5.  ,  7.  ,  9.  , 10.42, 11.26, 13.29, 13.82, 13.71]
result_3 = [ 1. ,   3. ,   5. ,   7.  ,  9.  , 10.73, 11.45, 13.01, 13.85, 13.04]
results = np.array([result_1, result_2, result_3])
print(results.mean(axis=0), results.std(axis=0))
exit()

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=233)
parser.add_argument('--steps', type=int, default=600)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--exp_name', type=str, default='ppo')
parser.add_argument('--env_setting', type=str, default='AX0_fast')
#parser.add_argument('--gnn_aggr', type=str, default='max')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--v_lr', type=float, default=1e-3)
parser.add_argument('--pi_lr', type=float, default=1e-5)
parser.add_argument('--save_freq', type=int, default=100)

args = parser.parse_args()

env_kwargs = get_env_kwargs(args.env_setting)
env_fn = lambda: GraphObservationEnvWrapper(Arena, env_kwargs)

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
device = torch.device(device_name)
ac = GNNActorCritic(PointConv, pi_kwargs, v_kwargs, device_name).to(device)
ac.load_state_dict(torch.load(args.model_path))

scores = []
for num_coins in [1,3,5,7,9,11,13,15,17,19]:
    current_scores = []
    for epoch in range(args.epochs):
        env_kwargs['num_coins'] = num_coins
        env = GraphObservationEnvWrapper(Arena, env_kwargs)
        o=env.reset().to(device)
        done=False
        while not done:
            a, v, logp = ac.step(o)

            next_o, r, done, _ = env.step(a.item())
            o = next_o.to(device)
        current_scores.append(env.score())
        print(f'num_coins: {num_coins}, score: ', env.score())
    scores.append(current_scores)

scores = np.array(scores)
score_mean = scores.mean(axis=1)
score_std = scores.std(axis=1)
print('mean: ', score_mean)
print('std : ', score_std)
np.save(osp.join(osp.dirname(osp.abspath(args.model_path)), "eval_results.npy"), scores)

