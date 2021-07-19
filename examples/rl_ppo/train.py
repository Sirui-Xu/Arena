from functools import partial
import argparse
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.run_utils import setup_logger_kwargs

from arena import Arena, Wrapper
from examples.rl_ppo.ppo import ppo
from examples.rl_ppo.ppo_core import MLPActorCritic


parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=4)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--exp_name', type=str, default='ppo')

args = parser.parse_args()

mpi_fork(args.cpu)  # run parallel code with mpi


logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


h,w=256,256
game_func = partial(Arena,
                    width=256, height=256,
                    object_size=32,
                    num_coins=1,
                    num_enemies=0,
                    num_bombs=0,
                    num_projectiles=0,
                    num_obstacles=0,
                    agent_speed=8,
                    enemy_speed=8,  # Since there is no enemy, the speed does not matter.
                    projectile_speed=8,
                    explosion_max_step=100,
                    explosion_radius=128,
                    reward_decay=1.0,
                    max_step=200)

env_func = lambda : Wrapper(game_func())


ppo(env_func, actor_critic=MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    logger_kwargs=logger_kwargs)