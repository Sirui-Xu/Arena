import pygame
from ..games import BilliardWorld, BilliardWorldMaze, BomberMan, BomberManMaze
from ..games import PacWorld, PacWorldMaze, ShootWorld, ShootWorld1d, ShootWorldMaze
from ..games import WaterWorld, WaterWorld1d, WaterWorldMaze, ARENA
from .randomness import Random
from .greedy import OneStep, TwoStep, GreedyCollectV0, GreedyCollectV1, GreedyCollectV2, GreedyCollectMax, GreedyArena
from .planning import PlanningCollect, PlanningPac, PlanningShoot1d, PlanningCollectMaze, PlanningArena
from .search import SearchBomberManMaze

import pygame
import json
import numpy as np

game_names = ['BilliardWorld', 'BilliardWorldMaze', 'BomberMan', 'BomberManMaze',
             'PacWorld', 'PacWorldMaze', 'ShootWorld', 'ShootWorld1d', 'ShootWorldMaze',
             'WaterWorld', 'WaterWorld1d', 'WaterWorldMaze', 'ARENA']
algorithm_names = ['Random', 'OneStep', 'TwoStep', 'GreedyCollectV0', 
                   'GreedyCollectV1', 'GreedyCollectV2', 'GreedyCollectMax', 
                   'PlanningCollect', 'PlanningPac', 'PlanningShoot1d', 'PlanningCollectMaze',
                   'SearchBomberManMaze', 'GreedyArena', 'PlanningArena']


def load_game(game_name, window_size, maze_width, num_creeps, fps=20):
    lower2upper = {name.lower():name for name in game_names}
    game = globals()[lower2upper[game_name.lower()]]
    if game.__name__ == "ARENA":
        game = game(width=128,
                    height=128,
                    object_size=8,
                    num_rewards=num_creeps,
                    num_enemies=num_creeps,
                    num_bombs=3,
                    num_projectiles=3,
                    num_obstacles=2 * num_creeps,
                    num_obstacles_groups=1,
                    agent_speed=0.25,
                    enemy_speed=0.25,
                    projectile_speed=1,
                    bomb_life=100,
                    bomb_range=4,
                    visualize=True)
    elif game.__name__[-4:] == "Maze":
        game = game(width=window_size, maze_width=maze_width, num_creeps=num_creeps)
    else:
        game = game(width=window_size, height=window_size, num_creeps=num_creeps, fps=fps)
    return game

def load_algorithm(env, alg_name):
    lower2upper = {name.lower():name for name in algorithm_names}
    return globals()[lower2upper[alg_name.lower()]](env)
