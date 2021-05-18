import pygame
import sys
sys.path.append('..')
from pgle.games import BilliardWorld, BilliardWorldMaze, BomberMan, BomberManMaze
from pgle.games import PacWorld, PacWorldMaze, ShootWorld, ShootWorld1d, ShootWorldMaze
from pgle.games import WaterWorld, WaterWorld1d, WaterWorldMaze
from randomness import Random
from greedy import OneStep, TwoStep, GreedyCollectV0, GreedyCollectV1, GreedyCollectV2
from planning import PlanningCollect

import pygame
import json
import numpy as np

game_names = ['BilliardWorld', 'BilliardWorldMaze', 'BomberMan', 'BomberManMaze',
             'PacWorld', 'PacWorldMaze', 'ShootWorld', 'ShootWorld1d', 'ShootWorldMaze',
             'WaterWorld', 'WaterWorld1d', 'WaterWorldMaze']
algorithm_names = ['Random', 'OneStep', 'TwoStep', 'GreedyCollectV0', 'GreedyCollectV1', 'GreedyCollectV2', 'PlanningCollect']

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return super(NpEncoder, self).default(obj)

def load_game(game_name, window_size, maze_width, num_creeps, fps):
    lower2upper = {name.lower():name for name in game_names}
    game = globals()[lower2upper[game_name.lower()]]
    if game.__name__[-4:] == "Maze":
        game = game(width=window_size, maze_width=maze_width, num_creeps=num_creeps)
    else:
        game = game(width=window_size, height=window_size, num_creeps=num_creeps, fps=fps)
    return game

def load_algorithm(env, alg_name):
    lower2upper = {name.lower():name for name in algorithm_names}
    return globals()[lower2upper[alg_name.lower()]](env)
    

    