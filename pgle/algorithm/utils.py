import pygame
from ..games import BilliardWorld, BilliardWorldMaze, BomberMan, BomberManMaze
from ..games import PacWorld, PacWorldMaze, ShootWorld, ShootWorld1d, ShootWorldMaze
from ..games import WaterWorld, WaterWorld1d, WaterWorldMaze
from .randomness import Random
from .greedy import OneStep, TwoStep, GreedyCollectV0, GreedyCollectV1, GreedyCollectV2, GreedyCollectMax
from .planning import PlanningCollect, PlanningPac, PlanningShoot1d, PlanningCollectMaze
from .search import SearchBomberManMaze

import pygame
import json
import numpy as np

game_names = ['BilliardWorld', 'BilliardWorldMaze', 'BomberMan', 'BomberManMaze',
             'PacWorld', 'PacWorldMaze', 'ShootWorld', 'ShootWorld1d', 'ShootWorldMaze',
             'WaterWorld', 'WaterWorld1d', 'WaterWorldMaze']
algorithm_names = ['Random', 'OneStep', 'TwoStep', 'GreedyCollectV0', 
                   'GreedyCollectV1', 'GreedyCollectV2', 'GreedyCollectMax', 
                   'PlanningCollect', 'PlanningPac', 'PlanningShoot1d', 'PlanningCollectMaze',
                   'SearchBomberManMaze']


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
    

    