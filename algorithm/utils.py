import pygame
import sys
sys.path.append('..')
from pgle.games.games import BilliardWorld, BilliardWorldMaze, BomberMan, BomberManMaze
from pgle.games.games import PacWorld, PacWorldMaze, ShootWorld, ShootWorld1d, ShootWorldMaze
from pgle.games.games import WaterWorld, WaterWorld1d, WaterWorldMaze
from random_algorithm import RandomAlgorithm
import pygame
game_names = ['BilliardWorld', 'BilliardWorldMaze', 'BomberMan', 'BomberManMaze',
             'PacWorld', 'PacWorldMaze', 'ShootWorld', 'ShootWorld1d', 'ShootWorldMaze',
             'WaterWorld', 'WaterWorld1d', 'WaterWorldMaze']
algorithm_names = ['RandomAlgorithm']

def load_game(game_name, window_size, maze_width, num_creeps):
    lower2upper = {name.lower():name for name in game_names}
    game = globals()[lower2upper[game_name.lower()]]
    if game.__name__[-4:] == "Maze":
        game = game(width=window_size, maze_width=maze_width, num_creeps=num_creeps)
    else:
        game = game(width=window_size, height=window_size, num_creeps=num_creeps)
    return game

def load_algorithm(env, alg_name):
    lower2upper = {name.lower():name for name in algorithm_names}
    return globals()[lower2upper[alg_name.lower()]](env)
    

    