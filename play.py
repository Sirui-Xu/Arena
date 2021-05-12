import pygame
import numpy as np
import sys
from pgle.games import BilliardWorld, BilliardWorldMaze, BomberMan, BomberManMaze
from pgle.games import PacWorld, PacWorldMaze, ShootWorld, ShootWorld1d, ShootWorldMaze
from pgle.games import WaterWorld, WaterWorld1d, WaterWorldMaze
import os
os.environ.pop("SDL_VIDEODRIVER")

lower2upper = {upper.lower():upper for upper in globals().keys()}
if len(sys.argv) >= 2:
    game_name = sys.argv[1]
    game = globals()[lower2upper[game_name.lower()]]
else:
    raise Exception('Please input a game name')

pygame.init()
if game.__name__[-4:] == "Maze":
    game = game(width=512, maze_width=15, num_creeps=3)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()
    fps = 5
    while True:
        for i in range(game.fps):
            dt = game.clock.tick_busy_loop(fps*game.fps)
            game.step()
            pygame.display.update()

        if game.game_over() is True:
            print(game.getGameState(), '\n')
            print("The overall score is {}.".format(game.score))
            break
        print(game.getGameState(), '\n')
else:
    game = game(width=512, height=512, num_creeps=3)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()
    while True:
        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
        if game.game_over() is True:
            print(game.getGameState(), '\n')
            print("The overall score is {}.".format(game.score))
            break
        print(game.getGameState(), '\n')