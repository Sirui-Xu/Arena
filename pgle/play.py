import pygame
import numpy as np
import sys
from .games import BilliardWorld, BilliardWorldMaze, BomberMan, BomberManMaze
from .games import PacWorld, PacWorldMaze, ShootWorld, ShootWorld1d, ShootWorldMaze
from .games import WaterWorld, WaterWorld1d, WaterWorldMaze
import os

def play(game_name):
    os.environ.pop("SDL_VIDEODRIVER")
    lower2upper = {upper.lower():upper for upper in globals().keys()}
    game = globals()[lower2upper[game_name.lower()]]
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

    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        game_name = sys.argv[1]
        play(game_name)
    else:
        raise Exception('Please input a game name')
