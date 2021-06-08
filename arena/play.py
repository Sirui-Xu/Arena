import pygame
import numpy as np
import sys
import time
from .games import ARENA
import os

def play():
    os.environ.pop("SDL_VIDEODRIVER")
    #lower2upper = {upper.lower():upper for upper in globals().keys()}
    #game = globals()[lower2upper[game_name.lower()]]
    pygame.init()

    game = ARENA()
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    seed = int(round(time.time() * 1000)) % (2 ** 32)
    game.rng = np.random.RandomState(seed)
    game.init()
    while True:
        dt = game.clock.tick_busy_loop(20)
        game.step(dt)
        pygame.display.update()
        if game.game_over() is True:
            #print(game.getGameState(), '\n')
            print("The overall score is {}.".format(game.score))
            break
        #print(game.getGameState(), '\n')

    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == "__main__":
    play()
