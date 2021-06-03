import pygame
import numpy as np
import sys
from arena import Arena 
import time
import os

def play():
    pygame.init()

    game = Arena()
    game.frozen = True
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    seed = int(round(time.time() * 1000)) % (2 ** 32)
    game.rng = np.random.RandomState(seed)
    game.init()
    while True:
        game.clock.tick_busy_loop(30)
        game.step()
        pygame.display.update()
        if game.game_over() is True:
            #print(game.getGameState(), '\n')
            print("The overall score is {}.".format(game.score))
            break
        #print(game.getGameState(), '\n')

    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == "__main__":
    play()
