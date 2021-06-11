import pygame
import numpy as np
import sys
import time
from .game import Arena
import os
from pygame.constants import K_TAB

def play():
    os.environ.pop("SDL_VIDEODRIVER")
    pygame.init()

    game = Arena()
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    seed = int(round(time.time() * 1000)) % (2 ** 32)
    game.rng = np.random.RandomState(seed)
    game.init()
    pygame.display.update()
    frozen = True
    while frozen:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key == K_TAB:
                    frozen = False
    while True:
        dt = game.clock.tick_busy_loop(20)
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
