import pygame
import numpy as np
import sys
import time
from .game import Arena
import os
from pygame.constants import K_TAB

def play(width=1280,
         height=720,
         object_size=32,
         obstacle_size=40,
         num_coins=50,
         num_enemies=50,
         num_bombs=3,
         explosion_max_step=100,
         explosion_radius=128,
         num_projectiles=3,
         num_obstacles=200,
         agent_speed=8,
         enemy_speed=0,
         p_change_direction=0.01,
         projectile_speed=32):

    os.environ.pop("SDL_VIDEODRIVER")
    pygame.init()

    game = Arena(width=width,
                 height=height,
                 object_size=object_size,
                 obstacle_size=obstacle_size,
                 num_coins=num_coins,
                 num_enemies=num_enemies,
                 num_bombs=num_bombs,
                 explosion_max_step=explosion_max_step,
                 explosion_radius=explosion_radius,
                 num_projectiles=num_projectiles,
                 num_obstacles=num_obstacles,
                 agent_speed=agent_speed,
                 enemy_speed=enemy_speed,
                 p_change_direction=p_change_direction,
                 projectile_speed=projectile_speed)

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
            # print(game.getGameState(), '\n')
            print("The overall score is {}.".format(game.score))
            break
        # print(game.getGameState(), '\n')

    os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == "__main__":
    play()
