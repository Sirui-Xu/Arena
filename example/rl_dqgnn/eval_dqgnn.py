import pygame
import numpy as np
import sys
from functools import partial
from pgle.games import BilliardWorld, BilliardWorldMaze, BomberMan, BomberManMaze
from pgle.games import PacWorld, PacWorldMaze, ShootWorld, ShootWorld1d, ShootWorldMaze
from pgle.games import WaterWorld, WaterWorld1d, WaterWorldMaze, ARENA
from pgle import PGLE
import os
import cv2
from tqdm import tqdm
from example.rl_dqgnn.train_dqgnn import PointConv, process_state
import torch
import copy

rl_path=os.path.dirname(__file__)


x_dim, pos_dim = 4,4
qnet_fn = partial(PointConv, input_dim=x_dim, pos_dim=pos_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qnet = qnet_fn()
state_dict = torch.load('/home/yiran/pc_mapping/PyGame-Graph-Based-Learning-Environment/example/rl_dqgnn/~/pc_mapping/PyGame-Graph-Based-Learning-Environment/saved_models/ep1800.pth')
qnet.load_state_dict(state_dict)
qnet = qnet.to(device)

def act_best(state):
    state = copy.deepcopy(state)
    state.batch = torch.zeros(len(state.x)).long()
    state = state.to(device)
    with torch.no_grad():
        #print(qnet(state, 1))

        best_action = qnet(state, 1).argmax()
    return best_action.cpu().item()

def play(game_name, fps=50):
    os.environ.pop("SDL_VIDEODRIVER")
    lower2upper = {upper.lower():upper for upper in globals().keys()}
    pygame.init()
    game = ARENA(width=64,
                height=64,
                object_size=8,
                num_rewards=1,
                num_enemies=0,
                num_bombs=0,
                num_projectiles=3,
                num_obstacles=0,
                num_obstacles_groups=1, # What is this?
                agent_speed=0.25,
                enemy_speed=0.25, # Since there is no enemy, the speed does not matter.
                projectile_speed=0.25,
                bomb_life=100,
                bomb_range=4,
                duration=200)

    env = PGLE(game, 10)
    t=10
    store_video=True
    checkpoints_path='./saved_videos/ARENA'
    sum_reward=0.0
    env.reset()
    init_state=copy.deepcopy(env.game.getGameState())
    #print('init state:', init_state)

    for i in tqdm(range(t)):
        state = env.reset()
        env.game.loadGameState(copy.deepcopy(init_state))
        #print('\nreseted state:', env.game.getGameState())
        #exit()
        state=process_state(state)
        if store_video:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            output_movie = cv2.VideoWriter(os.path.join(rl_path,
                                                        'ARENA_video/{}.mp4'.format(i)),
                                           fourcc, 6, (env.render().shape[0], env.render().shape[1]))

        for j in range(200):
            if store_video:
                output_movie.write(env.render())
            action = act_best(state)
            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            state = next_state
            if done:
                if store_video:
                    output_movie.write(env.render())
                break
        if store_video:
            output_movie.release()
        sum_reward += env.score()

if __name__ == "__main__":
    play('ARENA')
