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
from example.rl_dqgnn_zjia.train_dqgnn import PointConv, process_state
import torch
import copy

rl_path=os.path.dirname(__file__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qnet = PointConv()
qnet.eval()
state_dict = torch.load('/home/zjia/Research/Arena/models/num_episode=5000/model.ckpt')
qnet.load_state_dict(state_dict)
qnet = qnet.to(device)


def act_best(state):
    state = copy.deepcopy(state)
    state.batch = torch.zeros(len(state.x)).long()
    state = state.to(device)
    with torch.no_grad():
        best_action = qnet(state, 1).argmax()
    return best_action.cpu().item()

def play(game_name, fps=50, num_rewards=1):
    # The configuration AX0
    game = ARENA(width=64,
                height=64,
                object_size=8,
                num_rewards=num_rewards,
                num_enemies=0,
                num_bombs=0,
                num_projectiles=3,
                num_obstacles=0,
                num_obstacles_groups=1,
                agent_speed=0.25,
                enemy_speed=0.25, # Since there is no enemy, the speed does not matter.
                projectile_speed=0.25,
                bomb_life=100,
                bomb_range=4,
                duration=200)

    env = PGLE(game, 10)
    t=100
    store_video=True

    scores = []
    for i in tqdm(range(t)):
        state = env.reset()
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
        scores.append(env.score())
    print('num_rewards:', num_rewards, 'score:', np.mean(scores), 'std:', np.std(scores))


if __name__ == "__main__":
    os.environ.pop("SDL_VIDEODRIVER")
    lower2upper = {upper.lower():upper for upper in globals().keys()}
    pygame.init()
    for k in range(1, 14, 2):
        play('ARENA', num_rewards=k)
