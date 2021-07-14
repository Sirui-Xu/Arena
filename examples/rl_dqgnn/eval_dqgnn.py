import pygame
import numpy as np
import sys
from functools import partial
import os, sys
dqgnn_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(os.path.dirname(dqgnn_path))
sys.path.append(root_path)

from arena import Arena, Wrapper
import os
import cv2
from tqdm import tqdm
from examples.rl_dqgnn.train_dqgnn import PointConv, process_state
import torch
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path')
args=parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#qnet = PointConv(input_dim=4, pos_dim=4)
qnet = PointConv(input_dim=8, pos_dim=4)
qnet.eval()
state_dict = torch.load(args.model_path)
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
    game = Arena(width=64,
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

    env = Wrapper(game)
    t=20
    store_video=True

    scores = []
    for i in tqdm(range(t)):
        state = env.reset()
        state=process_state(state)
        if store_video:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            output_movie = cv2.VideoWriter(os.path.join(dqgnn_path,
                                                        'ARENA_video/{}.mp4'.format(i)),
                                           fourcc, 6, (env.render().shape[0], env.render().shape[1]))
            #print('saving to: ', os.path.join(dqgnn_path,
            #                                            'ARENA_video/{}.mp4'.format(i)))

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
            #print('video stored')
            output_movie.release()
        scores.append(env.score())
    print('num_rewards:', num_rewards, 'score:', np.mean(scores), 'std:', np.std(scores))


if __name__ == "__main__":
    os.environ.pop("SDL_VIDEODRIVER")
    lower2upper = {upper.lower():upper for upper in globals().keys()}
    pygame.init()
    play('ARENA', num_rewards=3)
    #for k in range(1, 14, 2):
    #    play('ARENA', num_rewards=k)