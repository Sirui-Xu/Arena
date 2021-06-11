import pygame
import numpy as np
import random
import os
import sys
from arena.algorithm.utils import load_algorithm
from utils import Logger, load_model
from dataset import GamePatch
from arena import Arena, Wrapper
import torch
import argparse
import json
import cv2
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints_path', type=str,
                    help='checkpoint path')
parser.add_argument('--visualize', action='store_true', 
                    help='display visualization')
parser.add_argument('--store_video', action='store_true', 
                    help='store video')
parser.add_argument('--resume_epoch', type=int, default=0,
                    help='resume epoch for the saved model')
parser.add_argument('--num_coins_list', nargs='+', type=int, default=[1, 3, 5, 7, 9],
                    help='maximum number of coins\' list')
parser.add_argument('--num_enemies_list', nargs='+', type=int, default=[1, 3, 5, 7, 9],
                    help='maximum number of enemies\' list')
parser.add_argument('--num_obstacles_list', nargs='+', type=int, default=[2, 6, 10, 14, 18], 
                    help='maximum number of obstacles\' list')
parser.add_argument('--num_episodes', type=int, default=100, 
                    help='num of episodes')
parser.add_argument('--rand_seed', type=int, default=24, 
                    help='random seed')
args = parser.parse_args()

with open(os.path.join(args.checkpoints_path, 'info.json'), 'r') as f:
    info = json.load(f)

alg_name = args.algorithm
rand_seed = args.rand_seed
test_time = info["test_time"]
width = info["width"]
height = info["height"]
object_size = info["object_size"]
obstacle_size = info["obstacle_size"]
num_coins_list = sorted(args.num_coins_list)
num_enemies_list = sorted(args.num_enemies_list)
num_bombs = info["num_bombs"]
explosion_max_step = info["explosion_max_step"]
explosion_radius = info["explosion_radius"]
num_projectiles = info["num_projectiles"]
num_obstacles_list = sorted(num_obstacles_list)
agent_speed = info["agent_speed"]
enemy_speed = info["enemy_speed"]
p_change_direction = info["p_change_direction"]
projectile_speed = info["projectile_speed"]
reward_decay = info["reward_decay"]
model = info["model"]
model_info = info["model_info"]
CROSS_ENTROPY = True

visualize = args.visualize
if args.store_video:
    visualize = True
num_episodes = args.num_episodes
asctime = time.asctime(time.localtime(time.time()))

os.makedirs(os.path.join(args.checkpoints_path, './results/result_{}/log'.format(asctime)), exist_ok=True)
logger = Logger(filename=os.path.join(args.checkpoints_path, './results/result_{}/log'.format(asctime), 'log.txt'), mode='a')
sys.stdout = logger

if args.store_video:
    os.makedirs(os.path.join(args.checkpoints_path, './results/result_{}/video'.format(asctime)), exist_ok=True)

policy_net = load_model(model, model_info).to(device)
resume_epoch = args.resume_epoch
if resume_epoch > 0:
    save_path = os.path.join(args.checkpoints_path, 'epoch_{}'.format(resume_epoch))
else:
    save_path = os.path.join(args.checkpoints_path, 'best_model')
save_state = torch.load(save_path)
policy_net.load_state_dict(save_state['policy_net'])
policy_net.eval()
print("==> Test setting:{}".format(info))
for num_coins in num_coins_list:
    for num_enemies in num_enemies_list:
        for num_obstacles in num_obstacles_list:
            print("============== test case ==============")
            print("==> num_coins: {}".format(num_coins))
            print("==> num_enemies: {}".format(num_enemies))
            print("==> num_obstacles: {}".format(num_obstacles))

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
                         projectile_speed=projectile_speed,
                         visualize=visualize,
                         reward_decay=reward_decay)

            env = Wrapper(game, rng=rand_seed)
            algorithm = load_algorithm(env, alg_name)

            sum_reward = 0
            sum_losses = 0
            for i in tqdm(range(num_episodes)):
                state = env.reset()
                if args.store_video:
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    output_movie = cv2.VideoWriter(os.path.join(args.checkpoints_path, './results/result_{}/video'.format(asctime), '{}_{}_{}_{}.mp4'.format(num_coins, num_enemies, num_obstacles, i)), fourcc, 6, (env.render().shape[0], env.render().shape[1]))
                losses = 0
                for j in range(test_time)):
                    if args.store_video:
                        output_movie.write(env.render())
                    if args.visualize:
                        print("State: {}".format(state))
                        cv2.imshow('Arena', env.render())
                        c = cv2.waitKey(0)
                    action = algorithm.exe()
                    action_list = [0 for _ in env.actions]
                    action_list[action] = 1
                    data = [{'state':state, 'action':action_list}]
                    data_loader = DataLoader(GamePatch(data),
                                             batch_size=1
                                            )
                    for data_torch in data_loader:
                        with torch.no_grad():
                            data_torch.to(device)
                            outputs = policy_net(data_torch)
                            q = outputs['q']
                            _action = q.argmax(-1)[0]
                            if CROSS_ENTROPY:
                                loss = F.cross_entropy(q, data_torch.y.argmax(-1), reduction='mean')  # (b,)               
                            else:
                                loss = F.mse_loss(q, data_torch.y, reduction='mean')
                            losses += loss
                    if args.visualize:
                        print("==> action is {}.".format(env.getActionName(_action)))
                    state, _, game_over, _ = env.step(_action)
                    if game_over:
                        if args.store_video:
                            output_movie.write(env.render())
                        if args.visualize:
                            print("State: {}".format(state))
                            cv2.imshow('Arena', env.render())
                            c = cv2.waitKey(0)
                        break
                if args.store_video:
                    output_movie.release()
                sum_reward += env.score()
                # print("==> In case {}, the model got {}.".format(i, env.score()))
                # print("==> The disparity (loss) between teacher policy and student model: {}.".format(losses / j))
                sum_losses += losses / j
            print("==> The average disparity (loss) between teacher policy and student model: {}.".format(sum_losses / test_time))
            print("==> The average performance in this setting is {}".format(sum_reward / test_time))