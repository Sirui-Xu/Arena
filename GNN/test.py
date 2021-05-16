import pygame
import numpy as np
import random
import os
import sys
sys.path.append('../algorithm')
sys.path.append('..')
from algorithm.utils import load_game, load_algorithm, NpEncoder
from GNN.utils import Logger, load_model
from dataset import GamePatch
from pgle import PGLE
import torch
import argparse
import json
import cv2
import torch.nn.functional as F
from torch_geometric.data import DataLoader

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
args = parser.parse_args()

with open(os.path.join(args.checkpoints_path, 'info.json'), 'r') as f:
    info = json.load(f)

game_name = info["game"]
alg_name = info["algorithm"]
window_size_initial = info["window_size"]
maze_size_list = info["maze_size_list"]
if game_name[:4] == "Maze":
    maze_size_list.extend([2*maze_size for maze_size in maze_size_list])
    maze_size_list = set(maze_size_list)
    info["maze_size_list"] = maze_size_list
num_creeps_list = info["num_creeps_list"]
num_creeps_list.extend([2*num_creeps for num_creeps in num_creeps_list])
num_creeps_list = set(num_creeps_list)
info["num_creeps_list"] = num_creeps_list
frequency_list = info["frequency_list"]
frequency_list.extend([frequency_list[i] + (frequency_list[i + 1] - frequency_list[i]) * random.random()  for i in range(len(frequency_list) - 1)])
model = info["model"]
model_info = info["model_info"]
CROSS_ENTROPY = True

os.makedirs(os.path.join(args.checkpoints_path, './result/log'), exist_ok=True)
logger = Logger(filename=os.path.join(args.checkpoints_path, './result/log/', 'log.txt'), mode='a')
sys.stdout = logger

if args.store_video:
    os.makedirs(os.path.join(args.checkpoints_path, './result/video'), exist_ok=True)

policy_net = load_model(model, model_info).to(device)
resume_epoch = args.resume_epoch
save_path = os.path.join(args.checkpoints_path, 'epoch_{}'.format(resume_epoch))
save_state = torch.load(save_path)
policy_net.load_state_dict(save_state['policy_net'])

print("==> Test setting:{}".format(info))
t = 100
for frequency in frequency_list:
    for maze_size in maze_size_list:
        for num_creeps in num_creeps_list:
            window_size = int(maze_size * window_size_initial / maze_size_list[0]) + 1
            print("============== test case ==============")
            print("==> window size: {}".format(window_size))
            print("==> maze size: {}".format(maze_size))
            print("==> number of creeps: {}".format(num_creeps))
            print("==> decision frequency: {}".format(frequency))
            game = load_game(game_name, window_size, maze_size, num_creeps, frequency)
            env = PGLE(game, 10)
            algorithm = load_algorithm(env, alg_name)
            sum_reward = 0
            sum_losses = 0
            for i in range(t):
                state = env.reset()
                if args.store_video:
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    output_movie = cv2.VideoWriter(os.path.join(args.checkpoints_path, './result/video', '{}_{}_{}_{}_{}_{}.mp4'.format(game_name, alg_name, maze_size, num_creeps, window_size, i)), fourcc, 6, (env.render().shape[0], env.render().shape[1]))
                losses = 0
                for j in range(200):
                    if args.store_video:
                        img = env.render()
                        img = np.rot90(img, 1)
                        img = img[::-1, :, :]
                        output_movie.write(img)
                    if args.visualize:
                        print("State: {}".format(state))
                        img = env.render()
                        img = np.rot90(img, 1)
                        img = img[::-1, :, :]
                        cv2.imshow('PGLE - {}'.format(game_name), img)
                        c = cv2.waitKey(0)
                    action = algorithm.exe()
                    if args.visualize:
                        print("==> action is {}.".format(env.getActionName(action)))
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

                    state, _, game_over, _ = env.step(_action)
                    if game_over:
                        if args.store_video:
                            img = env.render()
                            img = np.rot90(img, 1)
                            img = img[::-1, :, :]
                            output_movie.write(img)
                        if args.visualize:
                            print("State: {}".format(state))
                            img = env.render()
                            img = np.rot90(img, 1)
                            img = img[::-1, :, :]
                            cv2.imshow('PGLE - {}'.format(game_name), img)
                            c = cv2.waitKey(0)
                        break
                if args.store_video:
                    output_movie.release()
                sum_reward += env.score()
                print("==> In case {}, the model got {}.".format(i, env.score()))
                print("==> The disparity (loss) between teacher policy and student model: {}.".format(losses / j))
                sum_losses += losses / j
            print("==> The average disparity (loss) between teacher policy and student model: {}.".format(sum_losses / t))
            print("==> The average performance in this setting is {}".format(sum_reward / t))