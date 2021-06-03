import pygame
import numpy as np
import os
import sys
from pgle.algorithm.utils import load_game, load_algorithm
from log_utils import Logger
from pgle import PGLE
import argparse
import json
import cv2

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return super(NpEncoder, self).default(obj)

parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, required=True,
                    help='input the game name')
parser.add_argument('--algorithm', type=str, required=True,
                    help='input the algorithm name')
parser.add_argument('--window_size', type=int, default=48, 
                    help='size (W x H) of window')
parser.add_argument('--test_time', type=int, default=1000, 
                    help='test time')
parser.add_argument('--maze_size_list', nargs='+', type=int, default=[7],
                    help='size (W x H) of maze')
parser.add_argument('--num_creeps_list', nargs='+', type=int, default=[5],
                    help='the number of creeps in this game')
parser.add_argument('--frequency_list', nargs='+', type=int, default=[20], 
                    help='Decision frequency')
parser.add_argument('--visualize', action='store_true', 
                    help='display visualization')
parser.add_argument('--store_video', action='store_true', 
                    help='store video')
parser.add_argument('--store_data', action='store_true', 
                    help='store data')
args = parser.parse_args()

game_name = args.game
alg_name = args.algorithm
window_size_initial = args.window_size
maze_size_list = sorted(args.maze_size_list)
num_creeps_list = sorted(args.num_creeps_list)
frequency_list = sorted(args.frequency_list)

os.makedirs('./result/log', exist_ok=True)
logger = Logger(filename=os.path.join('./result/log/', 'log_{}_{}.txt'.format(game_name, alg_name)), mode='a')
sys.stdout = logger

if args.store_video:
    os.makedirs('./result/video', exist_ok=True)

if args.store_data:
    os.makedirs('./result/data', exist_ok=True)

print("==> Test for game {}".format(game_name))
print("==> Using {}".format(alg_name))
data = []
t = args.test_time
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
            env = PGLE(game)
            algorithm = load_algorithm(env, alg_name)

            sum_reward = 0
            for i in range(t):
                state = env.reset()

                if args.store_video:
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    output_movie = cv2.VideoWriter(os.path.join('./result/video', '{}_{}_{}_{}_{}_{}.mp4'.format(game_name, alg_name, maze_size, num_creeps, window_size, i)), fourcc, 6, (env.render().shape[0], env.render().shape[1]))
                
                for j in range(200):
                    if args.store_video:
                        output_movie.write(env.render())
                    if args.visualize:
                        # print("State: {}".format(state))
                        cv2.imshow('PGLE - {}'.format(game_name), env.render())
                        c = cv2.waitKey(0)
                    action = algorithm.exe()   
                    if args.store_data:
                        action_list = [0 for _ in env.actions]
                        action_list[action] = 1
                        data.append({'state':state, 'action':action_list})

                    state, _, game_over, _ = env.step(action)

                    if game_over:
                        if args.store_video:
                            output_movie.write(env.render())
                        if args.visualize:
                            # print("State: {}".format(state))
                            cv2.imshow('PGLE - {}'.format(game_name), env.render())
                            c = cv2.waitKey(0)
                        break
                    
                if args.store_video:
                    output_movie.release()
                sum_reward += env.score()
                print("==> In case {}, the algorithm got {}.".format(i, env.score()))
            print("==> The average performance in this setting is {}".format(sum_reward / t))
print(len(data))
if args.store_data:
    with open(os.path.join('./result/data/', "{}_{}_{}_{}_{}_{}.json".format(game_name, alg_name, maze_size_list, num_creeps_list, frequency_list, window_size)), 'w') as f:
        info = {"game": game_name,
                "algorithm": alg_name,
                "maze_size_list": maze_size_list,
                "num_creeps_list": num_creeps_list,
                "frequency_list": frequency_list,
                "window_size": window_size_initial}
        info["data"] = data
        json.dump(info, f, cls=NpEncoder)