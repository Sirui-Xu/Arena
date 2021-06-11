import pygame
import numpy as np
import os
import sys
from arena.algorithm.utils import load_algorithm
from arena import Arena
from log_utils import Logger
from arena import Wrapper
import argparse
import json
import cv2
import time

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

parser.add_argument('--algorithm', type=str, required=True,
                    help='input the algorithm name')
parser.add_argument('--rand_seed', type=int, default=24, 
                    help='random seed')
parser.add_argument('--test_time', type=int, default=200, 
                    help='test time')
parser.add_argument('--num_episodes', type=int, default=1000, 
                    help='num of episodes')
parser.add_argument('--width', type=int, default=128, 
                    help='width of the map')
parser.add_argument('--height', type=int, default=128, 
                    help='height of the map')
parser.add_argument('--object_size', type=int, default=8, 
                    help='size of non-obstacle objects')
parser.add_argument('--obstacle_size', type=int, default=16, 
                    help='size of obstacle')
parser.add_argument('--num_coins_list', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                    help='maximum number of coins\' list')
parser.add_argument('--num_enemies_list', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5],
                    help='maximum number of enemies\' list')
parser.add_argument('--num_obstacles_list', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], 
                    help='maximum number of obstacles\' list')
parser.add_argument('--num_bombs', type=int, default=3, 
                    help='maximum number of bombs that can exist in the map')
parser.add_argument('--explosion_max_step', type=int, default=100, 
                    help='number of steps a bomb remains in the map before it explodes')
parser.add_argument('--explosion_radius', type=int, default=128, 
                    help='explosion radius of the bombs')
parser.add_argument('--num_projectiles', type=int, default=3, 
                    help='maximum number of projectiles that can exist in the map')
parser.add_argument('--agent_speed', type=float, default=2.0, 
                    help='travel speed of the agent')
parser.add_argument('--enemy_speed', type=float, default=2.0, 
                    help='travel speed of the enemy')
parser.add_argument('--p_change_direction', type=float, default=0.01, 
                    help='probability each enemy changes direction in each step')
parser.add_argument('--projectile_speed', type=float, default=8.0, 
                    help='travel speed of the projectile')
parser.add_argument('--reward_decay', type=float, default=0.99, 
                    help='rate at which the value of each coin decays')                                        
parser.add_argument('--visualize', action='store_true', 
                    help='display visualization')
parser.add_argument('--store_video', action='store_true', 
                    help='store video')
parser.add_argument('--store_data', action='store_true', 
                    help='store data')
parser.add_argument('--data', type=str, default=None,
                    help='old dataset')
args = parser.parse_args()

if args.data is not None:
    with open(args.data, 'r') as f:
        info = json.load(f)
        alg_name = args.algorithm
        rand_seed = info["rand_seed"]
        test_time = info["test_time"]
        width = info["width"]
        height = info["height"]
        object_size = info["object_size"]
        obstacle_size = info["obstacle_size"]
        num_coins_list = info["num_coins_list"]
        num_enemies_list = info["num_enemies_list"]
        num_bombs = info["num_bombs"]
        explosion_max_step = info["explosion_max_step"]
        explosion_radius = info["explosion_radius"]
        num_projectiles = info["num_projectiles"]
        num_obstacles_list = info["num_obstacles_list"]
        agent_speed = info["agent_speed"]
        enemy_speed = info["enemy_speed"]
        p_change_direction = info["p_change_direction"]
        projectile_speed = info["projectile_speed"]
        reward_decay = info["reward_decay"]
else:
    alg_name = args.algorithm
    rand_seed = args.rand_seed
    test_time = args.test_time
    width = args.width
    height = args.height
    object_size = args.object_size
    obstacle_size = args.obstacle_size
    num_coins_list = sorted(args.num_coins_list)
    num_enemies_list = sorted(args.num_enemies_list)
    num_bombs = args.num_bombs
    explosion_max_step = args.explosion_max_step
    explosion_radius = args.explosion_radius
    num_projectiles = args.num_projectiles
    num_obstacles_list = sorted(args.num_obstacles_list)
    agent_speed = args.agent_speed
    enemy_speed = args.enemy_speed
    p_change_direction = args.p_change_direction
    projectile_speed = args.projectile_speed
    reward_decay = args.reward_decay

visualize = args.visualize
if args.store_video:
    visualize = True
num_episodes = args.num_episodes
asctime = time.asctime(time.localtime(time.time()))
os.makedirs('./results/result_{}/log'.format(asctime), exist_ok=True)
logger = Logger(filename=os.path.join('./results/result_{}/log'.format(asctime), 'log_{}.txt'.format(alg_name)), mode='a')
sys.stdout = logger

if args.store_video:
    os.makedirs('./results/result_{}/video'.format(asctime), exist_ok=True)

if args.store_data:
    os.makedirs('./results/result_{}/data'.format(asctime), exist_ok=True)

print("==> Using {}".format(alg_name))
data = []
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
            for i in range(num_episodes):
                state = env.reset()

                if args.store_video:
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    output_movie = cv2.VideoWriter(os.path.join('./results/result_{}/video'.format(asctime), '{}_{}_{}_{}.mp4'.format(num_coins, num_enemies, num_obstacles, i)), fourcc, 6, (env.render().shape[0], env.render().shape[1]))
                
                for j in range(test_time):
                    if args.store_video:
                        output_movie.write(env.render())
                    if args.visualize:
                        # print("State: {}".format(state))
                        cv2.imshow('Arena', env.render())
                        c = cv2.waitKey(0)
                    action = algorithm.exe()   
                    # print(env.getActionName(action))
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
                            cv2.imshow('Arena', env.render())
                            c = cv2.waitKey(0)
                        break
                    
                if args.store_video:
                    output_movie.release()
                sum_reward += env.score()
                print("==> In case {}, the algorithm got {}.".format(i, env.score()))
            print("==> The average performance in this setting is {}".format(sum_reward / num_episodes))
if args.store_data:
    print(len(data))
    with open(os.path.join('./results/result_{}/data/'.format(asctime), "{}.json".format(alg_name)), 'w') as f:
        info = {"algorithm": alg_name,
                "rand_seed": rand_seed,
                "test_time": test_time,
                "width": width,
                "height": height,
                "object_size": object_size,
                "obstacle_size": obstacle_size,
                "num_coins_list": num_coins_list,
                "num_enemies_list": num_enemies_list,
                "num_bombs": num_bombs,
                "explosion_max_step": explosion_max_step,
                "explosion_radius": explosion_radius,
                "num_projectiles": num_projectiles,
                "num_obstacles_list": num_obstacles_list,
                "agent_speed": agent_speed,
                "enemy_speed": enemy_speed,
                "p_change_direction": p_change_direction,
                "projectile_speed": projectile_speed,
                "reward_decay": reward_decay}
        info["data"] = data
        json.dump(info, f, cls=NpEncoder)