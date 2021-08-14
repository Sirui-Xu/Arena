import os, sys
import copy
import argparse
import random
import cv2
import pickle
import numpy as np
import torch
import pygame
from tqdm import tqdm
from functools import partial
import json

dqgnn_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(os.path.dirname(dqgnn_path))
sys.path.append(root_path)

from arena import Arena, Wrapper
from examples.rl_dqgnn.nn_utils import EnvStateProcessor, get_nn_func, GraphObservationEnvWrapper
from examples.env_setting_kwargs import get_env_kwargs_dict
from torch_geometric.data import Batch


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

class GNNQEvaluator():
    def __init__(self, model_path, nn_name, env_setting, num_trajs=100,
                 store_video=False, store_traj=False, only_success_traj=False, fps=50, eps=0.0):

        model_dir = os.path.dirname(model_path)
        video_dir = os.path.join(dqgnn_path, "videos")
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        video_path = os.path.join(dqgnn_path, f"videos/{os.path.basename(model_dir)}")
        if not os.path.exists(video_path):
            os.mkdir(video_path)
        if store_traj:
            traj_path_suffix='_success' if only_success_traj else ''
            self.traj_path = os.path.join(model_dir, f"traj{traj_path_suffix}.json")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(model_dir + '/network_kwargs.pkl', 'rb') as f:
            network_kwargs = pickle.load(f)

        nn_func = get_nn_func(nn_name)
        qnet = nn_func(**network_kwargs)
        qnet.eval()
        state_dict = torch.load(model_path)
        qnet.load_state_dict(state_dict)
        qnet = qnet.to(device)

        # os.environ.pop("SDL_VIDEODRIVER")

        #env_fn = lambda kwargs_dict: Wrapper(Arena(**kwargs_dict))
        env_fn = lambda env_kwargs: GraphObservationEnvWrapper(Arena, env_kwargs)
        env_kwargs_dict = get_env_kwargs_dict(env_setting)

        self.env_fn = env_fn
        self.env_kwargs_dict = env_kwargs_dict
        self.qnet = qnet
        self.device=device
        self.eps = eps
        self.video_path=video_path
        self.num_trajs = num_trajs
        self.store_video = store_video
        self.store_traj = store_traj
        self.only_success_traj=only_success_traj
        self.fps = fps

    def update_num_coins(self, num_coins):
        self.env_kwargs_dict['num_coins'] = num_coins

    def act_best(self, state):
        state = copy.deepcopy(state)
        state.batch = torch.zeros(len(state.x)).long()
        state = state.to(self.device)
        with torch.no_grad():
            best_action = self.qnet(state).argmax()
        return best_action.cpu().item()

    def act_eps_best(self, state):
        if random.random() < self.eps:
            return random.choice(np.arange(6))
        #state = copy.deepcopy(state)
        #state.batch = torch.zeros(len(state.x)).long()
        #state = state.to(self.device)
        state=Batch.from_data_list([state]).to(self.device)
        with torch.no_grad():
            best_action = self.qnet(state).argmax()
        return best_action.cpu().item()

    def evaluate(self, num_coins_min=None, num_coins_max=None):
        if num_coins_min is None:
            num_coins_min, num_coins_max = self.env_kwargs_dict['num_coins']

        trajs = []
        for num_coins in range(num_coins_min, num_coins_max+1):
            self.update_num_coins(num_coins)
            env = GraphObservationEnvWrapper(Arena, self.env_kwargs_dict)
            env.init()
            scores = []
            for traj_id in tqdm(range(self.num_trajs)):
                state = env.reset()
                state_raw = env._state_raw
                if self.store_video:
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    output_fname=os.path.join(self.video_path,f'{traj_id}.mp4')
                    output_movie = cv2.VideoWriter(output_fname,
                                                   fourcc, 6, (env.render().shape[0], env.render().shape[1]))
                    #print('\nsaving to: ', output_fname)
                current_traj = []
                for j in range(self.env_kwargs_dict['max_step']):
                    if self.store_video:
                        output_movie.write(env.render())
                    action = self.act_eps_best(state)
                    if self.store_traj:
                        action_list = [0 for _ in env.actions]
                        action_list[action] = 1
                        current_traj.append({'state': state_raw, 'action': action_list})
                    next_state, reward, done, _ = env.step(action)

                    next_state_raw = env._state_raw
                    state_raw = next_state_raw
                    state = next_state
                    if done:
                        if self.store_video:
                            output_movie.write(env.render())
                        break
                if self.store_video:
                    output_movie.release()
                if self.store_traj:
                    if not self.only_success_traj or int(env.score())==num_coins:
                        #print(f'traj saved, reward {env.score()}')
                        trajs.extend(current_traj)

                scores.append(env.score())

                print('reward:', env.score())

            print('num_coins:', self.env_kwargs_dict['num_coins'],
                  'score:', np.mean(scores), 'std:', np.std(scores))

        if self.store_traj:
            print(f'saving (s,a) dataset of size {len(trajs)}')
            with open(self.traj_path, 'w') as f:
                info = {"algorithm": "DoubleDQN",
                        "rand_seed": 0,
                        "test_time": 200,
                        "width": self.env_kwargs_dict['width'],
                        "height": self.env_kwargs_dict['height'],
                        "object_size": self.env_kwargs_dict['object_size'],
                        "obstacle_size": self.env_kwargs_dict['obstacle_size'],
                        "num_coins_list": [num_coins_min,num_coins_max],
                        "num_enemies_list": [0],
                        "num_bombs": 0,
                        "explosion_max_step": self.env_kwargs_dict['explosion_max_step'],
                        "explosion_radius": self.env_kwargs_dict['explosion_radius'],
                        "num_projectiles": self.env_kwargs_dict['num_projectiles'],
                        "num_obstacles_list": [0],
                        "agent_speed": self.env_kwargs_dict['agent_speed'],
                        "enemy_speed": self.env_kwargs_dict['enemy_speed'],
                        "p_change_direction": self.env_kwargs_dict['p_change_direction'],
                        "projectile_speed": self.env_kwargs_dict['projectile_speed'],
                        "reward_decay": self.env_kwargs_dict['reward_decay']}
                info["data"] = trajs
                json.dump(info, f, cls=NpEncoder)


        return {'score': np.mean(scores), 'std:': np.std(scores)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--store_video', action="store_true", default=False)
    parser.add_argument('--store_traj', action="store_true", default=False)
    parser.add_argument('--only_success_traj', action="store_true", default=False)
    parser.add_argument('--num_rewards', type=int, default=5)
    parser.add_argument('--num_trajs', type=int, default=100)
    parser.add_argument('--env_setting', type=str, default='legacy')
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--nn_name', type=str, default="PointConv")
    args = parser.parse_args()

    random.seed(2333)
    np.random.seed(2333)
    torch.manual_seed(2333)

    evaluator = GNNQEvaluator(model_path=args.model_path, nn_name=args.nn_name,
                              env_setting=args.env_setting, num_trajs = args.num_trajs,
                              store_video=args.store_video, store_traj=args.store_traj,
                              only_success_traj=args.only_success_traj, eps=args.eps)
    evaluator.update_num_coins(args.num_rewards)
    eval_result = evaluator.evaluate(args.num_rewards, args.num_rewards)
    #eval_result = evaluator.evaluate(1,5)