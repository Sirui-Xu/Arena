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

dqgnn_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(os.path.dirname(dqgnn_path))
sys.path.append(root_path)

from arena import Arena, Wrapper
from examples.rl_dqgnn.nn_utils import PointConv, EnvStateProcessor
from examples.env_setting_kwargs import get_env_kwargs_dict

random.seed(2333)
np.random.seed(2333)
torch.manual_seed(2333)

class GNNQEvaluator():
    def __init__(self, env_fn, env_kwargs_dict, qnet, device, eps=0.0):
        self.env_fn = env_fn
        self.env_kwargs_dict = env_kwargs_dict
        self.qnet = qnet
        self.device=device
        self.eps = eps

        self.state_processor = EnvStateProcessor(env_kwargs_dict)

    def act_best(self, state):
        state = copy.deepcopy(state)
        state.batch = torch.zeros(len(state.x)).long()
        state = state.to(self.device)
        with torch.no_grad():
            best_action = self.qnet(state, 1).argmax()
        return best_action.cpu().item()

    def act_eps_best(self, state):
        if random.random() < self.eps:
            return random.choice(np.arange(6))
        state = copy.deepcopy(state)
        state.batch = torch.zeros(len(state.x)).long()
        state = state.to(self.device)
        with torch.no_grad():
            best_action = self.qnet(state, 1).argmax()
        return best_action.cpu().item()

    def evaluate(self, num_trajs, fps=50, store_video=False, video_path=None):
        # The configuration AX0
        env = self.env_fn(self.env_kwargs_dict)
        env.init()

        scores = []
        for traj_id in tqdm(range(num_trajs)):
            state = env.reset()
            state = self.state_processor.process_state(state)
            if store_video:
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                output_fname=os.path.join(video_path,f'{traj_id}.mp4')
                output_movie = cv2.VideoWriter(output_fname,
                                               fourcc, 6, (env.render().shape[0], env.render().shape[1]))
                print('\nsaving to: ', output_fname)

            for j in range(self.env_kwargs_dict['max_step']):
                if store_video:
                    output_movie.write(env.render())
                action = self.act_eps_best(state)
                next_state, reward, done, _ = env.step(action)
                #next_state = process_state(next_state, obj_size=self.env_kwargs_dict['object_size'])
                next_state = self.state_processor.process_state(next_state)
                state = next_state
                if done:
                    if store_video:
                        output_movie.write(env.render())
                    break
            if store_video:
                #print('video stored')
                output_movie.release()
            output_movie.release()
            scores.append(env.score())

            print('reward:', env.score())

        print('num_coins:', self.env_kwargs_dict['num_coins'],
              'score:', np.mean(scores), 'std:', np.std(scores))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--store_video', action="store_true", default=False)
    parser.add_argument('--num_rewards', type=int, default=5)
    parser.add_argument('--num_trajs', type=int, default=20)
    parser.add_argument('--env_setting', type=str, default='legacy')
    args = parser.parse_args()

    model_dir = os.path.dirname(args.model_path)
    video_dir = os.path.join(dqgnn_path, "videos")
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    video_path = os.path.join(dqgnn_path, f"videos/{os.path.basename(model_dir)}")
    if not os.path.exists(video_path):
        os.mkdir(video_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_dir + '/network_kwargs.pkl', 'rb') as f:
        network_kwargs=pickle.load(f)

    qnet = PointConv(**network_kwargs)
    qnet.eval()
    state_dict = torch.load(args.model_path)
    qnet.load_state_dict(state_dict)
    qnet = qnet.to(device)

    #os.environ.pop("SDL_VIDEODRIVER")

    env_fn = lambda kwargs_dict: Wrapper(Arena(**kwargs_dict))
    env_kwargs_dict = get_env_kwargs_dict(args.env_setting)

    env_kwargs_dict['num_coins'] = args.num_rewards
    evaluator = GNNQEvaluator(env_fn, env_kwargs_dict, qnet, device)
    evaluator.evaluate(args.num_trajs, store_video=True, video_path = video_path)