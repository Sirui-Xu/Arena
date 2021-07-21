import os, sys
import copy
import argparse
import cv2
import numpy as np
import torch
import pygame
from tqdm import tqdm
from functools import partial

dqgnn_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(os.path.dirname(dqgnn_path))
sys.path.append(root_path)

from arena import Arena, Wrapper
from examples.rl_dqgnn.nn_utils import PointConv, process_state
from examples.env_setting_kwargs import get_env_kwargs_dict

class GNNQEvaluator():
    def __init__(self, env_fn, env_kwargs_dict, qnet, device):
        self.env_fn = env_fn
        self.env_kwargs_dict = env_kwargs_dict
        self.qnet = qnet
        self.device=device

    def act_best(self, state):
        state = copy.deepcopy(state)
        state.batch = torch.zeros(len(state.x)).long()
        state = state.to(self.device)
        with torch.no_grad():
            best_action = self.qnet(state, 1).argmax()
        return best_action.cpu().item()

    def play(self, num_trajs, fps=50, store_video=False, video_path=None):
        # The configuration AX0
        env = self.env_fn(self.env_kwargs_dict)
        env.init()

        scores = []
        for traj_id in tqdm(range(num_trajs)):
            state = env.reset()
            state=process_state(state)
            if store_video:
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                output_fname=os.path.join(video_path,f'{traj_id}.mp4')
                output_movie = cv2.VideoWriter(output_fname,
                                               fourcc, 6, (env.render().shape[0], env.render().shape[1]))
                #print('\nsaving to: ', output_fname)

            for j in range(self.env_kwargs_dict['max_step']):
                if store_video:
                    output_movie.write(env.render())
                action = self.act_best(state)
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
            output_movie.release()
            scores.append(env.score())
        print('num_coins:', self.env_kwargs_dict['num_coins'],
              'score:', np.mean(scores), 'std:', np.std(scores))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--store_video', action="store_true", default=False)
    parser.add_argument('--num_rewards', type=int, default=5)
    parser.add_argument('--num_trajs', type=int, default=20)
    parser.add_argument('--env_setting', type=str, default='AX0')
    args = parser.parse_args()

    model_dir = os.path.dirname(args.model_path)
    video_dir = os.path.join(dqgnn_path, "videos")
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    video_path = os.path.join(dqgnn_path, f"videos/{os.path.basename(model_dir)}")
    if not os.path.exists(video_path):
        os.mkdir(video_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # qnet = PointConv(input_dim=4, pos_dim=4)
    qnet = PointConv(input_dim=8, pos_dim=4)
    qnet.eval()
    state_dict = torch.load(args.model_path)
    qnet.load_state_dict(state_dict)
    qnet = qnet.to(device)

    #os.environ.pop("SDL_VIDEODRIVER")

    env_fn = lambda kwargs_dict: Wrapper(Arena(**kwargs_dict))
    env_kwargs_dict = get_env_kwargs_dict(args.env_setting)
    env_kwargs_dict['num_coins'] = args.num_rewards

    evaluator = GNNQEvaluator(env_fn, env_kwargs_dict, qnet, device)
    evaluator.play(args.num_trajs)