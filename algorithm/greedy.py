import pygame
import random
import sys
import numpy as np
import copy

class OneStep:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env

    def exe(self):
        rewards = []
        env_state = self.env.getEnvState()
        for action in range(self.n_action):
            _, reward, game_over, _ = self.env.step(action)
            if game_over:
                rewards.append(-100)
            else:
                rewards.append(reward)
            self.env.loadEnvState(env_state)
        actions = [i for i in range(self.n_action) if rewards[i] == max(rewards)]
        return random.choice(actions)

class TwoStep:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env

    def exe(self):
        rewards = []
        env_state = copy.deepcopy(self.env.getEnvState())
        actions = [[i, j] for i in range(self.n_action) for j in range(self.n_action)]
        for action in actions:
            sum_reward = 0
            for i in range(len(action)):
                _, reward, game_over, _ = self.env.step(action[i])
                if game_over and reward < 0:
                    rewards.append(-100 + i)
                    break
                else:
                    sum_reward += reward * (1 - 0.1*i)
            if not (game_over and reward < 0):
                rewards.append(sum_reward)
            print(self.env.getActionName(action[0]), self.env.getActionName(action[1]), rewards[-1])
            self.env.loadEnvState(env_state)
            print(self.env.getEnvState())
        assert len(actions) == len(rewards)
        indexs = [i for i in range(len(actions)) if rewards[i] == max(rewards)]
        index = random.choice(indexs)
        print(actions[index])
        return actions[index][0]
    
class GreedyWaterWorld:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env

    def exe(self):
        rewards = []
        env_state = self.env.getEnvState()
        for action in range(self.n_action):
            _, reward, game_over, _ = self.env.step(action)
            if game_over:
                rewards.append(-100)
            else:
                rewards.append(reward)
            self.env.loadEnvState(env_state)
        actions = [i for i in range(self.n_action) if rewards[i] == max(rewards)]
        return random.choice(actions)