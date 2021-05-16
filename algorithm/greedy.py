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
            if game_over and reward < 0:
                rewards.append(-100)
            else:
                rewards.append(reward)
            self.env.loadEnvState(env_state)
            # print(self.env.getEnvState())
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
            # print(self.env.getActionName(action[0]), self.env.getActionName(action[1]), rewards[-1])
            self.env.loadEnvState(env_state)
            # print(self.env.getEnvState())
        assert len(actions) == len(rewards)
        indexs = [i for i in range(len(actions)) if rewards[i] == max(rewards)]
        index = random.choice(indexs)
        # print(actions[index])
        return actions[index][0]
    
class GreedyCollectV0:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        assert self.name[:5] == "Water" or self.name[:8] == "Billiard"
        assert self.name[-4:] != "Maze" or self.name[-2:] != "1d"
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.actions_name = ["left", "right", "up", "down", "noop"]


    def exe(self):
        env_state = self.env.getEnvState()
        assert env_state["state"]["local"][0]["type"] == "player"
        player_pos = env_state["state"]["local"][0]["position"]
        min_dis = [self.env.game.width + 1, self.env.game.height + 1]
        for info in env_state["state"]["local"]:
            if info["type_index"][1] == 0:
                creep_pos = info["position"]
                dis = [creep_pos[0] - player_pos[0], creep_pos[1] - player_pos[1]]
                if dis[0] * dis[0] + dis[1] * dis[1] < min_dis[0] * min_dis[0] + min_dis[1] * min_dis[1]:
                    min_dis = dis
        
        projection = [d[0] * min_dis[0] + d[1] * min_dis[1] for d in self.directions]
        names = [self.actions_name[i] for i in range(len(projection)) if projection[i] > 0]
        projection = [projection[i] for i in range(len(projection)) if projection[i] > 0]
        assert len(names) <= 2
        if projection[0] / sum(projection) >= random.random():
            return self.env.getActionIndex(names[0])
        else:
            return self.env.getActionIndex(names[1])

class GreedyCollectV1:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        assert self.name[:5] == "Water" or self.name[:8] == "Billiard"
        assert self.name[-4:] != "Maze" or self.name[-2:] != "1d"
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.actions_name = ["left", "right", "up", "down", "noop"]

    def exe(self):
        rewards = []
        env_state = self.env.getEnvState()
        for action_name in self.actions_name:
            _, reward, game_over, _ = self.env.step(self.env.getActionIndex(action_name))
            if game_over and reward < 0:
                rewards.append(-100)
            else:
                rewards.append(reward)
            self.env.loadEnvState(env_state)
            # print(self.env.getEnvState())
        assert env_state["state"]["local"][0]["type"] == "player"
        player_pos = env_state["state"]["local"][0]["position"]
        min_dis = [self.env.game.width + 1, self.env.game.height + 1]
        for info in env_state["state"]["local"]:
            if info["type_index"][1] == 0:
                creep_pos = info["position"]
                dis = [creep_pos[0] - player_pos[0], creep_pos[1] - player_pos[1]]
                if dis[0] * dis[0] + dis[1] * dis[1] < min_dis[0] * min_dis[0] + min_dis[1] * min_dis[1]:
                    min_dis = dis
        projection = [d[0] * min_dis[0] + d[1] * min_dis[1] for d in self.directions]
        names = [self.actions_name[i] for i in range(len(projection)) if projection[i] > 0 and rewards[i] >= -0.5]
        projection = [projection[i] for i in range(len(projection)) if projection[i] > 0 and rewards[i] >= -0.5]
        assert len(names) <= 2
        if len(names) == 0:
            actions = [i for i in range(self.n_action) if rewards[i] == max(rewards)]
            return self.env.getActionIndex(self.actions_name[random.choice(actions)])
        else:
            if projection[0] / sum(projection) >= random.random():
                return self.env.getActionIndex(names[0])
            else:
                return self.env.getActionIndex(names[1])