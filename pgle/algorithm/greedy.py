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
        assert self.name[:5] == "Water" or self.name[:8] == "Billiard" or self.name[:3] == "Pac"
        assert self.name[-4:] != "Maze" or self.name[-2:] != "1d"
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.actions_name = ["left", "right", "up", "down", "noop"]


    def exe(self):
        env_state = self.env.getEnvState()
        assert env_state["state"]["local"][0]["type"] == "player"
        player_pos = env_state["state"]["local"][0]["position"]
        min_dis = [self.env.game.width + 1, self.env.game.height + 1]
        for info in env_state["state"]["local"]:
            if info["type"] == "creep" and info["_type"] == "GOOD":
                creep_pos = info["position"]
                dis = [creep_pos[0] - player_pos[0], creep_pos[1] - player_pos[1]]
                if dis[0] * dis[0] + dis[1] * dis[1] < min_dis[0] * min_dis[0] + min_dis[1] * min_dis[1]:
                    min_dis = dis
        
        projection = [d[0] * min_dis[0] + d[1] * min_dis[1] for d in self.directions]
        names = [self.actions_name[i] for i in range(len(projection)) if projection[i] > 0]
        projection = [projection[i] for i in range(len(projection)) if projection[i] > 0]
        assert len(names) <= 2
        if projection[0] / sum(projection) >= 0.5:
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
            if info["type"] == "creep" and info["_type"] == "GOOD":
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
            if projection[0] / sum(projection) >= 0.5:
                return self.env.getActionIndex(names[0])
            else:
                return self.env.getActionIndex(names[1])


class GreedyCollectV2:
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
        if max(rewards) > 0:
            actions = [i for i in range(self.n_action) if rewards[i] == max(rewards)]
            return self.env.getActionIndex(self.actions_name[random.choice(actions)])
        assert env_state["state"]["local"][0]["type"] == "player"
        player_pos = env_state["state"]["local"][0]["position"]
        min_dis = [self.env.game.width + 1, self.env.game.height + 1]
        for info in env_state["state"]["local"]:
            if info["type"] == "creep" and info["_type"] == "GOOD":
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
            if projection[0] / sum(projection) >= 0.5:
                return self.env.getActionIndex(names[0])
            else:
                return self.env.getActionIndex(names[1])

class GreedyCollectMax:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        assert self.name[:3] == "Pac"
        assert self.name[-4:] != "Maze" or self.name[-2:] != "1d"
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.actions_name = ["left", "right", "up", "down", "noop"]


    def exe(self):
        env_state = self.env.getEnvState()
        assert env_state["state"]["local"][0]["type"] == "player"
        player_pos = env_state["state"]["local"][0]["position"]
        min_dis = [self.env.game.width + 1, self.env.game.height + 1]
        max_value = 0
        for info in env_state["state"]["local"]:
            if info["type"] == "creep":
                creep_pos = info["position"]
                dis = [creep_pos[0] - player_pos[0], creep_pos[1] - player_pos[1]]
                if info["type_index"][1] > max_value:
                    min_dis = dis
                    max_value = info["type_index"][1]
        projection = [d[0] * min_dis[0] + d[1] * min_dis[1] for d in self.directions]
        names = [self.actions_name[i] for i in range(len(projection)) if projection[i] > 0]
        projection = [projection[i] for i in range(len(projection)) if projection[i] > 0]
        assert len(names) <= 2
        if projection[0] / sum(projection) >= 0.5:
            return self.env.getActionIndex(names[0])
        else:
            return self.env.getActionIndex(names[1])


class GreedyArena:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        self.name == "ARENA"
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.actions_name = ["left", "right", "up", "down", "noop", "shoot", "fire"]


    def exe(self):
        env_state = self.env.getEnvState()
        self.map = np.zeros(tuple(env_state["state"]["global"]["shape"]))
        width, height = env_state["state"]["global"]["shape"][0], env_state["state"]["global"]["shape"][1]
        min_dis_reward_index = 0
        min_dis_reward = width + height
        threat = []
        agent_pos = [0, 0]
        agent_radius = 0
        enemy_radius = 0
        reward_radius = 0
        agent_speed = 0
        agent_vel = [0, 0]
        agent_box = [0, 0, 0, 0]
        for i, info in enumerate(env_state["state"]["local"]):
            if info["type"] == "agent":
                agent_pos = info["position"]
                agent_radius = info["radius"]
                agent_speed = info["speed"]
                agent_box = info["box"]
                agent_vel = info["velocity"]
            if info["type"] == "reward":
                reward_pos = info["position"]
                reward_radius = info["radius"]                
                reward_box = info["box"]
                self.map[reward_box[0]:reward_box[2], reward_box[1]:reward_box[3]] = info["type_index"][0]
                dis = abs(reward_pos[0] - agent_pos[0]) + abs(reward_pos[1] - agent_pos[1])
                if dis < min_dis_reward:
                    min_dis_reward = dis
                    min_dis_reward_index = i   

            if info["type"] == "obstacle":       
                obstacle_box = info["box"]
                self.map[obstacle_box[0]:obstacle_box[2], obstacle_box[1]:obstacle_box[3]] = info["type_index"][0]

            if info["type"] == "blast":
                blast_box = info["box"]
                self.map[blast_box[0]:blast_box[2], blast_box[1]:blast_box[3]] = info["type_index"][0]


            if info["type"] == "enemy":
                enemy_pos = info["position"]
                enemy_radius = info["radius"]
                enemy_speed = info["speed"]
                enemy_pos_new = [enemy_pos[0] + info["velocity"][0], enemy_pos[1] + info["velocity"][1]]
                dis = abs(enemy_pos[0] - agent_pos[0]) + abs(enemy_pos[1] - agent_pos[1])
                if enemy_pos_new[0] < enemy_radius or enemy_pos_new[0] > width - enemy_radius or enemy_pos_new[1] < enemy_radius or enemy_pos_new[1] > height - enemy_radius:
                    enemy_speed = info["speed"]
                    for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if direction[0] * info["velocity"][0] + direction[1] * info["velocity"][1] > 0:
                            continue
                        enemy_pos_new = [enemy_pos[0] + enemy_speed * direction[0], enemy_pos[1] + enemy_speed * direction[1]]
                        enemy_box_new = [int(enemy_pos_new[0] - enemy_radius - 1), int(enemy_pos_new[1] - enemy_radius - 1),
                                        int(enemy_pos_new[0] + enemy_radius + 1), int(enemy_pos_new[1] + enemy_radius + 1)]
                        self.map[enemy_box_new[0]:enemy_box_new[2], enemy_box_new[1]:enemy_box_new[3]] = info["type_index"][0]
                        new_dis = abs(enemy_pos_new[0] - agent_pos[0]) + abs(enemy_pos_new[1] - agent_pos[1])
                        new_time = (max(abs(enemy_pos_new[0] - agent_pos[0]) - agent_radius - enemy_radius, 0)
                                 + max(abs(enemy_pos_new[1] - agent_pos[1]) - agent_radius - enemy_radius, 0)) / agent_speed
                        if new_dis < dis and new_time <= 4:
                            threat.append((1 / 3, enemy_pos_new))         
                else:        
                    enemy_box_new = [int(enemy_pos_new[0] - enemy_radius - 1), int(enemy_pos_new[1] - enemy_radius - 1),
                                    int(enemy_pos_new[0] + enemy_radius + 1), int(enemy_pos_new[1] + enemy_radius + 1)]
                    self.map[enemy_box_new[0]:enemy_box_new[2], enemy_box_new[1]:enemy_box_new[3]] = info["type_index"][0]
                    new_dis = abs(enemy_pos_new[0] - agent_pos[0]) + abs(enemy_pos_new[1] - agent_pos[1])
                    new_time = (max(abs(enemy_pos_new[0] - agent_pos[0]) - agent_radius - enemy_radius, 0)
                             + max(abs(enemy_pos_new[1] - agent_pos[1]) - agent_radius - enemy_radius, 0)) / agent_speed
                    if new_dis < dis and new_dis <= 4:
                        threat.append((1, enemy_pos_new))


            if info["type"] == "bomb":
                bomb_pos = info["position"]
                bomb_life = info["type_index"][2]
                bomb_range = info["type_index"][3]
                if agent_speed * bomb_life <= bomb_range: 
                    bomb_box = [int(bomb_pos[0] - bomb_range + agent_speed * bomb_life - 1),
                                int(bomb_pos[1] - bomb_range + agent_speed * bomb_life - 1),
                                int(bomb_pos[0] + bomb_range - agent_speed * bomb_life + 1),
                                int(bomb_pos[1] + bomb_range - agent_speed * bomb_life + 1)]
                    self.map[bomb_box[0]:bomb_box[2], bomb_box[1]:bomb_box[3]] = info["type_index"][0]
            
            if info["type"] == "projectile":
                pass

        heuristics = []
        choice = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        random.shuffle(choice)
        for direction in choice:
            agent_pos_new = [agent_pos[0] + agent_speed * direction[0], agent_pos[1] + agent_speed * direction[1]]
            agent_box_new = [int(agent_pos_new[0] - agent_radius), int(agent_pos_new[1] - agent_radius),
                            int(agent_pos_new[0] + agent_radius), int(agent_pos_new[1] + agent_radius)]

            if np.sum(self.map[agent_box_new[0]:agent_box_new[2], agent_box_new[1]:agent_box_new[3]] > 2) > 0:
                heuristics.append(1000)
                continue
            
            reward_info = env_state["state"]["local"][min_dis_reward_index]
            heuristic = (max((abs(agent_pos_new[0] - reward_info["position"][0]) - agent_radius - enemy_radius), 0)
                      + max((abs(agent_pos_new[1] - reward_info["position"][1]) - agent_radius - enemy_radius), 0)) / agent_speed
            for enemy_info in threat:
                c, enemy_pos_new = enemy_info
                time = ((max((abs(agent_pos_new[0] - enemy_pos_new[0]) - agent_radius - enemy_radius), 0)
                            + max((abs(agent_pos_new[1] - enemy_pos_new[1]) - agent_radius - enemy_radius), 0)) / agent_speed)
                if time > 0:
                    heuristic += 1 / time
                else:
                    heuristic += 100
            heuristics.append(heuristic)
        
        move_action_index = self.directions.index(choice[heuristics.index(min(heuristics))])
        if self.actions_name[move_action_index] == "noop":
            if random.random() > 1 / (env_state["state"]["global"]["projectiles_left"] + 1):
                return self.env.getActionIndex("shoot")
            else:
                return self.env.getActionIndex("noop")
        
        if agent_vel[0] * self.directions[move_action_index][0] + agent_vel[1] * self.directions[move_action_index][1] > 0:
            t = 1
            pos = [agent_pos[0], agent_pos[1]]
            while True:
                pos = [pos[0] + agent_vel[0], pos[1] + agent_vel[1]]
                if agent_vel[0] != 0:
                    dis = abs(pos[0] - env_state["state"]["local"][min_dis_reward_index]["position"][0])
                else:
                    dis = abs(pos[1] - env_state["state"]["local"][min_dis_reward_index]["position"][1])
                
                if pos[0] > width - agent_radius or pos[0] < agent_radius or pos[1] > height - agent_radius or pos[1] < agent_radius:
                    return self.env.getActionIndex(self.actions_name[move_action_index])

                box = [int(pos[0] - agent_radius), int(pos[1] - agent_radius),
                       int(pos[0] + agent_radius), int(pos[1] + agent_radius)]
                if np.sum(self.map[box[0]:box[2], box[1]:box[3]] == 2) > 0:
                    if random.random() < 1 / t:
                        return self.env.getActionIndex("shoot")
                    else:
                        return self.env.getActionIndex(self.actions_name[move_action_index])

                if dis < agent_radius + enemy_radius:
                    return self.env.getActionIndex(self.actions_name[move_action_index])
                t += 1
                if t > 100:
                    break
        else:
            return self.env.getActionIndex(self.actions_name[move_action_index])