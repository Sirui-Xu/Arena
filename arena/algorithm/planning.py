import pygame
import random
import sys
import numpy as np

def circle2box(pos, radius):
    return (pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius)

def in_box(point, box):
    return point[0] <= box[2] and point[0] >= box[0] and point[1] <= box[3] and point[1] >= box[1] 

class Planning:
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
        projectile_directions = []
        projectile_speed = 0
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
                enemy_radius = 1.2 * info["radius"]
                enemy_speed = info["speed"]
                enemy_pos_new = [enemy_pos[0] + info["velocity"][0], enemy_pos[1] + info["velocity"][1]]
                dis = abs(enemy_pos[0] - agent_pos[0]) + abs(enemy_pos[1] - agent_pos[1])
                if enemy_pos_new[0] < enemy_radius or enemy_pos_new[0] > width - enemy_radius or enemy_pos_new[1] < enemy_radius or enemy_pos_new[1] > height - enemy_radius:
                    enemy_speed = info["speed"]
                    for direction in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                        if direction[0] * info["velocity"][0] + direction[1] * info["velocity"][1] > 0:
                            continue
                        enemy_pos_new = [enemy_pos[0] + enemy_speed * direction[0], enemy_pos[1] + enemy_speed * direction[1]]
                        enemy_box_new = [int(enemy_pos_new[0] - enemy_radius - 1), int(enemy_pos_new[1] - enemy_radius - 1),
                                        int(enemy_pos_new[0] + enemy_radius + 1), int(enemy_pos_new[1] + enemy_radius + 1)]
                        self.map[enemy_box_new[0]:enemy_box_new[2], enemy_box_new[1]:enemy_box_new[3]] = info["type_index"][0]
                        new_dis = abs(enemy_pos_new[0] - agent_pos[0]) + abs(enemy_pos_new[1] - agent_pos[1])
                        new_time = (max(abs(enemy_pos_new[0] - agent_pos[0]) - agent_radius - enemy_radius, 0)
                                 + max(abs(enemy_pos_new[1] - agent_pos[1]) - agent_radius - enemy_radius, 0)) / agent_speed
                        if new_dis < dis and new_time <= 6:
                            threat.append([1 / 3, enemy_pos_new, direction])         
                else:        
                    enemy_box_new = [int(enemy_pos_new[0] - enemy_radius - 1), int(enemy_pos_new[1] - enemy_radius - 1),
                                    int(enemy_pos_new[0] + enemy_radius + 1), int(enemy_pos_new[1] + enemy_radius + 1)]
                    self.map[enemy_box_new[0]:enemy_box_new[2], enemy_box_new[1]:enemy_box_new[3]] = info["type_index"][0]
                    new_dis = abs(enemy_pos_new[0] - agent_pos[0]) + abs(enemy_pos_new[1] - agent_pos[1])
                    new_time = (max(abs(enemy_pos_new[0] - agent_pos[0]) - agent_radius - enemy_radius, 0)
                             + max(abs(enemy_pos_new[1] - agent_pos[1]) - agent_radius - enemy_radius, 0)) / agent_speed
                    # print(new_dis, dis, new_time)
                    if new_dis < dis and new_time <= 6:
                        threat.append([1, enemy_pos_new, info["velocity"]])


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
                projectile_pos = info["position"]
                projectile_box = info["box"]
                projectile_speed = info["speed"]
                self.map[projectile_box[0]:projectile_box[2], projectile_box[1]:projectile_box[3]] = info["type_index"][0]
        
        self.agent_pos = agent_pos
        self.agent_radius = agent_radius
        self.enemy_radius = enemy_radius
        self.reward_radius = reward_radius
        self.agent_speed = agent_speed
        self.agent_vel = agent_vel
        self.agent_box = agent_box

        # print(threat, self.map)
        FLAG, move_action_index = self.shortest_path()
        if FLAG is False:
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
                
                heuristic = 0
                if np.sum(self.map[agent_box_new[0]:agent_box_new[2], agent_box_new[1]:agent_box_new[3]] == 2) > 0:
                    heuristic += 2
                
                reward_info = env_state["state"]["local"][min_dis_reward_index]
                heuristic = (max((abs(agent_pos_new[0] - reward_info["position"][0]) - agent_radius - reward_radius + 1), 0)
                        + max((abs(agent_pos_new[1] - reward_info["position"][1]) - agent_radius - reward_radius + 1), 0)) / agent_speed
                for enemy_info in threat:
                    c, enemy_pos_new, enemy_direction = enemy_info
                    if enemy_direction[0] == 0:
                        time = (2 * (max((abs(agent_pos_new[0] - enemy_pos_new[0]) - agent_radius - enemy_radius), 0)
                                + max((abs(agent_pos_new[1] - enemy_pos_new[1]) - agent_radius - enemy_radius), 0)) / agent_speed)
                    else:
                        time = ((max((abs(agent_pos_new[0] - enemy_pos_new[0]) - agent_radius - enemy_radius), 0)
                                + 2 * max((abs(agent_pos_new[1] - enemy_pos_new[1]) - agent_radius - enemy_radius), 0)) / agent_speed)
                    if time > 0:
                        heuristic += c / time 
                    else:
                        heuristic += 100
                heuristics.append(heuristic)
            
            move_action_index = self.directions.index(choice[heuristics.index(min(heuristics))])
        # print(heuristics, self.actions_name[move_action_index])
        if self.actions_name[move_action_index] == "noop":
            if random.random() > 0.5 + 1 / (env_state["state"]["global"]["projectiles_left"] + 2):
                return self.env.getActionIndex("shoot")
            else:
                return self.env.getActionIndex("noop")
        
        if agent_vel[0] * self.directions[move_action_index][0] + agent_vel[1] * self.directions[move_action_index][1] > 0:
            # print("judge")
            t = 1
            pos = [agent_pos[0], agent_pos[1]]
            vel = agent_vel
            while True:
                pos = [pos[0] + vel[0], pos[1] + vel[1]]
                if agent_vel[0] != 0:
                    dis = abs(pos[0] - env_state["state"]["local"][min_dis_reward_index]["position"][0])
                else:
                    dis = abs(pos[1] - env_state["state"]["local"][min_dis_reward_index]["position"][1])
                
                if pos[0] > width - agent_radius or pos[0] < agent_radius or pos[1] > height - agent_radius or pos[1] < agent_radius:
                    # print(t, "out of map")
                    return self.env.getActionIndex(self.actions_name[move_action_index])

                box = [int(pos[0] - agent_radius), int(pos[1] - agent_radius),
                       int(pos[0] + agent_radius), int(pos[1] + agent_radius)]
                if np.sum(self.map[box[0]:box[2], box[1]:box[3]] == 2) > 0:
                    # print(t, "meet obstacle")
                    if random.random() < 1 / t:
                        return self.env.getActionIndex("shoot")
                    else:
                        return self.env.getActionIndex(self.actions_name[move_action_index])
                
                if t < 6:
                    for enemy_info in threat:
                        c, enemy_pos_new, enemy_vel = enemy_info
                        # print(enemy_pos_new)
                        if abs(enemy_pos_new[0] - pos[0]) < agent_radius + enemy_radius and abs(enemy_pos_new[1] - pos[1]) < agent_radius + enemy_radius:
                            # print("meet enemy")
                            if t == 1 or random.random() < c / (t - 1):
                                return self.env.getActionIndex("shoot")  
                        enemy_pos_new = [enemy_pos_new[0] + enemy_vel[0], enemy_pos_new[1] + enemy_vel[1]]
                                                    

                if np.sum(self.map[box[0]:box[2], box[1]:box[3]] == 6) > 0:
                    # print(t, "have shoot")
                    return self.env.getActionIndex(self.actions_name[move_action_index])

                if dis < agent_radius + reward_radius:
                    # print(t, "meet reward")
                    return self.env.getActionIndex(self.actions_name[move_action_index])
                t += 1
                if t > 100:
                    break
        
        return self.env.getActionIndex(self.actions_name[move_action_index])

    def shortest_path(self):
        # bfs
        path = []
        actions = []
        dxys = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dxys)
        openlist = []
        self.x, self.y = int(self.agent_pos[0]), int(self.agent_pos[1])
        openlist.append((self.agent_pos[0], self.agent_pos[1], None))
        search_map = np.zeros_like(self.map)
        search_map[self.x, self.y] = 1
        find_out = False
        end_point = None
        while len(openlist) > 0:
            node = openlist[0]
            direction = []
            for dxy in dxys:
                x = node[0] + dxy[0] * self.agent_speed
                y = node[1] + dxy[1] * self.agent_speed
                box = [int(x - self.agent_radius), int(y - self.agent_radius), int(x + self.agent_radius), int(y + self.agent_radius)]
                if not in_box((x, y), (self.agent_radius, self.agent_radius, self.map.shape[0]-self.agent_radius-1, self.map.shape[1]-self.agent_radius-1)) or search_map[int(x), int(y)] != 0:
                    continue
                if np.sum(self.map[box[0]:box[2], box[1]:box[3]] > 1) > 0:
                    continue
                # First search the path not close to the edge of the bad node
                flag = False
                for ddxy in dxys:
                    if ddxy[0] == -dxy[0] and ddxy[1] == -dxy[1]:
                        continue
                    x_ = x + ddxy[0] * self.agent_speed
                    y_ = y + ddxy[1] * self.agent_speed
                    if not (in_box((x_, y_), (self.agent_radius, self.agent_radius, self.map.shape[0]-self.agent_radius-1, self.map.shape[1]-self.agent_radius-1))):
                        continue
                    box_ = [int(x_ - self.agent_radius), int(y_ - self.agent_radius), int(x_ + self.agent_radius), int(y_ + self.agent_radius)]
                    if np.sum(self.map[box_[0]:box_[2], box_[1]:box_[3]] > 1) > 0:
                        flag = True
                        break
                if flag == True:
                    direction.append((x, y))
                else:
                    direction.insert(0, (x, y))

            for x, y in direction:
                openlist.append((x, y, node))
                box = [int(x - self.agent_radius), int(y - self.agent_radius), int(x + self.agent_radius), int(y + self.agent_radius)]
                search_map[int(x), int(y)] = 1
                if np.sum(self.map[box[0]:box[2], box[1]:box[3]] == 1) > 0 and (np.sum(self.map[box[0]:box[2], box[1]:box[3]] > 2) == 0 or int(node[0]) != int(self.x) or int(node[1]) != int(self.y)):
                    find_out = True
                    end_point = openlist[-1]
                    break
            if find_out is True:
                break
            openlist.pop(0)

        if find_out is False:
            return False, None

        father = end_point[2]
        while True:
            if father[2] == None:
                dxy = [end_point[0] - father[0], end_point[1] - father[1]]
                if dxy[0] == 0:
                    dxy[1] /= abs(dxy[1])
                else:
                    dxy[0] /= abs(dxy[0])
                break
            end_point = father
            father = end_point[2]

        return True, self.directions.index(tuple(dxy))
