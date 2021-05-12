import pygame
import random
import sys
import numpy as np

def in_box(point, box):
    return point[0] <= box[2] and point[0] >= box[0] and point[1] <= box[3] and point[1] >= box[1] 

class Planning:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        self.with_maze = (self.name[-4:] == "Maze")
        self.with_bomb = (self.name[:6] == "Bomber")
        self.with_bullet = (self.name[:5] == "Shoot")
        self.1d = (self.name[-2:] == "1d")
        self.state = env.getGameState()
        self.good_map = np.zeros(self.state["global"]["map_shape"])
        self.bad_map = np.zeros(self.state["global"]["map_shape"])
        self.map_shape = self.state["global"]["map_shape"]
        self.maze = None
        if with_maze:
            self.maze = self.state["global"]["maze"]
        self.gap = 0.5
        self.pos = None
        self.init_map()

    def judge_creep(self, type_index):
        if self.name[:8] == "Billiard" or self.name[:5] == "Water":
            if type_index == 1:
                return -1
            else:
                return 1
        
        if self.name[:3] == "Pac":
            return -1

        return 1


    def init_map(self):
        self.map_shape = 
        if with_maze:
            for info in self.state["local"]:
                pos = info["norm_position"]
                vel = info["norm_velocity"]
                aug_box = [pos[0] - vel[0] - int(self.gap), pos[1] - vel[1] - int(self.gap), pos[0] + vel[0] + int(self.gap) + 1, pos[1] + vel[1] + int(self.gap) + 1]
                if info.type == 'creep':
                    value = self.judge_creep(info["type_index"])
                    if value == -1:
                        aug_box = [pos[0] - vel[0] + int(self.gap), pos[1] - vel[1] + int(self.gap), pos[0] + vel[0] - int(self.gap) + 1, pos[1] + vel[1] - int(self.gap) + 1]
                    for x in range(aug_box[0], aug_box[2]):
                        for y in range(aug_box[1], aug_box[3]):
                            self.map[x, y] = value

                elif info.type == 'bomb':
                    if info["type_index"][1] < 0.1:
                        nbr = self.state["global"]["norm_bomb_range"]
                        aug_box_1 = [pos[0] - nbr - int(self.gap), pos[1] - int(self.gap), pos[0] + nbr + int(self.gap) + 1, pos[1] + int(self.gap) + 1]
                        aug_box_2 = [pos[0] - int(self.gap), pos[1] - nbr - int(self.gap), pos[0] + int(self.gap) + 1, pos[1] + nbr + int(self.gap) + 1]
                    else:
                        for x in range(aug_box_1[0], aug_box_1[2]):
                            for y in range(aug_box_1[1], aug_box_1[3]):
                                self.map[x, y] = 1
                        for x in range(aug_box_2[0], aug_box_2[2]):
                            for y in range(aug_box_2[1], aug_box_2[3]):
                                self.map[x, y] = 1
                
                elif info.type == "player":
                    self.pos = info["norm_position"]
        else:
            for info in self.state["local"]:
                box = info["norm_box"]
                vel = info["norm_velocity"]
                aug_box = [int(box[0] + vel[0] + 1 - self.gap), int(box[1] + vel[1] + 1 - self.gap), int(box[2] + vel[0] + self.gap) + 1, int(box[3] + vel[1] + self.gap) + 1]
                if info.type == 'creep':
                    value = self.judge_creep(info["type_index"])
                    if value == -1:
                        aug_box = [int(box[0] + vel[0] + 1 + self.gap), int(box[1] + vel[1] + 1 + self.gap), int(box[2] + vel[0] - self.gap) + 1, int(box[3] + vel[1] - self.gap) + 1]
                    for x in range(aug_box[0], aug_box[2]):
                        for y in range(aug_box[1], aug_box[3]):
                            self.map[x, y] = value

                elif info.type == 'bomb':
                    if info["type_index"][1] < 0.1:
                        nbr = self.state["global"]["norm_bomb_range"]
                        aug_box_1 = [int(box[0] - nbr + 1 - self.gap), int(box[1] + 1 - self.gap), int(box[2] + nbr + self.gap) + 1, int(box[3] + self.gap) + 1]
                        aug_box_2 = [int(box[0] + 1 - self.gap), int(box[1] - nbr + 1 - self.gap), int(box[2] + self.gap) + 1, int(box[3] + nbr + self.gap) + 1]
                    else:
                        for x in range(aug_box_1[0], aug_box_1[2]):
                            for y in range(aug_box_1[1], aug_box_1[3]):
                                self.map[x, y] = 1
                        for x in range(aug_box_2[0], aug_box_2[2]):
                            for y in range(aug_box_2[1], aug_box_2[3]):
                                self.map[x, y] = 1

                elif info.type == "player":
                    self.pos = info["norm_position"]

    def shortest_path(self):
        # bfs
        path = []
        actions = []
        dxys = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dxys)
        openlist = []
        openlist.append((self.pos[0], self.pos[1], None))
        search_map = self.map.copy()
        search_map[self.pos[0], self.pos[1]] = 1
        find_out = False
        end_point = None
        while len(openlist) > 0:
            node = openlist[0]
            direction = []
            for dxy in dxys:
                x = node[0] + dxy[0]
                y = node[1] + dxy[1]
                if not in_box((x, y), (0, 0, self.map_shape-1, self.map_shape-1)) or search_map[x, y] > 0:
                    continue
                # First search the path not close to the edge of the bad node
                flag = False
                for ddxy in dxys:
                    if ddxy[0] == -dxy[0] and ddxy[1] == -dxy[1]:
                        continue
                    if in_box((x + ddxy[0], y + ddxy[1]), (0, 0, self.map_shape-1, self.map_shape-1)) and search_map[x + ddxy[0], y + ddxy[1]] > 0:
                        flag = True
                        break
                if flag == True:
                    direction.append((x, y))
                else:
                    direction.insert(0, (x, y))

            for x, y in direction:
                openlist.append((x, y, node))
                search_map[x, y] = 1
                if self.max_reward_base > 0:
                    if self.good_dilation[x, y] + self.bad_dilation[x, y] == self.max_reward_base:
                        find_out = True
                        end_point = openlist[-1]
                        break
                # If the bad node covers the good node, approach first
                else:
                    if self.good_dilation[x, y] == self.max_reward_high:
                        find_out = True
                        end_point = openlist[-1]
                        break
            if find_out is True:
                break
            openlist.pop(0)

        if find_out is False:
            if self.env.add_noop_action:
                return find_out
            else:
                dxy = dxys[0]
                x = node.pos[0] + dxy[0]
                y = node.pos[1] + dxy[1]
                if x - self.x < 0:
                    self.actions.append('w')
                elif x - self.x > 0:
                    self.actions.append('s')

                if y - self.y < 0:
                    self.actions.append('a')
                elif y - self.y > 0:
                    self.actions.append('d')
                self.path.append(Node((x, y), Node((self.x, self.y), None)))
                return True

        father = end_point.father
        while(father != None):
            self.path.append(end_point)
            if end_point.pos[0] < father.pos[0]:
                self.actions.append('w')
            if end_point.pos[0] > father.pos[0]:
                self.actions.append('s')
            if end_point.pos[1] < father.pos[1]:
                self.actions.append('a')
            if end_point.pos[1] > father.pos[1]:
                self.actions.append('d')
            
            end_point = father
            father = end_point.father
        self.actions = self.actions[::-1]
        self.path = self.path[::-1]
        
        # if self.bad_dilation[self.path[0].pos[0], self.path[0].pos[1]] < 0:
        #     return False
        # print("done!")
        return find_out


    def exe(self, state):
        return random.randint(0, self.n_action - 1)
