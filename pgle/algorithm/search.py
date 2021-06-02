import os
import sys
import math
import copy
import random
import numpy as np
from collections import namedtuple
from queue import PriorityQueue
from itertools import count
import json
import pickle
import time
import cv2
from tqdm import tqdm
import random

directions = [(-1, 0), (0, -1), (0, 1), (1, 0), (0, 0)]
class Creep:
    def __init__(self, position, direction):
        self.distribution = [{} for _ in directions]
        self.update_distribution = [{} for _ in directions]
        self.distribution_add(1, direction, position)
        self.position = position
        self.update()

    def sample(self, n):
        info = []
        prob = []
        for i, direction in enumerate(directions):
            for position in self.distribution[i].keys():
                info.append((position, direction))
                prob.append(self.distribution[i][position])
        info.append((-1,-1),(0,0))
        prob.append(1 - sum(prob))
        return np.random.choice(info, n, p=prob)

    def split(self, maze):
        for i, old_direction in enumerate(directions):
            for old_position in self.distribution[i]:
                new_directions, possibilities = self.posible_directions(old_direction, old_position, maze)
                update_possibility = self.distribution[self.dir2id(old_direction)][old_position]
                for j, new_direction in enumerate(new_directions):
                    new_position = creep_walk(old_position, new_direction)
                    self.distribution_add(update_possibility * possibilities[j], new_direction, new_position)

    def posible_directions(self, old_direction, old_position, maze, p=0.9):
        new_directions = [new_direction for new_direction in directions[:-1] 
                               if (new_direction != backward(old_direction)) and
                               (maze[creep_walk(old_position, new_direction)] == 0)]
        if len(new_directions) > 0:
            possibility = []
            if old_direction in new_directions:
                possibility = [(1 - p) / len(new_directions) for _ in new_directions]
                possibility[new_directions.index(old_direction)] += p 
            else:
                possibility = [1 / len(new_directions) for _ in new_directions]
            return new_directions, possibility
        
        if maze[creep_walk(old_position, backward(old_direction))] == 0:
            return [backward(old_direction)], [1]

        return [directions[-1]], [1]

    def collision(self, agent_old_position, agent_old_direction):
        return self.distribution_remove_direction(agent_old_position, backward(agent_old_direction))
    
    def coincidence(self, agent_old_position, agent_old_direction):
        return self.distribution_remove(creep_walk(agent_old_position, agent_old_direction))

    def explosion(self, explosion_zone):
        possibility = 0
        for position in explosion_zone:
            possibility += self.distribution_remove(position)
        return possibility

    def pipeline(self, agent_old_position, agent_old_direction, explosion_zone, maze):
        self.split(maze)
        danger = self.collision(agent_old_position, agent_old_direction) + self.coincidence(agent_old_position, agent_old_direction)
        reward = self.explosion(explosion_zone)
        self.update()
        return danger, reward

    def dir2id(self, direction):
        if direction not in directions:
            raise Exception('not a valid direction.')
        return directions.index(direction)
    
    def distribution_add(self, possibility, direction, position):
        distribution_direction = self.update_distribution[self.dir2id(direction)]
        if position not in distribution_direction.keys():
            distribution_direction[position] = possibility
        else:
            distribution_direction[position] += possibility

    def distribution_remove(self, position):
        possibility = 0
        for direction in directions:
            distribution_direction = self.update_distribution[self.dir2id(direction)]
            if position in distribution_direction.keys():
                possibility += distribution_direction.pop(position)
        return possibility

    def distribution_remove_direction(self, position, direction):
        possibility = 0
        distribution_direction = self.update_distribution[self.dir2id(direction)]
        if position in distribution_direction.keys():
            possibility += distribution_direction.pop(position)
        return possibility

    def update(self):
        self.distribution = self.update_distribution.copy()
        self.update_distribution = [{} for _ in directions]

    def __str__(self):
        return str(self.distribution)

class Creeps:
    def __init__(self, env):
        self.creeps = []
        for c in env.game.creeps:
            position = env.game.real2vir(c.pos.x, c.pos.y)
            direction = (int(c.direction.x), int(c.direction.y))
            self.creeps.append(Creep(position, direction))
        # self.creeps_ = copy.deepcopy(self.creeps)
        self.danger_all = [0 for _ in self.creeps]
        self.reward_all = [0 for _ in self.creeps]

    def sample(self, n):
        creeps_info = []
        for creep in self.creeps:
            creeps_info.append(creep.sample(n))
        data_batch = []
        for i in range(n):
            data = {}
            data["creep_pos"] = []
            data["creep_dir"] = []
            for creep_info in creeps_info:
                if creep_info[i][0] == (-1, -1):
                    continue
                else:
                    data["creep_pos"].append(list(creep_info[i][0]))
                    data["creep_dir"].append(list(creep_info[i][1]))
            data_batch.append(data)
        return data_batch

    def step(self, agent_old_position, agent_old_direction, explosion_zone, maze):
        for i, creep in enumerate(self.creeps):
            danger, reward = creep.pipeline(agent_old_position, agent_old_direction, explosion_zone, maze)
            self.danger_all[i] = danger
            self.reward_all[i] = reward

            # creep_ = self.creeps_[i]
            # _, reward_ = creep_.pipeline((0, 0), (0, 0), explosion_zone, maze)
            # self.reward_all_ = reward_
        return self.danger(), self.reward()

    def danger(self):
        return np.array(self.danger_all)

    def reward(self):
        return np.array(self.reward_all)# self.reward_all, self.reward_all_

    def __str__(self):
        s = 'danger from creep=' + str(self.danger_all) + ' reward=' + str(self.reward_all) + '\n'
        for i, creep in enumerate(self.creeps):
            s += 'creep ' + str(i) +': '+ str(creep) + '\n'
        return s

class Bombs:
    def __init__(self, env):
        self.bomb_range = env.game.BOMB_RANGE
        self.bomb_life = env.game.BOMB_LIFE
        self.bombs_dict = {}
        self.maze = np.copy(env.game.maze)
        self.update_maze = np.copy(self.maze)
        for b in env.game.bombs:
            self.bombs_dict[env.game.real2vir(b.pos.x, b.pos.y)] = b.life

    def _cal_explode_pos(self, bomb_pos):
        explode_pos = []
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        width, height = self.maze.shape
        for bomb_range in range(self.bomb_range+1):
            deldirs = []
            for direction in dirs:
                pos = (bomb_pos[0] + direction[0] * bomb_range, bomb_pos[1] + direction[1] * bomb_range)
                if pos[0] < 1 or pos[0] >= width - 1 or pos[1] < 1 or pos[1] >= height - 1:
                    deldirs.append(direction)
                    continue
                if pos[0] % 2 == 0 and pos[1] % 2 == 0:
                    deldirs.append(direction)
                    continue
                if self.maze[pos[0], pos[1]] == 1:
                    deldirs.append(direction)
                explode_pos.append(pos)
            dirs = [direction for direction in dirs if direction not in deldirs]
            if len(dirs) == 0:
                break
        return explode_pos

    def simulate_explode(self):
        explode_bomb_pos = []
        for bomb_pos in self.bombs_dict.keys():
            self.bombs_dict[bomb_pos] -= 1
            if self.bombs_dict[bomb_pos] <= 0.5:
                explode_bomb_pos.append(bomb_pos)

        for pos in explode_bomb_pos:
            self.bombs_dict.pop(pos)

        explode_pos = set()
        while len(explode_bomb_pos) > 0:
            bomb_pos = explode_bomb_pos.pop(0)
            explode_pos_temp = self._cal_explode_pos(bomb_pos)
            involve_bomb_pos = [pos for pos in self.bombs_dict.keys() if pos in explode_pos_temp]
            for pos in involve_bomb_pos:
                self.bombs_dict.pop(pos)
            explode_bomb_pos.extend(involve_bomb_pos)
            explode_pos = explode_pos.union(set(explode_pos_temp))
        self.update_maze = np.copy(self.maze)
        for pos in explode_pos:
            self.update_maze[pos] = 0
        return list(explode_pos)

    def add_bomb(self, position):
        self.maze[position] = 2
        self.bombs_dict[position] = self.bomb_life
    
    def update(self):
        self.maze = np.copy(self.update_maze)

    def __str__(self):
        return "bomb info: " + str(self.bombs_dict)

class State:
    def __init__(self, env, proportion=1, confidence_decay_creep=1., confidence_decay_bomb=1.):
        self.creeps = Creeps(env)
        self.bombs = Bombs(env)
        self.actions = []
        self.player_pos = env.game.real2vir(env.game.player.pos.x, env.game.player.pos.y)
        self.danger_creep = np.array(self.creeps.danger_all, dtype=np.float64)
        self.danger_bomb = 0
        self.reward_creep = np.array(self.creeps.danger_all, dtype=np.float64)
        self.proportion = proportion
        self.confidence_creep = 1
        self.confidence_bomb = 1
        self.confidence_decay_creep = confidence_decay_creep
        self.confidence_decay_bomb = confidence_decay_bomb
        self.score = 0

    def sample(self, n):
        data_batch = self.creeps.sample(n)
        bomb_pos, bomb_life = [], []
        for pos, life in self.bombs.bombs_dict.items():
            bomb_pos.append(pos)
            bomb_life.append(life)
        for i in range(n):
            data_batch[i]["bomb_pos"] = bomb_pos
            data_batch[i]["bomb_life"] = bomb_life
            data_batch[i]["player_x"] = self.player_pos[0]
            data_batch[i]["player_y"] = self.player_pos[1]
        return data_batch

    def get_score(self):
        danger_creep = 1 - np.prod(1 - self.danger_creep)
        self.danger = 1 - (1 - danger_creep) * (1 - self.danger_bomb)
        self.reward = np.sum(self.reward_creep)
        self.score = (1 - self.danger) * self.reward - self.proportion * self.danger

    def step(self, action):
        self.actions.append(action)
        new_player_pos, player_dir = walk(action, self.player_pos)
        if self.bombs.maze[new_player_pos] != 0 and action != 'n':
            return False
        if action == 'j':
            self.bombs.add_bomb(self.player_pos)
        self.explode_pos = self.bombs.simulate_explode()
        danger_creep, reward_creep = self.creeps.step(self.player_pos, player_dir, self.explode_pos, self.bombs.maze)
        self.bombs.update()
        self.danger_creep += self.confidence_decay_creep * danger_creep
        self.reward_creep += self.confidence_decay_bomb * reward_creep # reward_2
        self.player_pos = new_player_pos
        if self.player_pos in self.explode_pos:
            self.danger_bomb = 1
        self.confidence_update()
        self.get_score()
        return self.score > -self.proportion / 2
    
    def finish(self):
        while len(self.bombs.bombs_dict.keys()) > 0:
            self.explode_pos = self.bombs.simulate_explode()
            danger_creep, reward_creep = self.creeps.step((0, 0), (0, 0), self.explode_pos, self.bombs.maze)
            self.bombs.update()
            self.reward_creep += self.confidence_bomb * reward_creep # reward_2
            self.confidence_update()
        self.get_score()

    def confidence_update(self):
        self.confidence_creep *= self.confidence_decay_creep
        self.confidence_bomb *= self.confidence_decay_bomb

    def __lt__(self, other):
        return self.score < other.score

    def same(self, other):
        return self.player_pos == other.player_pos and self.creeps == other.creeps and self.bombs == other.bombs

    def __str__(self):
        s = "action sequence: " + str(self.actions) + '\n'
        s += "player_pos: " + str(self.player_pos) + '\n'
        s += str(self.creeps) + '\n'
        s += str(self.bombs) + '\n'
        if len(self.actions) > 0:
            s += "explosion zone:" + str(self.explode_pos) + '\n'
            s += "danger from bomb: " + str(self.player_pos in self.explode_pos) + '\n'
            s += "all danger from creeps: " + str(list(self.danger_creep)) + '\n'
            s += "all danger from bombs: " + str(self.danger_bomb) + '\n'
        s += '\n'
        return s

class SearchBomberManMaze:
    def __init__(self, env, depth=4, buffer_length=6*6*2):
        super().__init__()
        assert env.name[:6] == "Bomber" and env.name[-4:] == "Maze"
        self.env = env
        self.action_space = ['j', 'n', 'w', 's', 'a', 'd']
        self.reference_action = []
        self.actions = []
        self.init()
        self.depth = depth
        self.buffer_length = buffer_length


    def init(self):
        self.reset()
        self.actions = []

    def reset(self):
        # print('begin to initialize...')
        self.state = State(self.env)
        self.reference_action = []
        random.shuffle(self.action_space)
        # self.maze = np.copy(self.env.game.maze)
        # self.width, self.height = self.maze.shape
        # self.creeps_pos = []
        # self.creeps_dir = []
        # self.bombs_pos = []
        # self.pos2life = {}
        # self.player_pos = self.env.game.real2vir(self.env.game.player.pos.x, self.env.game.player.pos.y)
        # for c in self.env.game.creeps:
        #     self.creeps_pos.append(self.env.game.real2vir(c.pos.x, c.pos.y))
        #     self.creeps_dir.append((int(c.direction.x), int(c.direction.y)))
        # for b in self.env.game.bombs:
        #     self.bombs_pos.append(self.env.game.real2vir(b.pos.x, b.pos.y))
        #     self.pos2life[self.bombs_pos[-1]] = b.life

    def sample(self):
        actions = []
        buffer = []
        states = []
        states.append(self.state)
        for i in range(self.depth):
            random.shuffle(states)
            states.sort(reverse=True)
            buffer = states[:self.buffer_length]
            states.clear()
            if len(self.reference_action) > i:
                self.action_space.pop(self.reference_action[i])
                self.action_space.insert(0, self.reference_action[i])
            while(len(buffer) > 0):
                state = buffer.pop(0)
                # print("state waiting for spliting: " + str(state))
                for action in self.action_space:
                    state_new = copy.deepcopy(state)
                    valid = state_new.step(action)
                    if valid:
                        states.append(state_new)
                        # print("splited state: " + str(state_new))

        data = {}
        for state in states:
            data[tuple(state.actions)] = state.sample()
        return data

    def search(self):
        actions = []
        buffer = []
        states = []
        states.append(self.state)
        # print(self.state)
        # print(pos2dirs, pos2prob)
        for i in range(self.depth):
            # print(i)
            random.shuffle(states)
            states.sort(reverse=True)
            buffer = states[:self.buffer_length]
            states.clear()
            if len(self.reference_action) > i:
                self.action_space.pop(self.reference_action[i])
                self.action_space.insert(0, self.reference_action[i])
            while(len(buffer) > 0):
                state = buffer.pop(0)
                # print("state waiting for spliting: " + str(state))
                for action in self.action_space:
                    state_new = copy.deepcopy(state)
                    valid = state_new.step(action)
                    if valid:
                        states.append(state_new)
                        # print("splited state: " + str(state_new))

        for state in states:
            state.finish()

        states.sort(reverse=True)
        if len(states) > 0:
            if len(states[0].actions) > 1:
                self.reference_action = states[0].actions[1:]
            else:
                self.reference_action = []
            # print(states[0])
            return states[0].actions, states[0].score, states[0].danger, states[0].reward
        else:
            self.reference_action = []
            return ['n'], -1, 1, 0
                

    def exe(self):
        self.init()
        creeps_pos = [creep.position for creep in self.state.creeps.creeps]
        bombs_pos = self.state.bombs.bombs_dict.keys()
        find_out, _actions, path = find_shortest_path(self.state.player_pos, creeps_pos, self.state.bombs.maze)
        if find_out:
            self.distance = len(_actions)
            self.direction = _actions[0]
        else:
            self.distance = -1
            self.direction = 'n'
        self.actions, self.scores, self.danger_scores, self.explosion_scores = self.search()
        # print(self.actions, self.scores, self.danger_scores, self.explosion_scores, self.distance, self.direction)
        if self.danger_scores < 1e-1 and self.explosion_scores < 1e-1 and not simple_in_danger(bombs_pos, self.depth, self.state.player_pos, creeps_pos):
            self.actions = [self.direction]
        action_name = ['fire', 'noop', 'up', 'down', 'left', 'right']
        action_space = ['j', 'n', 'w', 's', 'a', 'd']
        action_name = action_name[action_space.index(self.actions[0])]
        return self.env.getActionIndex(action_name)


def simple_in_danger(bombs_pos, depth, player_pos, creeps_pos):
    for i, pos in enumerate(bombs_pos):
        if abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1]) <= depth:
            return True
    
    for i, pos in enumerate(creeps_pos):
        if abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1]) <= depth:
            return True

    return False

class Node:
    def __init__(self, pos, father):
        self.pos = pos
        self.father = father

def find_shortest_path(player_pos, creeps_pos, maze):
    path = []
    actions = []
    dxys = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    openlist = []
    search_map = np.zeros_like(maze)
    openlist.append(Node(player_pos, None))
    search_map[player_pos] = 1
    if player_pos in creeps_pos:
        return True, [], [openlist[0]]
    find_out = False
    end_point = None
    while len(openlist) > 0:
        node = openlist[0]
        for dxy in dxys:
            x = node.pos[0] + dxy[0]
            y = node.pos[1] + dxy[1]
            if maze[x, y] != 0 or search_map[x, y] == 1:
                continue
            openlist.append(Node((x, y), node))
            search_map[x, y] = 1
            if (x, y) in creeps_pos:
                find_out = True
                end_point = openlist[-1]
                break
        if find_out is True:
            break
        openlist.pop(0)

    if find_out is False:
        return False, None, None
                
    father = end_point.father
    while(father != None):
        path.append(end_point)
        if end_point.pos[0] < father.pos[0]:
            actions.append('a')
        if end_point.pos[0] > father.pos[0]:
            actions.append('d')
        if end_point.pos[1] < father.pos[1]:
            actions.append('w')
        if end_point.pos[1] > father.pos[1]:
            actions.append('s')
        
        end_point = father
        father = end_point.father
    actions = actions[::-1]
    path = path[::-1]
    return find_out, actions, path

def walk(action, player_pos):
    if action == 'a':
        _player_pos = (player_pos[0] - 1, player_pos[1])
    elif action == 'd':
        _player_pos = (player_pos[0] + 1, player_pos[1])
    elif action == 'w':
        _player_pos = (player_pos[0], player_pos[1] - 1)
    elif action == 's':
        _player_pos = (player_pos[0], player_pos[1] + 1)
    else:
        _player_pos = player_pos
    _player_dir = (_player_pos[0] - player_pos[0], _player_pos[1] - player_pos[1])
    return _player_pos, _player_dir

def backward(direction):
    return (-direction[0], -direction[1])
    
def creep_walk(position, direction):
    return (position[0] + direction[0], position[1] + direction[1])