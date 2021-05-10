import pygame
import random
import sys
import numpy as np
class Greedy:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        self.with_maze = (self.name[-4:] == "Maze")
        self.with_bomb = (self.name[:6] == "Bomber")
        self.with_bullet = (self.name[:5] == "Shoot")
        self.1d = (self.name[-2:] == "1d")
        self.state = env.getGameState()
        self.map = np.zeros(self.state["global"]["map_shape"])
        self.init_map()

    def init_map(self):
        for info in self.state["local"]:
            if info.type == 'creep' or info.type == 'bomb':
                self.map[tuple(info["discrete_position"])] = 1
        
        if self.with_maze:
            return

        

    def shortest_path(self):


    def exe(self, state):
        return random.randint(0, self.n_action - 1)
