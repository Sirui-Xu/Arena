import pygame
import random
import sys

class RandomAlgorithm:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        
    def exe(self, state):
        return random.randint(0, self.n_action - 1)
