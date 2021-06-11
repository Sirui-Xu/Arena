import pygame
from .randomness import Random
from .greedy import OneStep, Greedy
from .planning import Planning

import pygame
import json
import numpy as np

algorithm_names = ['Random', 'OneStep', 'Greedy', 'Planning']

def load_algorithm(env, alg_name):
    lower2upper = {name.lower():name for name in algorithm_names}
    return globals()[lower2upper[alg_name.lower()]](env)
