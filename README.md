# PyGame-Graph-Based-Learning-Environment

**PyGame Graph Based Learning Environment (PGLE)** is a learning environment, mimicking the [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) interface and [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment). This platform is designed to support the community to investigate compositional generalizability (a.k.a. combinatorial generalizability) of models and algorithms, especially **Graph Neural Netwworks**.

## Games

At present, there are five types of object-oriented games in the environment, a total of twelve available games. We are still committed to exploring and developing other games that can be well abstracted into graphs or sets, while this abstraction will not make games particularly simple.

## Algorithms

We also provide handcrafted algorithms for every game. These algorithms can be used as a baseline for the comparison of graph neural networks, and can also be used as teacher policies for graph neural networks to imitate, since it is not very successful to use graph neural networks directly for reinforcement learning in some of our games at present.

## Installation

The main part of PGLE only requires the following dependencies:
* numpy
* pygame

If you also want to use our code to test the heuristic algorithm, you would also need the following dependencies:
* argparse
* json
* opencv
  
Clone the repo and install with pip.

```bash
git clone https://github.com/Sirui-Xu/PyGame-Graph-Based-Learning-Environment.git
cd PyGame-Graph-Based-Learning-Environment/
pip install -e .
``` 

## How to play these games yourself

Hear's an example of playing bomberman. 

```bash
cd PyGame-Graph-Based-Learning-Environment/
python play.py bomberman
``` 

You need the `w, s, a, d` and `space` keys on your keyboard to move and place bombs.

## Getting started

A `PGLE` instance requires a game exposing a set of control methods. To see the required methods look at `pgle/games/base.py`. 

Here's an example of importing billiardworldmaze from the games library within PGLE:

```python
from pgle.games import BilliardWorldMaze

game = BilliardWorldMaze(width=48, height=48, num_creeps=3)
```

It's important to change the map size and the number of nodes as a test for compositional generalization.

Next we configure and initialize PGLE:

```python
from pgle import PGLE

p = PGLE(game)
p.init()
```

You are free to use any agent with the PGLE. Below we create a fictional agent and grab the valid actions:

```python
myAgent = MyAgent(p.getActionSet())
```

We can now have our agent, with the help of PGLE, interact with the game over a certain number of frames:

```python

nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
    action = myAgent.pickAction(reward, state)
    state, reward, game_over, info = p.step(action)
    if game_over: #check if the game is over
        state = p.reset()

```

The state contains the local and the global information. Here's an example.

```python
state = {"local": local_state, "global": global_state}
local_state = [ {'type': 'player', 'type_index': 0, 'position': [472.5, 402.5], 'velocity': [0, 0], 'speed': 70, 'box': [388, 458, 416, 486], 'discrete_position': [13, 11]}, 
                {'type': 'creep', 'type_index': 3, 'position': [315.0, 52.5], 'velocity': [35.0, 0.0], 'speed': 35, 'box': [38, 301, 66, 329], 'discrete_position': [9, 1]}, 
                {'type': 'creep', 'type_index': 2, 'position': [52.5, 280.0], 'velocity': [0.0, -35.0], 'speed': 35, 'box': [266, 38, 294, 66], 'discrete_position': [1, 8]} ]
global_state = {'map_shape': [15, 15], 
                'maze': array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                            [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), # a numpy array where 1 for the wall and 0 for the free space
                'rate_of_progress': 0.25555555555555554# Percentage of game progress. The game ended when 100% progressed.
               }
```

Just like that we have our agent interacting with our game environment. A specific example can be referred to `algorithm/test.py`

## Test heuristic algorithm

```bash
cd algorithm
python test.py --game ${game_name} --algorithm ${algorithm_name}
```
${game_name} should be one of the available game's name. ${algorithm_name} should be something like `randomalgorithm`.


## Acknowledgement
This environment refers a lot to ntasfi's [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment)