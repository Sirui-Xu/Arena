# Arena: A Scalable and Configurable Benchmark for Policy Learning

Arena is a scalable and configurable benchmark for policy learning. It is an object-based game-like environment. The game logic is reminiscent of many classic games such as Pac-Man and Bomberman. An instance of the Arena benchmark starts with an arbitrarily sized region (i.e., the arena) containing a controllable agent as well as an arbitrary number of destructable obstacles, enemies, and collectable coins. The agent can move in four directions, fire projectiles, as well as place bombs. The goal is to control the agent to collect as many coins as possible in the shortest amount of time, potentially kill enemies and destroy obstacles using the projectiles and bombs along the way. 
 
## Installation

The main part of PGLE only requires the following dependencies:
* numpy
* pygame
  
Clone the repo and install with pip.

```bash
git clone https://github.com/Sirui-Xu/Arena.git
cd Arena/
pip install -e .
``` 

## How to play these games yourself

```bash
cd examples/
python play.py
``` 

Use `w, s, a, d` to move, `space` to place bombs, and `j` to fire projectiles.

## Getting started

Here's an example of importing Arena from the games library within Wrapper:

```python
from arena import Arena

game = Arena(width=1280,
             height=720,
             object_size=32,
             obstacle_size=40,
             num_coins=50,
             num_enemies=50,
             num_bombs=3,
             explosion_max_step=100,
             explosion_radius=128,
             num_projectiles=3,
             num_obstacles=200,
             agent_speed=8,
             enemy_speed=8,
             p_change_direction=0.01,
             projectile_speed=32,
             visualize=True,
             reward_decay=0.99)
```

It's important to change the map size and the number of objects as a test for scalability.

Next we configure and initialize Wrapper:

```python
from arena import Wrapper

p = Wrapper(game)
p.init()
```

You are free to use any agent with the Wrapper. Below we create a fictional agent and grab the valid actions:

```python
myAgent = MyAgent(p.getActionSet())
```

We can now have our agent, with the help of Wrapper, interact with the game over a certain number of frames:

```python

nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
    action = myAgent.pickAction(reward, state)
    state, reward, game_over, info = p.step(action)
    if game_over: #check if the game is over
        state = p.reset()

```
Just like that we have our agent interacting with our game environment. A specific example can be referred to `example/test.py`

## Test heuristic policy

```bash
cd example
python algorithm.py --algorithm ${algorithm_name} --store_data
```
${algorithm_name} should be something like `random`.

## Train GNN policy

```
cd example
python train.py --dataset ${data_path} --checkpoints_path ${checkpoints_path} --model ${model_name}
```

## Test GNN policy

```
cd example
python test.py --checkpoints_path ${checkpoints_path}
```

## Acknowledgements
We referred to the [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment) for some of the implementations.

<!--
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
local_state = [ {'type': 'player', 'type_index': 0, 'position': [472.5, 402.5], 'velocity': [0, 0], 'speed': 70, 'box': [388, 458, 416, 486]}, 
                {'type': 'creep', 'type_index': 3, 'position': [315.0, 52.5], 'velocity': [35.0, 0.0], 'speed': 35, 'box': [38, 301, 66, 329], }, 
                {'type': 'creep', 'type_index': 2, 'position': [52.5, 280.0], 'velocity': [0.0, -35.0], 'speed': 35, 'box': [266, 38, 294, 66] ]
global_state = {
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

Just like that we have our agent interacting with our game environment. A specific example can be referred to `example/test.py`

## Test heuristic algorithm

```bash
cd example
python test.py --game ${game_name} --algorithm ${algorithm_name}
```
${game_name} should be one of the available game's name. ${algorithm_name} should be something like `random`.



## Acknowledgement
This environment refers a lot to ntasfi's [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment)


**PyGame Graph Based Learning Environment (PGLE)** is a learning environment, mimicking the [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) interface and [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment). This platform is designed to support the community to investigate compositional generalizability (a.k.a. combinatorial generalizability) of models and algorithms, especially **Graph Neural Networks**.

## Games

At present, there are five types of object-oriented games in the environment, a total of twelve available games. We are still committed to exploring and developing other games that can be well abstracted into graphs or sets, while this abstraction will not make games particularly simple.

* WaterWorld

    This environment has multiple objects of assorted types and colors. Picking up the wrong object produces a negative reward.

* PacWorld

    This environment has multiple objects of assorted colors. The lighter the color, the higher the reward. The agent need to pick up objects in the limited time.

* BilliardWorld

    This environment has multiple objects of assorted colors. The agent need to pick up objects in order. The order is represented by the color of the object from dark to light.

* Shootworld

    In this game, the agent needs to avoid the target, but needs to destroy all targets by shooting.

* BomberMan

    In this game, the agent needs to avoid the target, but needs to place bomb to destroy all objects.



**With or without maze**

If there was a wall in the map, the agent could not pass through the wall, but some walls could be destroyed by shooting or placing bombs.

For example, The game WaterWorld with maze is called WaterWorldMaze.


 **Complexity Analysis**

assumed that the moving range of each step of the agent is $m_a$. The moving range of each step of the agent is $m_o$. The density of objects in the game is $\rho$ and there are $n$ objects. The number of bombs placed is $n_b$

| Game | WaterWorld | PacWorld | BilliardWorld | Shootworld | Bomberman |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Graph size | $1+\rho(m_a+m_o)$ | 1 | $1+\rho(m_a+m_o)$ | $2+\rho(m_a+m_o)$ | $1+n_b+\rho(m_a+m_o)$ |
|Graph diameter| 2 |2 | 2 | 3| 3|

If the number of nodes and diameter of the maze graph is $n_w$ and $d_w$

|Game Setting| w/o maze | w/ maze|
|---|---|---|
|Graph size| $s$ | $s+n_w$|
|Graph diameter| $d$ | $d+d_w$|
## Algorithms

We also provide handcrafted algorithms for every game. These algorithms can be used as a baseline for the comparison of graph neural networks, and can also be used as teacher policies for graph neural networks to imitate, since it is not very successful to use graph neural networks directly for reinforcement learning in some of our games at present.
 -->
