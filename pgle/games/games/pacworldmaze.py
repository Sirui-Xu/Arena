import pygame
import sys
import math

from ..base import PyGameWrapper, Player, Creep, Wall

from ..utils import vec2d, percent_round_int, generate_random_maze
from pygame.constants import K_w, K_a, K_s, K_d

class PacWorldMaze(PyGameWrapper):
    """
    The lighter the color, the higher the score. Need to collect node with limited time.
    Parameters
    ----------
    width : int
        Screen width.
    maze_width : int
        Maze width.
    num_creeps : int (default: 3)
        The number of creeps on the screen at once.
    NO_SPEED : bool (default: false)
        whether the node can move.
    """

    def __init__(self,
                 width=48,
                 maze_width=7,
                 num_creeps=3,
                 NO_SPEED=False,
                 fps=10):

        self.real_width = (maze_width // 2) * 2 + 1
        self.wall_width = width // self.real_width + 1
        width = self.wall_width * self.real_width
        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s
        }

        PyGameWrapper.__init__(self, width, width, actions=actions)
        self.BG_COLOR = (255, 255, 255)
        self.N_CREEPS = num_creeps
        self.CREEP_TYPES = ["GOOD", "BAD"]
        self.CREEP_COLORS = [(40, 240, 40), (150, 95, 95)]
        radius = percent_round_int(self.wall_width, 0.23)
        # print(width, self.wall_width, self.real_width, radius)
        self.CREEP_RADII = [radius, radius]
        self.CREEP_REWARD = [
            self.rewards["positive"],
            self.rewards["negative"]]
        if NO_SPEED:
            self.CREEP_SPEED = 0
        else:
            self.CREEP_SPEED = self.wall_width / 2
        self.AGENT_COLOR = (30, 30, 70)
        self.AGENT_SPEED = self.wall_width
        self.AGENT_RADIUS = radius
        self.AGENT_INIT_POS = (0, 0)
        self.UNIFORM_SPEED = True
        self.creep_counts = {
            "GOOD": 0,
            "BAD": 0
        }

        self.WALL_COLOR = (20, 10, 10)
        self.dx = 0
        self.dy = 0
        self.dx_next = 0
        self.dy_next = 0
        self.player = None
        self.creeps = None
        self.walls = None
        self.maze = None
        self.assigned_values = None
        self.fps = fps

    def vir2real(self, x, y):
        return ((x+0.5) * self.wall_width, (y+0.5) * self.wall_width)
    
    def real2vir(self, x, y):
        return (int(x / self.wall_width), int(y / self.wall_width))

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    self.dx_next -= self.wall_width

                if key == self.actions["right"]:
                    self.dx_next += self.wall_width

                if key == self.actions["up"]:
                    self.dy_next -= self.wall_width

                if key == self.actions["down"]:
                    self.dy_next += self.wall_width

    def _add_creep(self, creep_type, rand, value):
        # creep_type = self.rng.choice([0, 1])

        creep = None
        pos = (0, 0)
        dist = 0.0
        vir_pos = (self.rng.randint(0, self.real_width//2)*2+1, self.rng.randint(0, self.real_width//2)*2+1)
        pos = self.vir2real(*vir_pos)

        while dist < self.AGENT_RADIUS + self.CREEP_RADII[creep_type] + 1:
            vir_pos = (self.rng.randint(0, self.real_width//2)*2+1, self.rng.randint(0, self.real_width//2)*2+1)
            pos = self.vir2real(*vir_pos)
            dist = math.sqrt(
                (self.player.pos.x - pos[0])**2 + (self.player.pos.y - pos[1])**2)

        creep = Creep(
            (5, 150*rand+50, 10),
            self.CREEP_RADII[creep_type],
            pos,
            (0, 0),
            self.CREEP_SPEED,
            self.CREEP_REWARD[creep_type] * value * self.N_CREEPS,
            self.CREEP_TYPES[creep_type],
            self.width,
            self.height,
            0,
            150*rand+50,
        )

        self.creeps.add(creep)

        self.creep_counts[self.CREEP_TYPES[creep_type]] += 1

    def _direction_adjustment(self, p=0.9):
        for creep in self.creeps.sprites():
            creep.speed = self.CREEP_SPEED
            creep_pos_old = (creep.pos.x, creep.pos.y)
            creep_pos_new = (creep.pos.x + creep.direction.x * creep.speed * 1, creep.pos.y + creep.direction.y * creep.speed * 1)
            vir_creep_pos_old = self.real2vir(*creep_pos_old)
            vir_creep_pos_new = self.real2vir(*creep_pos_new)

            if self.maze[vir_creep_pos_new] == 0 and vir_creep_pos_new != vir_creep_pos_old and self.rng.rand() < p:
                continue

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            feasible_directions = []
            self.rng.shuffle(directions)
            for direction in directions:
                creep_pos_new_ = (creep.pos.x + direction[0] * creep.speed * 1, creep.pos.y + direction[1] * creep.speed * 1)
                if self.maze[self.real2vir(*creep_pos_new_)] == 0:
                    if direction[0] == -creep.direction.x and direction[1] == -creep.direction.y:
                        feasible_directions.append(direction)
                    else:
                        feasible_directions.insert(0, direction)
                        break
                    
            if len(feasible_directions) == 0:
                creep.speed = 0
                continue
            feasible_direction = feasible_directions[0]
            creep.direction.x = feasible_direction[0]
            creep.direction.y = feasible_direction[1]

    def getGameState(self):
        player_vir_pos = self.real2vir(self.player.pos.x, self.player.pos.y)
        player_vir_vel = [self.player.vel.x / self.wall_width, self.player.vel.y / self.wall_width]
        player_vir_spd = self.AGENT_SPEED / self.wall_width
        player_vir_box = [self.player.rect.left / self.wall_width - 0.5,
                          self.player.rect.top / self.wall_width - 0.5,     
                          self.player.rect.right / self.wall_width - 0.5,                       
                          self.player.rect.bottom / self.wall_width - 0.5, 
                         ]
        player_state = {'type':'player', 
                        'type_index': [0, -1], 
                        'position': [self.player.pos.x, self.player.pos.y],
                        'velocity': [self.player.vel.x, self.player.vel.y],
                        'speed': self.AGENT_SPEED,
                        'box': [self.player.rect.left, self.player.rect.top, self.player.rect.right, self.player.rect.bottom],
                        'norm_position': [player_vir_pos[0], player_vir_pos[1]],
                        'norm_velocity': player_vir_vel,
                        'norm_speed': player_vir_spd,
                        'norm_box': player_vir_box,
                       }

        state = [player_state]
        order = list(range(len(self.creeps.sprites())))
        # self.rng.shuffle(order)
        for i in order:
            c = self.creeps.sprites()[i]
            vir_pos = self.real2vir(c.pos.x, c.pos.y)
            vir_vel = [c.direction.x * c.speed / self.wall_width, c.direction.y * c.speed / self.wall_width]
            vir_spd = c.speed / self.wall_width
            vir_box = [c.rect.left / self.wall_width - 0.5,
                       c.rect.top / self.wall_width - 0.5, 
                       c.rect.right / self.wall_width - 0.5,
                       c.rect.bottom / self.wall_width - 0.5, 
                      ]
            creep_state = {'type':'creep', 
                           'type_index': [1, c.reward],  
                           'position': [c.pos.x, c.pos.y],
                           'velocity': [c.direction.x * c.speed, c.direction.y * c.speed],
                           'speed': c.speed,
                           'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                           '_color': c.idx,
                           '_type': "GOOD",
                           'norm_position': [vir_pos[0], vir_pos[1]],
                           'norm_velocity': vir_vel,
                           'norm_speed': vir_spd,
                           'norm_box': vir_box,
                          }
            state.append(creep_state)

        global_state = {'norm_shape':[self.maze.shape[0], self.maze.shape[1]], 
                        'shape': [self.width, self.height],
                        'maze':self.maze, 
                        'rate_of_progress': (self.ticks * self.wall_width / self.fps) / (self.width + self.height),
                        'ticks': self.ticks,
                        'times': int((self.width + self.height) / (self.AGENT_SPEED / self.fps) + 0.5),
                        'score': self.score}
        return {'local':state, 'global':global_state}

    def loadGameState(self, state):
        self.maze = state["global"]["maze"]
        self.creep_counts = {"GOOD": 0, "BAD": 0}
        if self.creeps is None:
            self.creeps = pygame.sprite.Group()
        else:
            self.creeps.empty()
        for info in state["local"]:
            if info["type"] == "player":
                self.AGENT_INIT_POS = info["position"]
                if self.player is None:
                    self.player = Player(
                        self.AGENT_RADIUS, self.AGENT_COLOR,
                        self.AGENT_SPEED, self.AGENT_INIT_POS,
                        self.width, self.height,
                        self.UNIFORM_SPEED
                    )

                else:
                    self.player.pos = vec2d(self.AGENT_INIT_POS)
                    self.player.vel = vec2d(info["velocity"])
                    self.player.rect.center = self.AGENT_INIT_POS
            if info["type"] == "creep":
                reward = info["type_index"][1]
                creep = Creep(
                    (5, info["_color"], 10),
                    self.CREEP_RADII[0],
                    info["position"],
                    info["velocity"],
                    info["speed"],
                    reward,
                    self.CREEP_TYPES[0],
                    self.width,
                    self.height,
                    0,
                    info["_color"],
                )
                self.creeps.add(creep)

                self.creep_counts[self.CREEP_TYPES[0]] += 1

        self.score = state["global"]["score"]
        self.ticks = state["global"]["ticks"]
        self.lives = -1
        self.dx_next, self.dy_next = 0, 0
        # self.screen.fill(self.BG_COLOR)
        # self.player.draw(self.screen)
        # self.creeps.draw(self.screen)
        # self.walls.draw(self.screen)
        
    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return (self.creep_counts['GOOD'] + self.creep_counts['BAD'] == 0) or self.ticks * self.wall_width / self.fps >= (self.width + self.height)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """

        self.assigned_values = list(range(self.N_CREEPS))
        self.maze = generate_random_maze(self.real_width, self.real_width, complexity=.1, density=.1)
        # print(self.maze)
        self.creep_counts = {"GOOD": 0, "BAD": 0}
        vir_pos = (self.rng.randint(0, self.real_width // 2)*2+1, self.rng.randint(0, self.real_width//2)*2+1)
        self.AGENT_INIT_POS = self.vir2real(*vir_pos)

        if self.player is None:
            self.player = Player(
                self.AGENT_RADIUS, self.AGENT_COLOR,
                self.AGENT_SPEED, self.AGENT_INIT_POS,
                self.width, self.height,
                self.UNIFORM_SPEED
            )

        else:
            self.player.pos = vec2d(self.AGENT_INIT_POS)
            self.player.vel = vec2d((0.0, 0.0))
            self.player.rect.center = self.AGENT_INIT_POS

        if self.creeps is None:
            self.creeps = pygame.sprite.Group()
        else:
            self.creeps.empty()

        if self.walls is None:
            self.walls = pygame.sprite.Group()
        else:
            self.walls.empty()

        sum_assigned_values = sum(self.assigned_values)
        for i in range(self.N_CREEPS):
            self._add_creep(0, self.assigned_values[i], self.assigned_values[i] / sum_assigned_values)

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] == 1:
                    self.walls.add(Wall(self.vir2real(i, j), self.wall_width, self.wall_width, self.WALL_COLOR))

        self.score = 0
        self.ticks = 0
        self.lives = -1
        self.dx_next, self.dy_next = 0, 0
        self._direction_adjustment()
        self.screen.fill(self.BG_COLOR)
        self.player.draw(self.screen)
        self.creeps.draw(self.screen)
        self.walls.draw(self.screen)

    def step(self):
        """
            Perform one step of game emulation.
        """
        self.screen.fill(self.BG_COLOR)

        if self.dx_next == 0 and self.dy_next == 0:
            self._handle_player_events()
        else:
            pygame.event.pump()

        if self.ticks % self.fps == 0:
            self.dx, self.dy = self.dx_next, self.dy_next
            self.dx_next, self.dy_next = 0, 0
            if self.dx == 0 and self.dy == 0:
                player_pos_new = (self.player.pos.x + self.player.vel.x, self.player.pos.y + self.player.vel.y)
            else:
                player_pos_new = (self.player.pos.x + self.dx, self.player.pos.y + self.dy) 
            vir_player_pos_new = self.real2vir(*player_pos_new)        
            if self.maze[vir_player_pos_new] != 0:
                self.player.vel.x = 0
                self.player.vel.y = 0
                self.dx = 0
                self.dy = 0

            if self.ticks != 0:
                self._direction_adjustment()

        self.player.update(self.dx, self.dy, 1 / self.fps)                    
        self.creeps.update(1 / self.fps)
        hits = pygame.sprite.spritecollide(self.player, self.creeps, False)
        for creep in hits:
            self.creep_counts[creep.TYPE] -= 1
            self.score += creep.reward
            creep.kill()

        if self.creep_counts["GOOD"] == 0:
            self.score += self.rewards["win"]

        self.player.draw(self.screen)
        self.creeps.draw(self.screen)
        self.walls.draw(self.screen)
        self.ticks += 1

if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = PacWorldMaze(width=512, maze_width=15, num_creeps=5)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()
    fps = 5
    while True:
        for i in range(game.fps):
            dt = game.clock.tick_busy_loop(fps * game.fps)
            game.step()
            pygame.display.update()
        if game.game_over() is True:
            print("The overall score is {}.".format(game.score))
            break
        print(game.getGameState(), '\n')