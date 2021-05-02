import pygame
import sys
sys.path.append("..")
import math

from base import PyGameWrapper, Player, Creep, Wall

from utils import vec2d, percent_round_int, generate_random_maze
from pygame.constants import K_w, K_a, K_s, K_d

class WaterWorldMaze(PyGameWrapper):
    """
    Based Karpthy's WaterWorld in `REINFORCEjs`_.
    .. _REINFORCEjs: https://github.com/karpathy/reinforcejs
    Parameters
    ----------
    width : int
        Screen width.
    maze_width : int
        Maze width.
    num_creeps : int (default: 3)
        The number of creeps on the screen at once.
    UNIFORM_SPEED : bool (default: false)
        The agent has an uniform speed or not
    NO_SPEED : bool (default: false)
        whether the node can move.
    """

    def __init__(self,
                 width=48,
                 maze_width=7,
                 num_creeps=3,
                 UNIFORM_SPEED=True,
                 NO_SPEED=False,
                 fps=3):

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
        radius = percent_round_int(self.wall_width, 0.4)
        self.CREEP_RADII = [radius, radius]
        self.CREEP_REWARD = [
            self.rewards["positive"],
            self.rewards["negative"]]
        if NO_SPEED:
            self.CREEP_SPEED = 0
        else:
            self.CREEP_SPEED = self.wall_width
        self.AGENT_COLOR = (30, 30, 70)
        self.AGENT_SPEED = self.wall_width
        self.AGENT_RADIUS = radius
        self.AGENT_INIT_POS = (0, 0)
        self.UNIFORM_SPEED = UNIFORM_SPEED
        self.creep_counts = {
            "GOOD": 0,
            "BAD": 0
        }

        self.WALL_COLOR = (20, 10, 10)
        self.dx = 0
        self.dy = 0
        self.player = None
        self.creeps = None
        self.walls = None
        self.maze = None

    def vir2real(self, x, y):
        return ((x+0.5) * self.wall_width, (y+0.5) * self.wall_width)
    
    def real2vir(self, x, y):
        return (int(x / self.wall_width - 0.5), int(y / self.wall_width - 0.5))

    def _handle_player_events(self):
        self.dx = 0
        self.dy = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    self.dx -= self.AGENT_SPEED

                if key == self.actions["right"]:
                    self.dx += self.AGENT_SPEED

                if key == self.actions["up"]:
                    self.dy -= self.AGENT_SPEED

                if key == self.actions["down"]:
                    self.dy += self.AGENT_SPEED

    def _add_creep(self, creep_type):
        # creep_type = self.rng.choice([0, 1])

        creep = None
        pos = (0, 0)
        dist = 0.0

        # this space will always be empty
        vir_pos = (self.rng.randint(0, self.real_width//2)*2+1, self.rng.randint(0, self.real_width//2)*2+1)
        pos = self.vir2real(*vir_pos)

        while dist < self.AGENT_RADIUS + self.CREEP_RADII[creep_type] + 1:
            vir_pos = (self.rng.randint(0, self.real_width//2)*2+1, self.rng.randint(0, self.real_width//2)*2+1)
            pos = self.vir2real(*vir_pos)
            dist = math.sqrt(
                (self.player.pos.x - pos[0])**2 + (self.player.pos.y - pos[1])**2)

        creep = Creep(
            self.CREEP_COLORS[creep_type],
            self.CREEP_RADII[creep_type],
            pos,
            (0, 0),
            self.CREEP_SPEED,
            self.CREEP_REWARD[creep_type],
            self.CREEP_TYPES[creep_type],
            self.width,
            self.height,
            0
        )

        self.creeps.add(creep)

        self.creep_counts[self.CREEP_TYPES[creep_type]] += 1

    def _direction_adjustment(self, p=0.9):
        creep_movement_list = []
        for creep in self.creeps.sprites():
            creep.speed = self.CREEP_SPEED
            creep_pos_old = (creep.pos.x, creep.pos.y)
            creep_pos_new = (creep.pos.x + creep.direction.x * creep.speed * 1, creep.pos.y + creep.direction.y * creep.speed * 1)
            vir_creep_pos_old = self.real2vir(*creep_pos_old)
            vir_creep_pos_new = self.real2vir(*creep_pos_new)

            if self.maze[vir_creep_pos_new] == 0 and vir_creep_pos_new != vir_creep_pos_old and self.rng.rand() < p:
                creep_movement = (vir_creep_pos_new[0], vir_creep_pos_new[1], vir_creep_pos_old[0], vir_creep_pos_old[1])
                creep_movement_list.append((creep, creep_movement))
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

            creep_pos_new = (creep.pos.x + creep.direction.x * creep.speed * 1, creep.pos.y + creep.direction.y * creep.speed * 1)
            vir_creep_pos_new = self.real2vir(*creep_pos_new)

            creep_movement = (vir_creep_pos_new[0], vir_creep_pos_new[1], vir_creep_pos_old[0], vir_creep_pos_old[1])
            creep_movement_list.append((creep, creep_movement))
        return creep_movement_list

    def getGameState(self):
        player_state = {'type':'player', 
                        'type_index': 0, 
                        'position': [self.player.pos.x, self.player.pos.y],
                        'velocity': [self.player.vel.x, self.player.vel.y],
                        'speed': self.AGENT_SPEED,
                        'box': [self.player.rect.top, self.player.rect.left, self.player.rect.bottom, self.player.rect.right]
                       }

        state = [player_state]
        order = list(range(len(self.creeps.sprites())))
        # self.rng.shuffle(order)
        for i in order:
            c = self.creeps.sprites()[i]
            creep_state = {'type':'creep', 
                           'type_index': self.CREEP_TYPES.index(c.TYPE) + 1, 
                           'position': [c.pos.x, c.pos.y],
                           'velocity': [c.direction.x * c.speed, c.direction.y * c.speed],
                           'speed': c.speed,
                           'box': [c.rect.top, c.rect.left, c.rect.bottom, c.rect.right]
                          }
            state.append(creep_state)

        return state, self.maze

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return (self.creep_counts['GOOD'] == 0)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
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

        for i in range(self.N_CREEPS):
            if i < self.N_CREEPS // 2:
                self._add_creep(0)
            else:
                self._add_creep(1)

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] == 1:
                    self.walls.add(Wall(self.vir2real(i, j), self.wall_width, self.wall_width, self.WALL_COLOR))

        self.score = 0
        self.ticks = 0
        self.lives = -1
        self.screen.fill(self.BG_COLOR)
        self.player.draw(self.screen)
        self.creeps.draw(self.screen)
        self.walls.draw(self.screen)

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt = 1
        self.screen.fill(self.BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()

        
        if self.dx == 0 and self.dy == 0:
            player_pos_new = (self.player.pos.x + self.player.vel.x * dt, self.player.pos.y + self.player.vel.y * dt)
        else:
            player_pos_new = (self.player.pos.x + self.dx * dt, self.player.pos.y + self.dy * dt)
        vir_player_pos_new = self.real2vir(*player_pos_new)
        player_pos = (self.player.pos.x, self.player.pos.y)
        vir_player_pos = self.real2vir(*player_pos)
        player_movement = (vir_player_pos[0], vir_player_pos[1], vir_player_pos_new[0], vir_player_pos_new[1])
                
        if self.maze[vir_player_pos_new] != 0:
            self.player.vel.x = 0
            self.player.vel.y = 0
        else:
            self.player.update(self.dx, self.dy, dt)
            # print(self.player.pos.x, self.player.pos.y, self.dx, self.dy)
            # if len(pygame.sprite.spritecollide(self.player, self.walls, False)) != 0:
            #     self.player.vel.x = 0
            #     self.player.vel.y = 0
            # player_movement = (vir_player_pos[0], vir_player_pos[1], vir_player_pos_new[0], vir_player_pos_new[1])
            
        creep_movement_list = self._direction_adjustment()
        self.creeps.update(dt)

        hits = pygame.sprite.spritecollide(self.player, self.creeps, False)
        for creep in hits:
            self.creep_counts[creep.TYPE] -= 1
            self.score += creep.reward
            creep.kill()
        for creep, creep_movement in creep_movement_list:
            if creep_movement == player_movement:
                self.creep_counts[creep.TYPE] -= 1
                self.score += creep.reward
                creep.kill()

        if self.creep_counts["GOOD"] == 0:
            self.score += self.rewards["win"]

        self.player.draw(self.screen)
        self.creeps.draw(self.screen)
        self.walls.draw(self.screen)

if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = WaterWorldMaze(width=512, maze_width=15, num_creeps=10)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(5)
        game.step(dt)
        pygame.display.update()
        # print(game.getGameState())
        if game.game_over() is True:
            print("The overall score is {}.".format(game.score))
            break