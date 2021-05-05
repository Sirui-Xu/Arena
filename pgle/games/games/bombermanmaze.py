import pygame
import sys
sys.path.append("..")
import math

from base import PyGameWrapper, Player, Creep, Wall, Bomb

from utils import vec2d, percent_round_int, generate_random_maze
from pygame.constants import K_w, K_a, K_s, K_d, K_SPACE

class BomberManMaze(PyGameWrapper):
    """
    Shot minus 1 point, hit plus 2 points.
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
            "down": K_s,
            "shoot": K_SPACE,
        }

        PyGameWrapper.__init__(self, width, width, actions=actions)
        self.BG_COLOR = (255, 255, 255)
        self.N_CREEPS = num_creeps
        self.CREEP_TYPES = ["GOOD"]
        self.CREEP_COLORS = [(40, 240, 40), (150, 95, 95)]
        radius = percent_round_int(self.wall_width, 0.4)
        # print(width, self.wall_width, self.real_width, radius)
        self.CREEP_RADII = [radius, radius]
        self.CREEP_REWARD = [
            self.rewards["positive"]]
        if NO_SPEED:
            self.CREEP_SPEED = 0
        else:
            self.CREEP_SPEED = self.wall_width
        self.AGENT_COLOR = (30, 30, 70)
        self.AGENT_SPEED = self.wall_width
        self.AGENT_RADIUS = radius
        self.AGENT_INIT_POS = (0, 0)
        self.UNIFORM_SPEED = True
        self.creep_counts = {
            "GOOD": 0,
        }

        self.WALL_COLOR = (5, 5, 30)
        self.FIX_WALL_COLOR = (30, 5, 5)
        self.BOMB_COLOR = (70, 30, 30)
        self.BOMB_RADIUS = percent_round_int(self.wall_width, 0.48)
        self.BOMB_LIFE = 8
        self.BOMB_RANGE = 2
        self.EXPLODE_COLOR = (120, 220, 180)
        self.EXPLODE_SHAPE = (self.wall_width, self.wall_width)
        self.dx = 0
        self.dy = 0
        self.shoot = 0
        self.dx_next = 0
        self.dy_next = 0
        self.shoot_next = 0
        self.bomb_dict = None
        self.player = None
        self.creeps = None
        self.walls = None
        self.bombs = None
        self.explosion = None
        self.fix_walls = None
        self.maze = None
        self.fps = fps

    def vir2real(self, x, y):
        return ((x+0.5) * self.wall_width, (y+0.5) * self.wall_width)
    
    def real2vir(self, x, y):
        return (int(x / self.wall_width - 0.5), int(y / self.wall_width - 0.5))

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    self.dx_next -= self.AGENT_SPEED

                if key == self.actions["right"]:
                    self.dx_next += self.AGENT_SPEED

                if key == self.actions["up"]:
                    self.dy_next -= self.AGENT_SPEED

                if key == self.actions["down"]:
                    self.dy_next += self.AGENT_SPEED
                
                if key == self.actions["shoot"]:
                    self.shoot_next += 1

    def _add_bomb(self):
        pos = (self.player.pos.x, self.player.pos.y)
        vir_pos = self.real2vir(*pos)
        if self.maze[vir_pos] != 0:
            return

        bomb = Bomb(
            self.BOMB_COLOR,
            self.BOMB_RADIUS,
            pos,
            self.BOMB_LIFE,
            self.BOMB_RANGE,
            self.width,
            self.height
        )
        self.bombs.add(bomb)
        self.maze[vir_pos] = 3

    def _cal_explode_pos(self, bomb_pos, bomb_info):
        bomb_range, dirs = bomb_info
        deldirs = []
        for direction in dirs:
            pos = (bomb_pos[0] + direction[0]*bomb_range*self.EXPLODE_SHAPE[0], bomb_pos[1] + direction[1]*bomb_range*self.EXPLODE_SHAPE[1])
            vir_pos = self.real2vir(*pos)
            if vir_pos[0] < 1 or vir_pos[0] >= self.real_width - 1 or vir_pos[1] < 1 or vir_pos[1] >= self.real_width - 1:
                deldirs.append(direction)
                continue
            if vir_pos[0] % 2 == 0 and vir_pos[1] % 2 == 0:
                deldirs.append(direction)
                continue
            if self.maze[vir_pos[0], vir_pos[1]] == 1:
                deldirs.append(direction)
            self.maze[vir_pos[0], vir_pos[1]] = 0
            explosion = Wall(pos, self.EXPLODE_SHAPE[0], self.EXPLODE_SHAPE[1], self.EXPLODE_COLOR)
            self.explosion.add(explosion)
        dirs = [direction for direction in dirs if direction not in deldirs]
        return (bomb_range + 1, dirs)

    def explode(self):
        bombs_pos = list(self.bomb_dict.keys())
        for bomb_pos in bombs_pos:
            if self.bomb_dict[bomb_pos][0] > self.BOMB_RANGE:
                self.bomb_dict.pop(bomb_pos)
                continue
            else:
                self.bomb_dict[bomb_pos] = self._cal_explode_pos(bomb_pos, self.bomb_dict[bomb_pos])

        hits = pygame.sprite.groupcollide(self.bombs, self.explosion, False, False)
        for bomb in hits.keys():
            if (bomb.pos.x, bomb.pos.y) not in self.bomb_dict:
                self.bomb_dict[(bomb.pos.x, bomb.pos.y)] = (1, [(0,1),(0,-1),(1,0),(-1,0)])
                bomb.kill()


    def _add_creep(self, creep_type):
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
        for creep in self.creeps.sprites():
            creep.speed = self.CREEP_SPEED
            creep_pos_old = (creep.pos.x, creep.pos.y)
            creep_pos_new = (creep.pos.x + creep.direction.x * creep.speed, creep.pos.y + creep.direction.y * creep.speed)
            vir_creep_pos_old = self.real2vir(*creep_pos_old)
            vir_creep_pos_new = self.real2vir(*creep_pos_new)

            if self.maze[vir_creep_pos_new] == 0 and vir_creep_pos_new != vir_creep_pos_old and self.rng.rand() < p:
                continue

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            feasible_directions = []
            self.rng.shuffle(directions)
            for direction in directions:
                creep_pos_new_ = (creep.pos.x + direction[0] * creep.speed, creep.pos.y + direction[1] * creep.speed)
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
        player_state = {'type':'player', 
                        'type_index': [0, -1], 
                        'position': [self.player.pos.x, self.player.pos.y],
                        'velocity': [self.player.vel.x, self.player.vel.y],
                        'speed': self.AGENT_SPEED,
                        'box': [self.player.rect.top, self.player.rect.left, self.player.rect.bottom, self.player.rect.right]
                       }

        state = [player_state]
        for c in self.creeps:
            creep_state = {'type':'creep', 
                        'type_index': [1, -1], 
                        'position': [c.pos.x, c.pos.y],
                        'velocity': [c.direction.x * c.speed, c.direction.y * c.speed],
                        'speed': c.speed,
                        'box': [c.rect.top, c.rect.left, c.rect.bottom, c.rect.right]
                        }
            state.append(creep_state)

        for b in self.bombs:
            bomb_state = {'type':'bomb', 
                        'type_index': [2, b.life], 
                        'position': [b.pos.x, b.pos.y],
                        'velocity': [0, 0],
                        'speed': 0,
                        'box': [b.rect.top, b.rect.left, b.rect.bottom, b.rect.right]
                        }
            state.append(bomb_state)

        return state, {'maze':self.maze, 'bomb_life':self.BOMB_LIFE, 'bomb_range':(self.EXPLODE_SHAPE[0]*self.BOMB_RANGE, self.EXPLODE_SHAPE[1]*self.BOMB_RANGE)}


    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return (self.creep_counts['GOOD'] == 0) or self.player is None# or self.ticks * self.wall_width / self.fps >= self.N_CREEPS * (self.width + self.height)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        self.maze = generate_random_maze(self.real_width, self.real_width, complexity=.2, density=.2)
        # print(self.maze)
        self.creep_counts = {"GOOD": 0}
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

        if self.fix_walls is None:
            self.fix_walls = pygame.sprite.Group()
        else:
            self.fix_walls.empty()

        if self.explosion is None:
            self.explosion = pygame.sprite.Group()
        else:
            self.explosion.empty()

        if self.bombs is None:
            self.bombs = pygame.sprite.Group()
        else:
            self.bombs.empty()

        for i in range(self.N_CREEPS):
            self._add_creep(0)

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if i % 2 == 0 and j % 2 == 0:
                    self.maze[i, j] = 2
                    self.fix_walls.add(Wall(self.vir2real(i, j), self.wall_width, self.wall_width, self.FIX_WALL_COLOR))
                elif self.maze[i, j] == 1:
                    self.walls.add(Wall(self.vir2real(i, j), self.wall_width, self.wall_width, self.WALL_COLOR))


        self.score = 0
        self.ticks = 0
        self.lives = -1
        self.dx_next, self.dy_next, self.shoot_next = 0, 0, 0
        self.after_explosion = False
        self.screen.fill(self.BG_COLOR)
        self.player.draw(self.screen)
        self.creeps.draw(self.screen)
        self.walls.draw(self.screen)
        self.fix_walls.draw(self.screen)

    def step(self):
        """
            Perform one step of game emulation.
        """
        self.screen.fill(self.BG_COLOR)
        if self.player is None:
            return
        if self.ticks % self.fps == 0:
            self.bomb_dict = {}
            self.after_explosion = False
            self.explosion.empty()
            self.score += self.rewards["tick"]
            self.dx, self.dy, self.shoot = self.dx_next, self.dy_next, self.shoot_next
            self.dx_next, self.dy_next, self.shoot_next = 0, 0, 0
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
            if self.shoot > 0:
                self._add_bomb()
                self.player.vel.x = 0
                self.player.vel.y = 0
            self._direction_adjustment()
            for bomb in self.bombs:
                if bomb.life <= 1:
                    self.bomb_dict[(bomb.pos.x, bomb.pos.y)] = (0, [(0,1),(0,-1),(1,0),(-1,0)])
                    bomb.kill()

        if self.dx_next == 0 and self.dy_next == 0 and self.shoot_next == 0:
            self._handle_player_events()
        else:
            pygame.event.pump()

        self.player.update(self.dx, self.dy, 1 / self.fps)   
        self.creeps.update(1 / self.fps)     
        self.bombs.update(1 / self.fps)   

        hits = pygame.sprite.spritecollide(self.player, self.creeps, False)

        if len(self.bomb_dict) > 0:
            self.explode()

        self.player.draw(self.screen)
        self.bombs.draw(self.screen)
        self.explosion.draw(self.screen)
        self.creeps.draw(self.screen)
        self.walls.draw(self.screen)
        self.fix_walls.draw(self.screen)

        if len(hits) != 0:
            self.player.kill()
            self.player = None
            return

        hits = pygame.sprite.spritecollide(self.player, self.explosion, False)
        if len(hits) != 0:
            self.player.kill()
            self.player = None

        hits = pygame.sprite.groupcollide(self.creeps, self.explosion, True, False)
        for creep in hits.keys():
            self.creep_counts[creep.TYPE] -= 1
            self.score += creep.reward
            # self._add_creep(1)

        if self.creep_counts["GOOD"] == 0:
            self.score += self.rewards["win"]

        hits = pygame.sprite.groupcollide(self.walls, self.explosion, True, False)
        self.ticks += 1

if __name__ == "__main__":
    import numpy as np
    pygame.init()
    game = BomberManMaze(width=512, maze_width=15, num_creeps=1)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()
    fps = 5
    while True:
        for i in range(game.fps):
            dt = game.clock.tick_busy_loop(fps*game.fps)
            game.step()
            pygame.display.update()

        if game.game_over() is True:
            print("The overall score is {}.".format(game.score))
            break
        print(game.getGameState(), '\n')