import pygame
import sys
import math
import random
import numpy as np
from .base import PyGameWrapper
from .base import Agent, Bombv, Blast, Enemy, Obstacle, Projectile, Reward

from .base import vec2d
from pygame.constants import K_w, K_a, K_s, K_d, K_j, K_SPACE


class Arena(PyGameWrapper):
    def __init__(self,
                 width=1280,
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
                 enemy_speed=0,
                 projectile_speed=32,
                 visualize=True,
                 ):

        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s,
            "shoot": K_j,
            "fire": K_SPACE
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)
        self.BG_COLOR = (0, 0, 0)
        if not (num_enemies >= 0 and num_coins > 0 and num_bombs >= 0 and 
                num_projectiles >= 0 and num_obstacles >= 0):
            raise Exception('Need a positive number of objects or stuff.')
        self.N_ENEMIES = num_enemies
        self.N_REWARDS = num_coins
        self.N_BOMBS = num_bombs
        self.N_PROJECTILES = num_projectiles
        self.N_OBSTACLES = num_obstacles
        if not (object_size >= 2):
            raise Exception('The objects must have at least two pixel width and height.')
        if not (obstacle_size >= 4 + object_size):
            raise Exception('The obstacle must be four pixels larger than the object.') 
        self.OBJECT_SIZE = object_size
        self.OBSTACLE_SIZE = obstacle_size
        if not (enemy_speed < object_size and agent_speed < object_size and projectile_speed <= object_size):
            raise Exception('Speed must less than object size.')
        if not (agent_speed >= 1 and projectile_speed >= 1):
            raise Exception('Agent and projectile must have speed must larger than 1.')
        if enemy_speed < 1:
            enemy_speed = 0
            print("enemies' speed is set to zero") 
        self.ENEMY_SPEED = enemy_speed
        self.AGENT_SPEED = agent_speed
        self.PROJECTILE_SPEED = projectile_speed
        if (explosion_radius // object_size) * object_size != explosion_radius:
            explosion_radius = (explosion_radius // object_size) * object_size
            print("explosion radius is set to {} for convenience".format(explosion_radius))
        self.EXPLOSION_MAX_STEP = explosion_max_step
        self.EXPLOSION_RADIUS = explosion_radius
        if not (self.AGENT_SPEED * self.EXPLOSION_MAX_STEP > self.EXPLOSION_RADIUS):
            raise Exception('Agent cannot escape the explosion just setting off, need larger speed or longer explosion_max_step or smaller explosion_radius')
        self.dx, self.dy, self.shoot, self.fire = 0, 0, 0, 0
        self.player = None
        self.enemies = None
        self.reward_objects = None
        self.bombs = None
        self.projectiles = None
        self.obstacles = None
        self.blasts = None

        self.visualize = visualize
        self.fps = 20


    def _handle_player_events(self):
        self.dx, self.dy, self.shoot, self.fire = 0, 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[self.actions["left"]]:
            self.dx -= self.AGENT_SPEED
        elif keys[self.actions["right"]]:
            self.dx += self.AGENT_SPEED
        elif keys[self.actions["up"]]:
            self.dy -= self.AGENT_SPEED
        elif keys[self.actions["down"]]:
            self.dy += self.AGENT_SPEED
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key = event.key

                if self.dx == 0 and self.dy == 0:
                    if key == self.actions["left"]:
                        self.dx -= self.AGENT_SPEED

                    if key == self.actions["right"]:
                        self.dx += self.AGENT_SPEED

                    if key == self.actions["up"]:
                        self.dy -= self.AGENT_SPEED

                    if key == self.actions["down"]:
                        self.dy += self.AGENT_SPEED

                if key == self.actions["shoot"]:
                    self.shoot += 1

                if key == self.actions["fire"]:
                    self.fire += 1

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
    def generate_random_maze(self, width, height, num):
        r"""Generate a random maze array. 
        
        It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
        is ``1`` and for free space is ``0``. 
        
        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """
        # Only odd shapes
        shape = (width, height)
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        t = 0
        # Fill borders
        # Z[0, :] = Z[-1, :] = 1
        # Z[:, 0] = Z[:, -1] = 1
        # Make aisles
        while True:
           y, x = self.rng.randint(0, shape[0]), self.rng.randint(0, shape[1]) 
           if Z[y, x] == 1:
               continue
           Z[y, x] = 1
           t += 1
           if t == num:
               break
           while True:
               neighbours = []
               if x > 1:             neighbours.append((y, x - 2))
               if x < shape[1] - 2:  neighbours.append((y, x + 2))
               if y > 1:             neighbours.append((y - 2, x))
               if y < shape[0] - 2:  neighbours.append((y + 2, x))
               flag = False
               if len(neighbours):
                   y_,x_ = neighbours[self.rng.randint(0, len(neighbours))]
                   if Z[y_, x_] == 0:
                       Z[y_, x_] = 1
                       t += 1
                       flag = True
                       if t == num:
                           break
                       if Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] == 0:
                           Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                           t += 1
                           flag = True
                           if t == num:
                               break
                   y, x = y_, x_
                   if not flag:
                       break
           if t == num:
               break
        # while True:
        #     y, x = self.rng.randint(0, shape[0]), self.rng.randint(0, shape[1]) 
        #     if Z[y, x] == 1:
        #         continue
        #     Z[y, x] = 1
        #     t += 1
        #     if t == num:
        #         break
        #     while True:
        #         neighbours = []
        #         if x > 0 and Z[y, x - 1] == 0:             neighbours.append((y, x - 1))
        #         if x < shape[1] - 1 and Z[y, x + 1] == 0:  neighbours.append((y, x + 1))
        #         if y > 0 and Z[y - 1, x] == 0:             neighbours.append((y - 1, x))
        #         if y < shape[0] - 1 and Z[y + 1, x] == 0:  neighbours.append((y + 1, x))
        #         if len(neighbours):
        #             y_,x_ = neighbours[self.rng.randint(0, len(neighbours))]
        #             if Z[y_, x_] == 0:
        #                 Z[y_, x_] = 1
        #                 t += 1
        #                 if t == num:
        #                     break
        #             y, x = y_, x_
        #         else:
        #             break
        #     if t == num:
        #         break
        return Z.astype(int)

    def _add_obstacles(self, shape, edge_x, edge_y):
        if self.N_OBSTACLES == 0:
            self.maze = np.zeros((self.width // shape, self.height // shape))
            return
        else:
            self.maze = self.generate_random_maze(self.width // shape, self.height // shape, num=self.N_OBSTACLES)
        for i in range(self.width // shape):
            for j in range(self.height // shape):
                obstacle = None
                pos = (i, j)
                if self.maze[pos] == 1:
                    real_pos = (edge_x + (pos[0] + 0.5) * shape, edge_y + (pos[1] + 0.5) * shape)
                    obstacle = Obstacle(
                        shape // 2, real_pos
                    )
                    self.obstacles.add(obstacle)

    def _add_agent(self, shape, edge_x, edge_y):
        pos = (self.rng.randint(0, (self.width // shape)), self.rng.randint(0, (self.height // shape)))
        while(self.maze[pos] > 0):
            pos = (self.rng.randint(0, (self.width // shape)), self.rng.randint(0, (self.height // shape)))
        
        AGENT_INIT_POS = (edge_x + (pos[0] + 0.5) * shape, edge_y + (pos[1] + 0.5) * shape)
        self.player.pos = vec2d(AGENT_INIT_POS)
        self.player.rect.center = AGENT_INIT_POS


    def _add_enemies(self, shape, edge_x, edge_y):
        for i in range(self.N_ENEMIES):
            enemy = None
            for t in range(10):
                pos = (self.rng.randint(0, (self.width // shape)), self.rng.randint(0, (self.height // shape)))
                if self.maze[pos] == 0:
                    real_pos = (edge_x + (pos[0] + 0.5) * shape, edge_y + (pos[1] + 0.5) * shape)
                    dist = math.sqrt((self.player.pos.x - real_pos[0])**2 + (self.player.pos.y - real_pos[1])**2)
                    if dist > 2 * self.OBJECT_SIZE:
                        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                        enemy = Enemy(
                            self.OBJECT_SIZE // 2,
                            real_pos,
                            directions[self.rng.choice(list(range(4)))],
                            self.ENEMY_SPEED,
                            self.width,
                            self.height,
                        )
                        self.enemies.add(enemy)
                        break
            if t >= 10:
                print("WARNING: Need a bigger map!")

    def _add_rewards(self, shape, edge_x, edge_y):
        for i in range(self.N_REWARDS):
            reward = None
            for t in range(10):
                pos = (self.rng.randint(0, (self.width // shape)), self.rng.randint(0, (self.height // shape)))
                if self.maze[pos] == 0:
                    real_pos = (edge_x + (pos[0] + 0.5) * shape, edge_y + (pos[1] + 0.5) * shape)
                    dist = math.sqrt((self.player.pos.x - real_pos[0])**2 + (self.player.pos.y - real_pos[1])**2)
                    if dist > 2 * self.OBJECT_SIZE:
                        reward = Reward(
                            self.OBJECT_SIZE // 2,
                            real_pos,
                            1,
                        )
                        self.reward_objects.add(reward)
                        break
            if t >= 10:
                print("WARNING: Need a bigger map!")
    
    def add_projectile(self):
        if len(self.projectiles) < self.N_PROJECTILES and self.shoot > 0:
            projectile = Projectile(
                                    self.OBJECT_SIZE // 2, 
                                    (self.player.pos.x, self.player.pos.y), 
                                    (self.player.direction.x, self.player.direction.y), 
                                    self.PROJECTILE_SPEED, 
                                    self.width,
                                    self.height)
            self.projectiles.add(projectile)

    def add_bomb(self):
        if len(self.bombs) < self.N_BOMBS and self.fire > 0:
            pos = (self.player.pos.x, self.player.pos.y)

            bomb = Bombv(
                self.OBJECT_SIZE // 2,
                pos,
                self.EXPLOSION_MAX_STEP,
                self.EXPLOSION_RADIUS,
            )

            if len(pygame.sprite.spritecollide(bomb, self.bombs, False)) == 0:
                self.bombs.add(bomb)
            else:
                bomb.kill()
    
    def _cal_blast_pos(self, bomb):
        for i in range(-self.EXPLOSION_RADIUS // self.OBJECT_SIZE, self.EXPLOSION_RADIUS // self.OBJECT_SIZE + 1):
            for j in range(-self.EXPLOSION_RADIUS // self.OBJECT_SIZE, self.EXPLOSION_RADIUS // self.OBJECT_SIZE + 1):
                if abs(i) + abs(j) > self.EXPLOSION_RADIUS // self.OBJECT_SIZE:
                    continue
                vir_pos = (bomb.pos.x + i*self.OBJECT_SIZE, bomb.pos.y + j*self.OBJECT_SIZE)
                if vir_pos[0] < self.OBJECT_SIZE / 2 or vir_pos[0] >= self.width - self.OBJECT_SIZE / 2 or vir_pos[1] < self.OBJECT_SIZE / 2 or vir_pos[1] >= self.height - self.OBJECT_SIZE / 2:
                    continue
                blast = Blast(self.OBJECT_SIZE // 2, vir_pos)
                self.blasts.add(blast)

    def blast(self):
        # self.blasts.empty()
        for bomb in self.bombs:
            if bomb.life < 1:
                self._cal_blast_pos(bomb)
                bomb.kill()

        hits = pygame.sprite.groupcollide(self.bombs, self.blasts, True, False)
        while len(hits) > 0:
            for bomb in hits.keys():
                self._cal_blast_pos(bomb)
            hits = pygame.sprite.groupcollide(self.bombs, self.blasts, True, False)

        hits = pygame.sprite.groupcollide(self.bombs, self.projectiles, True, False)
        while len(hits) > 0:
            for bomb in hits.keys():
                self._cal_blast_pos(bomb)
            hits = pygame.sprite.groupcollide(self.bombs, self.blasts, True, False)

    def getGameState(self):
        state = []
        if self.player is not None:
            player_state = {'type':'agent', 
                            'type_index': [0, 0, -1, -1], 
                            'position': [self.player.pos.x, self.player.pos.y],
                            'radius': self.player.radius,
                            'velocity': [self.AGENT_SPEED * self.player.direction.x, self.AGENT_SPEED * self.player.direction.y],
                            'speed': self.AGENT_SPEED,
                            'box': [self.player.rect.left, self.player.rect.top, self.player.rect.right, self.player.rect.bottom],
                        }
            state.append(player_state)
        for c in self.reward_objects.sprites():
            reward_state = {'type':'reward', 
                           'type_index': [1, c.reward, -1, -1], 
                           'position': [c.pos.x, c.pos.y],
                           'radius': c.radius,
                           'velocity': [0, 0],
                           'speed': 0,
                           'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                          }
            state.append(reward_state)
        for c in self.obstacles.sprites():
            obstacle_state = {'type':'obstacle', 
                           'type_index': [2, -1, -1, -1], 
                           'position': [c.pos.x, c.pos.y],
                           'radius': c.radius,
                           'velocity': [0, 0],
                           'speed': 0,
                           'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                          }
            state.append(obstacle_state)
        for c in self.blasts.sprites():
            blast_state = {'type':'blast', 
                                'type_index': [3, -1, int(c.life), -1], 
                                'position': [c.pos.x, c.pos.y],
                                'radius': c.radius,
                                'velocity': [0, 0],
                                'speed': 0,
                                'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                            }
            state.append(blast_state)
        for c in self.enemies.sprites():
            enemy_state = {'type':'enemy', 
                           'type_index': [4, -1, -1, -1], 
                           'position': [c.pos.x, c.pos.y],
                           'radius': c.radius,
                           'velocity': [c.direction.x * c.speed, c.direction.y * c.speed],
                           'speed': c.speed,
                           'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                          }
            state.append(enemy_state)
        for c in self.bombs.sprites():
            bomb_state = {'type':'bomb', 
                           'type_index': [5, 0, int(c.life), c.explode_range], 
                           'position': [c.pos.x, c.pos.y],
                           'radius': c.radius,
                           'velocity': [0, 0],
                           'speed': 0,
                           'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                          }
            state.append(bomb_state)
        for c in self.projectiles.sprites():
            projectile_state = {'type':'projectile', 
                                'type_index': [6, -1, -1, -1], 
                                'position': [c.pos.x, c.pos.y],
                                'radius': c.radius,
                                'velocity': [c.direction.x * c.speed, c.direction.y * c.speed],
                                'speed': c.speed,
                                'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                            }
            state.append(projectile_state)

        return {'local':state, 'global':{'ticks': self.ticks, 'shape': [self.width, self.height],
                                         'score': self.score, 'bombs_left':self.N_BOMBS - len(self.bombs), 'projectiles_left': self.N_PROJECTILES - len(self.projectiles)}}

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return len(self.reward_objects) == 0 or self.player == None# or self.ticks >= self.duration# or self.ticks > self.N_ENEMIES * (self.width + self.height)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        # self.assigned_values = list(range(1, self.N_ENEMIES+1))
        # self.assigned_values = [_ / self.assigned_values[-1] for _ in self.assigned_values]

        AGENT_INIT_POS = (0, 0)

        if self.player is None:
            self.player = Agent(
                self.OBJECT_SIZE // 2,
                AGENT_INIT_POS, self.AGENT_SPEED,
                self.width, self.height,
            )
        else:
            self.player.pos = vec2d(AGENT_INIT_POS)
            self.player.rect.center = AGENT_INIT_POS

        if self.enemies is None:
            self.enemies = pygame.sprite.Group()
        else:
            self.enemies.empty()

        if self.reward_objects is None:
            self.reward_objects = pygame.sprite.Group()
        else:
            self.reward_objects.empty()
        
        if self.obstacles is None:
            self.obstacles = pygame.sprite.Group()
        else:
            self.obstacles.empty()

        if self.blasts is None:
            self.blasts = pygame.sprite.Group()
        else:
            self.blasts.empty()

        if self.bombs is None:
            self.bombs = pygame.sprite.Group()
        else:
            self.bombs.empty()

        if self.projectiles is None:
            self.projectiles = pygame.sprite.Group()
        else:
            self.projectiles.empty()

        shape = self.OBSTACLE_SIZE
        edge_x = (self.width - (self.width // shape) * shape) / 2
        edge_y = (self.height - (self.height // shape) * shape) / 2

        self._add_obstacles(shape, edge_x, edge_y)
        self._add_agent(shape, edge_x, edge_y)
            
        self._add_enemies(shape, edge_x, edge_y)
        self._add_rewards(shape, edge_x, edge_y)
        
        self.score = 0
        self.ticks = 0
        if self.visualize:
            self.draw()
    
    def draw(self):
        self.screen.fill(self.BG_COLOR)
        self.obstacles.draw(self.screen)
        self.bombs.draw(self.screen)
        self.projectiles.draw(self.screen)
        self.blasts.draw(self.screen)
        self.reward_objects.draw(self.screen)
        self.enemies.draw(self.screen)
        self.player.draw(self.screen)

    def step(self):
        """
            Perform one step of game emulation.
        """
        self.score += -0.001

        self._handle_player_events()
        self.player.update(self.dx, self.dy, self.obstacles)
        self.add_projectile()
        self.add_bomb()
        self.enemies.update(self.obstacles)
        self.projectiles.update()
        self.bombs.update()
        self.blasts.update()

        hits = pygame.sprite.spritecollide(self.player, self.reward_objects, True)
        for node in hits:
            self.score += node.reward
        
        self.blast()
        hits = pygame.sprite.groupcollide(self.enemies, self.blasts, True, False)
        hits = pygame.sprite.groupcollide(self.obstacles, self.blasts, True, False)
        hits = pygame.sprite.groupcollide(self.projectiles, self.obstacles, True, True)
        for bullet in hits.keys():
            for obstacles in hits[bullet]:
                self.blasts.add(Blast(self.OBJECT_SIZE // 2, (obstacles.pos.x, obstacles.pos.y)))
        hits = pygame.sprite.groupcollide(self.projectiles, self.blasts, True, False)
        hits = pygame.sprite.groupcollide(self.projectiles, self.enemies, True, True)
        for bullet in hits.keys():
            for enemy in hits[bullet]:
                self.blasts.add(Blast(self.OBJECT_SIZE // 2, (enemy.pos.x, enemy.pos.y)))
        hits_enemies = pygame.sprite.spritecollide(self.player, self.enemies, False)
        hits_blasts = pygame.sprite.spritecollide(self.player, self.blasts, False)
        hits_projectiles = pygame.sprite.spritecollide(self.player, self.projectiles, False)

        if self.visualize:
            self.draw()

        self.ticks += 1
        if len(hits_enemies) != 0 or len(hits_blasts) != 0 or len(hits_projectiles) != 0:
            self.player.kill()
            self.player = None


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = ARENA(width=512, height=512)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
        if game.game_over() is True:
            print("The overall score is {}.".format(game.score))
            break
        print(game.getGameState(), '\n')
