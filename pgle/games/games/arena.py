import pygame
import sys
import math
import random
from ..base import PyGameWrapper
from ..base import Agent, Bombv, Blast, Enemy, Obstacle, Projectile, Reward

from ..utils import vec2d, percent_round_int
from pygame.constants import K_w, K_a, K_s, K_d, K_j, K_SPACE


class ARENA(PyGameWrapper):
    """
    Parameters
    ----------
    width : int
        Screen width.
    height : int
        Screen height, recommended to be same dimension as width.
    num_enemies : int (default: 3)
        The number of enemies on the screen at once.
    num_bombs : int (default: 3)
        The number of bombs that agent can placed.
    num_projectiles : int (default: 3)
        The number of projectiles that agent can fired.
    """

    def __init__(self,
                 width=256,
                 height=256,
                 num_reward=5,
                 num_enemies=5,
                 num_bombs=3,
                 num_projectiles=3,
                 num_obstacles=20,
                 visualize=True,
                 fps=20):

        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s,
            "shoot": K_j,
            "fire": K_SPACE
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)
        self.BG_COLOR = (255, 255, 255)
        self.N_ENEMIES = num_enemies
        self.N_REWARDS = num_reward
        self.N_BOMBS = num_bombs
        self.N_PROJECTILES = num_projectiles
        self.N_OBSTACLES = num_obstacles
        self.SHAPE = 30
        self.SPEED = 30
        self.ENEMY_SPEED = self.SPEED * 2
        self.AGENT_SPEED = self.SPEED * 2
        self.BULLET_SPEED = self.SPEED * 4
        self.BOMB_LIFE = 5
        self.BOMB_RANGE = 2 
        self.dx, self.dy, self.shoot, self.fire = 0, 0, 0, 0
        self.agent = None
        self.enemies = None
        self.reward_nodes = None
        self.bombs = None
        self.projectiles = None
        self.obstacles = None
        self.blasts = None

        self.visualize = visualize
        self.fps = 25

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

                if key == self.actions["shoot"]:
                    self.shoot += 1

                if key == self.actions["fire"]:
                    self.fire += 1

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


    def _add_obstacle(self):
        obstacle = None
        pos = (0, 0)
        dist = 0.0
        for t in range(10):
            radius = 0.5 * self.SHAPE + 1
            pos = self.rng.uniform(radius, self.height - radius, size=2)
            dist = math.sqrt(
                (self.agent.pos.x - pos[0])**2 + (self.agent.pos.y - pos[1])**2)
            obstacle = Obstacle(
                pos, self.SHAPE // 2
            )
            if len(pygame.sprite.spritecollide(obstacle, self.obstacles, False)) > 0:
                obstacle.kill()
                continue
            else:
                if dist > 2 * self.SHAPE:
                    self.obstacles.add(obstacle)
                    return True
                else:
                    continue
        return False

    
    def _add_enemy(self):
        enemy = None
        pos = (0, 0)
        dist = 0.0
        for t in range(10):
            radius = 0.5 * self.SHAPE + 1
            pos = self.rng.uniform(radius, self.height - radius, size=2)
            dist = math.sqrt(
                (self.agent.pos.x - pos[0])**2 + (self.agent.pos.y - pos[1])**2)

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            enemy = Enemy(
                self.SHAPE // 2,
                pos,
                directions[self.rng.choice(list(range(4)))],
                self.ENEMY_SPEED,
                self.width,
                self.height,
            )
            if len(pygame.sprite.spritecollide(enemy, self.obstacles, False)) > 0:
                enemy.kill()
                continue
            else:
                if dist > 2 * self.SHAPE:
                    self.enemies.add(enemy)
                    return True
                else:
                    continue
        return False

    def _add_reward(self, shape, reward):
        node = None
        pos = (0, 0)
        dist = 0.0
        for t in range(10):
            radius = shape / 2 + 1
            pos = self.rng.uniform(radius, self.height - radius, size=2)
            dist = math.sqrt(
                (self.agent.pos.x - pos[0])**2 + (self.agent.pos.y - pos[1])**2)

            node = Reward(
                pos,
                shape // 2,
                reward,
            )
            if len(pygame.sprite.spritecollide(node, self.obstacles, False)) > 0:
                node.kill()
                continue
            else:
                if dist > shape + self.SHAPE:
                    self.reward_nodes.add(node)
                    return True
                else:
                    continue
        return False
    
    def add_projectile(self):
        if len(self.projectiles) < self.N_PROJECTILES and self.shoot > 0:
            projectile = Projectile(
                                    self.SHAPE // 2, 
                                    (self.agent.pos.x, self.agent.pos.y), 
                                    (self.agent.direction.x, self.agent.direction.y), 
                                    self.BULLET_SPEED, 
                                    self.width,
                                    self.height)
            self.projectiles.add(projectile)

    def add_bomb(self):
        if len(self.bombs) < self.N_BOMBS and self.fire > 0:
            pos = (self.agent.pos.x, self.agent.pos.y)

            bomb = Bombv(
                self.SHAPE // 2,
                pos,
                self.BOMB_LIFE,
                self.BOMB_RANGE,
            )

            if len(pygame.sprite.spritecollide(bomb, self.bombs, False)) == 0:
                self.bombs.add(bomb)
            else:
                bomb.kill()
    
    def _cal_blast_pos(self, bomb):
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        for bomb_range in range(self.BOMB_RANGE+1):
            for direction in dirs:
                vir_pos = (bomb.pos.x + direction[0]*bomb_range*self.SHAPE, bomb.pos.y + direction[1]*bomb_range*self.SHAPE)
                if vir_pos[0] < self.SHAPE / 2 or vir_pos[0] >= self.width - self.SHAPE / 2 or vir_pos[1] < self.SHAPE / 2 or vir_pos[1] >= self.height - self.SHAPE / 2:
                    continue
                blast = Blast(vir_pos, self.SHAPE // 2)
                self.blasts.add(blast)

    def blast(self):
        # self.blasts.empty()
        for bomb in self.bombs:
            if bomb.life < 1 / self.fps:
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
        player_state = {'type':'player', 
                        'type_index': [0, -1], 
                        'position': [self.agent.pos.x, self.agent.pos.y],
                        'velocity': [self.agent.vel.x / self.fps, self.agent.vel.y / self.fps],
                        'speed': self.AGENT_SPEED / self.fps,
                        'box': [self.agent.rect.left, self.agent.rect.top, self.agent.rect.right, self.agent.rect.bottom],
                       }
        state = [player_state]
        order = list(range(len(self.enemies.sprites())))
        # self.rng.shuffle(order)
        for i in order:
            c = self.enemies.sprites()[i]
            enemy_state = {'type':'enemy', 
                           'type_index': [1, self.enemy_TYPES.index(c.TYPE)], 
                           'position': [c.pos.x, c.pos.y],
                           'velocity': [c.direction.x * c.speed / self.fps, c.direction.y * c.speed / self.fps],
                           'speed': c.speed / self.fps,
                           'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                           '_jitter_speed': c.jitter_speed,
                           '_type': c.TYPE,
                          }
            state.append(enemy_state)

        return {'local':state, 'global':{'ticks': self.ticks, 'shape': [self.width, self.height],
                                         'score': self.score}}

    def loadGameState(self, state):
        self.enemy_counts = {"GOOD": 0, "BAD": 0}
        if self.enemies is None:
            self.enemies = pygame.sprite.Group()
        else:
            self.enemies.empty()
        for info in state["local"]:
            if info["type"] == "player":
                self.AGENT_INIT_POS = info["position"]
                if self.agent is None:
                    self.agent = Player(
                        self.AGENT_RADIUS, self.AGENT_COLOR,
                        self.AGENT_SPEED, self.AGENT_INIT_POS,
                        self.width, self.height,
                        self.UNIFORM_SPEED
                    )

                else:
                    self.agent.pos = vec2d(self.AGENT_INIT_POS)
                    self.agent.vel = vec2d(info["velocity"])
                    self.agent.rect.center = self.AGENT_INIT_POS
            if info["type"] == "enemy":
                enemy_type = info["type_index"][1]
                enemy = enemy(
                    self.enemy_COLORS[enemy_type],
                    self.enemy_RADII[enemy_type],
                    info["position"],
                    info["velocity"],
                    info["speed"] * self.fps,
                    self.enemy_REWARD[enemy_type],
                    self.enemy_TYPES[enemy_type],
                    self.width,
                    self.height,
                    info["_jitter_speed"]
                )

                self.enemies.add(enemy)

                self.enemy_counts[self.enemy_TYPES[enemy_type]] += 1

        self.score = state["global"]["score"]
        self.ticks = state["global"]["ticks"]
        self.lives = -1
        # self.screen.fill(self.BG_COLOR)
        # self.agent.draw(self.screen)
        # self.enemies.draw(self.screen)

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return len(self.reward_nodes) == 0 or self.agent == None # or self.ticks > self.N_ENEMIES * (self.width + self.height)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        AGENT_INIT_POS = self.rng.uniform(self.SHAPE / 2 + 1, self.height - self.SHAPE / 2 - 1, size=2)

        if self.agent is None:
            self.agent = Agent(
                self.SHAPE // 2,
                self.AGENT_SPEED, AGENT_INIT_POS,
                self.width, self.height,
            )
        else:
            self.agent.pos = vec2d(AGENT_INIT_POS)
            self.agent.rect.center = AGENT_INIT_POS

        self.assigned_values = list(range(1, self.N_ENEMIES+1))
        self.assigned_values = [_ / self.assigned_values[-1] for _ in self.assigned_values]

        if self.enemies is None:
            self.enemies = pygame.sprite.Group()
        else:
            self.enemies.empty()

        if self.reward_nodes is None:
            self.reward_nodes = pygame.sprite.Group()
        else:
            self.reward_nodes.empty()
        
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

        # edge_x = (self.width - (self.width // self.SHAPE - 2) * self.SHAPE) // 2
        # edge_y = (self.height - (self.height // self.SHAPE - 2) * self.SHAPE) // 2

        # for i in range(self.width // self.SHAPE):
        #     for j in range(self.height // self.SHAPE):
        #         if i == 0 and j == 0:
        #             self.fix_obstacles.add(obstacle((edge_x / 2, edge_y / 2), edge_x, edge_y, color=(0,0,0), FIXED=True))
        #         elif i == 0 and j == self.height // self.SHAPE - 1:
        #             self.fix_obstacles.add(obstacle((edge_x / 2, self.height - edge_y / 2), edge_x, edge_y, color=(0,0,0), FIXED=True))
        #         elif i == self.width // self.SHAPE - 1 and j == 0:
        #             self.fix_obstacles.add(obstacle((self.width - edge_x / 2, edge_y / 2), edge_x, edge_y, color=(0,0,0), FIXED=True))
        #         elif i == self.width // self.SHAPE - 1 and j == self.height // self.SHAPE - 1:
        #             self.fix_obstacles.add(obstacle((self.width - edge_x / 2, self.height - edge_y / 2), edge_x, edge_y, color=(0,0,0), FIXED=True))
        #         elif i == 0:
        #             self.fix_obstacles.add(obstacle((edge_x / 2, edge_y + (j - 0.5) * self.SHAPE), edge_x, self.SHAPE, color=(0,0,0), FIXED=True))
        #         elif i == self.width // self.SHAPE - 1:
        #             self.fix_obstacles.add(obstacle((self.width - edge_x / 2, edge_y + (j - 0.5) * self.SHAPE), edge_x, self.SHAPE, color=(0,0,0), FIXED=True))
        #         elif j == 0:
        #             self.fix_obstacles.add(obstacle((edge_x + (i - 0.5) * self.SHAPE, edge_y / 2), self.SHAPE, edge_y, color=(0,0,0), FIXED=True))
        #         elif j == self.height // self.SHAPE - 1:
        #             self.fix_obstacles.add(obstacle((edge_x + (i - 0.5) * self.SHAPE, self.height - edge_y / 2), self.SHAPE, edge_y, color=(0,0,0), FIXED=True))
        #         else:
        #             pass
        #             # self.background.add(obstacle((edge_x + (i - 0.5) * self.SHAPE, edge_y + (j - 0.5) * self.SHAPE), self.SHAPE, self.SHAPE, color=(0,0,0), FIXED=True, GRASS=True))

        # # self.fix_obstacles.add(obstacle((self.SHAPE / 2, self.height / 2), self.SHAPE, self.height, color=(0,0,0), FIXED=True))
        # # self.fix_obstacles.add(obstacle((self.width - self.SHAPE / 2, self.height / 2), self.SHAPE, self.height, color=(0,0,0), FIXED=True))
        # # self.fix_obstacles.add(obstacle((self.width / 2, self.SHAPE / 2), self.width, self.SHAPE, color=(0,0,0), FIXED=True))
        # # self.fix_obstacles.add(obstacle((self.width / 2, self.height - self.SHAPE / 2), self.width, self.SHAPE, color=(0,0,0), FIXED=True))

        for i in range(self.N_ENEMIES):
            FLAG = self._add_enemy()
            if FLAG is False:
                print("need a bigger map")

        for i in range(self.N_OBSTACLES):
            FLAG = self._add_obstacle()
            if FLAG is False:
                print("need a bigger map")

        for i in range(self.N_REWARDS):
            FLAG = self._add_reward(int(self.SHAPE * (self.assigned_values[i] + 1)), self.assigned_values[i])
            if FLAG is False:
                print("need a bigger map")
        
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
        self.reward_nodes.draw(self.screen)
        self.enemies.draw(self.screen)
        self.agent.draw(self.screen)

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt = 1 / self.fps
        self.score += -0.001

        self._handle_player_events()
        self.agent.update(self.dx, self.dy, dt, self.obstacles)
        self.add_projectile()
        self.add_bomb()
        self.enemies.update(dt, self.obstacles)
        self.projectiles.update(dt)
        self.bombs.update(dt)
        self.blasts.update(dt)

        hits = pygame.sprite.spritecollide(self.agent, self.reward_nodes, True)
        for node in hits:
            self.score += node.reward
        
        self.blast()
        hits = pygame.sprite.groupcollide(self.enemies, self.blasts, True, False)
        hits = pygame.sprite.groupcollide(self.obstacles, self.blasts, True, False)
        hits = pygame.sprite.groupcollide(self.projectiles, self.obstacles, True, True)
        for bullet in hits.keys():
            for obstacles in hits[bullet]:
                self.blasts.add(Blast((obstacles.pos.x, obstacles.pos.y), self.SHAPE // 5))
        hits = pygame.sprite.groupcollide(self.projectiles, self.blasts, True, False)
        hits = pygame.sprite.groupcollide(self.projectiles, self.enemies, True, True)
        for bullet in hits.keys():
            for enemy in hits[bullet]:
                self.blasts.add(Blast((enemy.pos.x, enemy.pos.y), self.SHAPE // 5))

        hits = pygame.sprite.spritecollide(self.agent, self.enemies, True)
        if len(hits) > 0:
            self.agent.kill()
            self.agent = None
            return

        hits = pygame.sprite.spritecollide(self.agent, self.blasts, False)
        if len(hits) != 0:
            self.agent.kill()
            self.agent = None
            return

        if self.visualize:
            self.draw()
        self.ticks += dt


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