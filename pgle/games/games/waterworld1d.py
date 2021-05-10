import pygame
import sys
import math

from ..base import PyGameWrapper, Player, Creep, Wall

from ..utils import vec2d, percent_round_int
from pygame.constants import K_w, K_a, K_s, K_d


class WaterWorld1d(PyGameWrapper):
    """
    Based Karpthy's WaterWorld in `REINFORCEjs`_.
    .. _REINFORCEjs: https://github.com/karpathy/reinforcejs
    Parameters
    ----------
    width : int
        Screen width.
    height : int
        Screen height, recommended to be same dimension as width.
    num_creeps : int (default: 3)
        The number of creeps on the screen at once.
    UNIFORM_SPEED : bool (default: false)
        The agent has an uniform speed or not
    NO_SPEED : bool (default: false)
        whether the node can move.
    """

    def __init__(self,
                 width=48,
                 height=48,
                 num_creeps=3,
                 UNIFORM_SPEED=True,
                 NO_SPEED=False,
                 fps=20):

        actions = {
            "left": K_a,
            "right": K_d,
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)
        self.BG_COLOR = (255, 255, 255)
        self.N_CREEPS = num_creeps
        self.CREEP_TYPES = ["GOOD", "BAD"]
        self.CREEP_COLORS = [(40, 240, 40), (150, 95, 95)]
        radius = percent_round_int(min(width, height), 0.047)
        self.CREEP_RADII = [radius, radius]
        self.CREEP_REWARD = [
            self.rewards["positive"],
            self.rewards["negative"]]
        if NO_SPEED:
            self.CREEP_SPEED = 0
        else:
            self.CREEP_SPEED = width
        self.AGENT_COLOR = (30, 30, 70)
        self.AGENT_SPEED = width
        self.AGENT_RADIUS = radius
        self.AGENT_INIT_POS = None
        self.UNIFORM_SPEED = UNIFORM_SPEED
        self.WALL_COLOR = (255, 255, 255)
        self.creep_counts = {
            "GOOD": 0,
            "BAD": 0
        }

        self.dx = 0
        self.dy = 0
        self.player = None
        self.creeps = None
        self.walls = None
        self.fps = fps
        self.wall_width = self.CREEP_SPEED / fps
        vir_size = self.real2vir(self.width, self.height)
        self.map_shape = [vir_size[0] + 1, vir_size[1] + 1]

    def vir2real(self, x, y):
        return ((x+0.5) * self.wall_width, (y+0.5) * self.wall_width)
    
    def real2vir(self, x, y):
        return (int(x / self.wall_width), int(y / self.wall_width))

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

    def _add_creep(self, creep_type):
        # creep_type = self.rng.choice([0, 1])

        creep = None
        pos = (0, 0)
        dist = 0.0

        while dist < self.AGENT_RADIUS + self.CREEP_RADII[creep_type] + 1:
            radius = self.CREEP_RADII[creep_type] * 1.5
            pos = self.rng.uniform(radius, self.height - radius, size=2)
            dist = math.sqrt(
                (self.player.pos.x - pos[0])**2 + (self.player.pos.y - pos[1])**2)

        creep = Creep(
            self.CREEP_COLORS[creep_type],
            self.CREEP_RADII[creep_type],
            pos,
            self.rng.uniform(-1, 1, size=2),
            self.rng.rand() * self.CREEP_SPEED,
            self.CREEP_REWARD[creep_type],
            self.CREEP_TYPES[creep_type],
            self.width,
            self.height,
            self.rng.rand()
        )

        self.creeps[creep_type].add(creep)

        self.creep_counts[self.CREEP_TYPES[creep_type]] += 1

    def getGameState(self):
        player_vir_pos = self.real2vir(self.player.pos.x, self.player.pos.y)
        player_vir_vel = [self.player.vel.x / self.fps / self.wall_width, self.player.vel.y / self.fps / self.wall_width]
        player_vir_spd = self.AGENT_SPEED / self.fps / self.wall_width
        player_vir_box = [self.player.rect.left / self.wall_width - 0.5,
                          self.player.rect.top / self.wall_width - 0.5,  
                          self.player.rect.right / self.wall_width - 0.5,
                          self.player.rect.bottom / self.wall_width - 0.5, 
                          ]

        player_state = {'type':'player', 
                        'type_index': 0, 
                        'position': [self.player.pos.x, self.player.pos.y],
                        'velocity': [self.player.vel.x / self.fps, self.player.vel.y / self.fps],
                        'speed': self.AGENT_SPEED / self.fps,
                        'box': [self.player.rect.left, self.player.rect.top, self.player.rect.right, self.player.rect.bottom],
                        'norm_position': [player_vir_pos[0], player_vir_pos[1]],
                        'norm_velocity': player_vir_vel,
                        'norm_speed': player_vir_spd,
                        'norm_box': player_vir_box,
                       }


        state = [player_state]
        for c in self.creeps[0]:
            vir_pos = [c.pos.x / self.wall_width - 0.5, c.pos.y / self.wall_width - 0.5]
            vir_vel = [c.direction.x * c.speed / self.fps / self.wall_width, c.direction.y * c.speed / self.fps / self.wall_width]
            vir_spd = c.speed / self.fps / self.wall_width
            vir_box = [c.rect.left / self.wall_width - 0.5,
                       c.rect.top / self.wall_width - 0.5,   
                       c.rect.right / self.wall_width - 0.5,
                       c.rect.bottom / self.wall_width - 0.5,
                       ]
            creep_state = {'type':'creep', 
                           'type_index': self.CREEP_TYPES.index(c.TYPE) + 1, 
                           'position': [c.pos.x, c.pos.y],
                           'velocity': [c.direction.x * c.speed / self.fps, c.direction.y * c.speed / self.fps],
                           'speed': c.speed / self.fps,
                           'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                           'norm_position': vir_pos,
                           'norm_velocity': vir_vel,
                           'norm_speed': vir_spd,
                           'norm_box': vir_box,
                          }
            state.append(creep_state)
        for c in self.creeps[1]:
            vir_pos = [c.pos.x / self.wall_width - 0.5, c.pos.y / self.wall_width - 0.5]
            vir_vel = [c.direction.x * c.speed / self.fps / self.wall_width, c.direction.y * c.speed / self.fps / self.wall_width]
            vir_spd = c.speed / self.fps / self.wall_width
            vir_box = [c.rect.left / self.wall_width - 0.5,
                       c.rect.top / self.wall_width - 0.5,   
                       c.rect.right / self.wall_width - 0.5,
                       c.rect.bottom / self.wall_width - 0.5,
                       ]
            creep_state = {'type':'creep', 
                           'type_index': self.CREEP_TYPES.index(c.TYPE) + 1, 
                           'position': [c.pos.x, c.pos.y],
                           'velocity': [c.direction.x * c.speed / self.fps, c.direction.y * c.speed / self.fps],
                           'speed': c.speed / self.fps,
                           'box': [c.rect.left, c.rect.top, c.rect.right, c.rect.bottom],
                           'norm_position': vir_pos,
                           'norm_velocity': vir_vel,
                           'norm_speed': vir_spd,
                           'norm_box': vir_box,
                          }
            state.append(creep_state)

        return {'local':state, 'global':{'map_shape':self.map_shape}, 'rate_of_progress': (self.ticks * self.AGENT_SPEED) / self.N_CREEPS * (self.width + self.height))}

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return (self.creep_counts['GOOD'] == 0) or self.ticks * self.AGENT_SPEED >= self.N_CREEPS * (self.width + self.height)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        self.creep_counts = {"GOOD": 0, "BAD": 0}
        self.AGENT_INIT_POS = (self.rng.uniform(self.AGENT_RADIUS, self.width - self.AGENT_RADIUS), self.height - self.AGENT_RADIUS)

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
            self.creeps = [pygame.sprite.Group(), pygame.sprite.Group()]
        else:
            self.creeps[0].empty()
            self.creeps[1].empty()

        for i in range(self.N_CREEPS):
            if i < self.N_CREEPS // 2:
                self._add_creep(0)
            else:
                self._add_creep(1)

        if self.walls is None:
            self.walls = Wall((self.width / 2, self.height - 5), self.width, 5, self.WALL_COLOR)

        self.score = 0
        self.ticks = 0
        self.lives = -1
        self.screen.fill(self.BG_COLOR)
        self.player.draw(self.screen)
        self.creeps[0].draw(self.screen)
        self.creeps[1].draw(self.screen)

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()
        self.player.update(self.dx, self.dy, dt)
        self.creeps[0].update(dt)
        self.creeps[1].update(dt)

        hits = pygame.sprite.spritecollide(self.player, self.creeps[0], True)
        for creep in hits:
            self.creep_counts[creep.TYPE] -= 1
            self.score += creep.reward
            # self._add_creep(1)
        
        hits = pygame.sprite.spritecollide(self.player, self.creeps[1], True)
        for creep in hits:
            self.creep_counts[creep.TYPE] -= 1
            self.score += creep.reward
            # self._add_creep(1)

        hits = pygame.sprite.spritecollide(self.walls, self.creeps[0], True)
        for creep in hits:
            self.creep_counts[creep.TYPE] -= 1

        if self.creep_counts["GOOD"] == 0:
            self.score += self.rewards["win"]

        self.player.draw(self.screen)
        self.creeps[0].draw(self.screen)
        self.creeps[1].draw(self.screen)
        self.ticks += dt

if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = WaterWorld1d(width=512, height=512, num_creeps=10)
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