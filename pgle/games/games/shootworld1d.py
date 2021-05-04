import pygame
import sys
sys.path.append("..")
import math

from base import PyGameWrapper, Player, Creep, Bullet

from utils import vec2d, percent_round_int
from pygame.constants import K_w, K_a, K_s, K_d, K_SPACE


class ShootWorld1d(PyGameWrapper):
    """
    Move in a line. Shot minus 1 point, hit plus 2 points.
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
                 UNIFORM_SPEED=False,
                 NO_SPEED=False):

        actions = {
            "left": K_a,
            "right": K_d,
            "shoot": K_SPACE,
        }
        PyGameWrapper.__init__(self, width, height, actions=actions)
        self.BG_COLOR = (255, 255, 255)
        self.N_CREEPS = num_creeps
        self.CREEP_TYPES = ["GOOD"]
        self.CREEP_COLORS = [(40, 240, 40)]
        radius = percent_round_int(min(width, height), 0.047)
        self.CREEP_RADII = radius
        self.CREEP_REWARD = [
            self.rewards["positive"] * 2]
        if NO_SPEED:
            self.CREEP_SPEED = 0
        else:
            self.CREEP_SPEED = width
        self.AGENT_COLOR = (30, 30, 30)
        self.AGENT_SPEED = width
        self.AGENT_RADIUS = radius
        self.AGENT_INIT_POS = (self.width // 2, self.height)
        self.UNIFORM_SPEED = UNIFORM_SPEED
        self.creep_counts = {
            "GOOD": 0
        }
        self.BULLET_TYPE = "BULLET"
        self.BULLET_COLOR = (60, 30, 90)
        self.BULLET_SPEED = 1.5*width
        self.BULLET_RADIUS = percent_round_int(min(width, height), 0.020)
        self.dx = 0
        self.dy = 0
        self.shoot = 0
        self.player = None
        self.creeps = None
        self.bullets = None

    def _handle_player_events(self):
        self.dx = 0
        self.dy = 0
        self.shoot = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["shoot"]:
                    self.shoot += 1

                if key == self.actions["left"]:
                    self.dx -= self.AGENT_SPEED

                if key == self.actions["right"]:
                    self.dx += self.AGENT_SPEED


    def _add_bullets(self):
        bullet = Bullet(self.BULLET_COLOR, 
                        self.BULLET_RADIUS, 
                        (self.player.pos.x, self.player.pos.y), 
                        (0, -1), 
                        self.BULLET_SPEED, 
                        self.BULLET_TYPE,
                        self.width,
                        self.height)
        self.bullets.add(bullet)


    def _add_creep(self, creep_type, radius):
        # creep_type = self.rng.choice([0, 1])

        creep = None
        pos = (0, 0)
        dist = 0.0

        while dist < self.AGENT_RADIUS + self.CREEP_RADII + 1:
            d = self.CREEP_RADII * 1.5
            pos = self.rng.uniform(d, self.height - d, size=2)
            dist = math.sqrt(
                (self.player.pos.x - pos[0])**2 + (self.player.pos.y - pos[1])**2)

        creep = Creep(
            self.CREEP_COLORS[creep_type],
            radius,
            pos,
            self.rng.choice([-1, 1], 2),
            self.rng.rand() * self.CREEP_SPEED,
            self.CREEP_REWARD[creep_type],
            self.CREEP_TYPES[creep_type],
            self.width,
            self.height,
            self.rng.rand()
        )

        self.creeps.add(creep)

        self.creep_counts[self.CREEP_TYPES[creep_type]] += 1

    def getGameState(self):
        player_state = {'type':'player', 
                        'type_index': 0, 
                        'position': [self.player.pos.x, self.player.pos.y],
                        'velocity': [self.player.vel.x, self.player.vel.y],
                        'speed': self.AGENT_SPEED,
                        'box': [self.player.rect.top, self.player.rect.left, self.player.rect.bottom, self.player.rect.right]
                       }

        state = [player_state]
        for c in self.creeps:
            creep_state = {'type':'creep', 
                        'type_index': 1, 
                        'position': [c.pos.x, c.pos.y],
                        'velocity': [c.direction.x * c.speed, c.direction.y * c.speed],
                        'speed': c.speed,
                        'box': [c.rect.top, c.rect.left, c.rect.bottom, c.rect.right]
                        }
            state.append(creep_state)

        for b in self.bullets:
            bullet_state = {'type':'bullet', 
                        'type_index': 2, 
                        'position': [b.pos.x, b.pos.y],
                        'velocity': [b.direction.x * b.speed, b.direction.y * b.speed],
                        'speed': b.speed,
                        'box': [b.rect.top, b.rect.left, b.rect.bottom, b.rect.right]
                        }
            state.append(bullet_state)
        return state, None

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return (self.creep_counts['GOOD'] == 0) or self.ticks > self.N_CREEPS * (self.width + self.height)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        self.creep_counts = {"GOOD": 0}
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
            self.creeps = pygame.sprite.Group()
        else:
            self.creeps.empty()

        if self.bullets is None:
            self.bullets = pygame.sprite.Group()
        else:
            self.bullets.empty()

        for i in range(self.N_CREEPS):
            self._add_creep(0, self.CREEP_RADII)

        self.score = 0
        self.ticks = 0
        self.lives = -1
        self.screen.fill(self.BG_COLOR)
        self.player.draw(self.screen)
        self.creeps.draw(self.screen)
        self.bullets.draw(self.screen)


    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()
        self.player.update(self.dx, 0, dt)
        if self.shoot > 0:
            self._add_bullets()
            self.score -= 1
        hits = pygame.sprite.groupcollide(self.bullets, self.creeps, True, True)
        for bullet in hits.keys():
            for creep in hits[bullet]:
                self.creep_counts[creep.TYPE] -= 1
                self.score += creep.reward
                # self._add_creep(1)

        if self.creep_counts["GOOD"] == 0:
            self.score += self.rewards["win"]

        self.creeps.update(dt)
        self.bullets.update(dt)

        self.player.draw(self.screen)
        self.creeps.draw(self.screen)
        self.bullets.draw(self.screen)
        self.ticks += self.AGENT_SPEED * dt

if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = ShootWorld1d(width=512, height=512, num_creeps=10)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
        # print(game.getGameState())
        if game.game_over() is True:
            print("The overall score is {}.".format(game.score))
            break