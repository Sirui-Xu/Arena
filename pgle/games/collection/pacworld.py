import pygame
import sys
sys.path.append("..") 
import math

from base import PyGameWrapper, Player, Creep

from utils import vec2d, percent_round_int
from pygame.constants import K_w, K_a, K_s, K_d


class PacWorld(PyGameWrapper):
    """
    The lighter the color, the higher the score
    Parameters
    ----------
    width : int
        Screen width.
    height : int
        Screen height, recommended to be same dimension as width.
    num_creeps : int (default: 3)
        The number of creeps on the screen at once.
    UNIFORM_SPEED : bool (default: false)
        The agent has an uniform speed or not.
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
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s
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
            self.CREEP_SPEED = 0.5 * width
        self.AGENT_COLOR = (30, 30, 30)
        self.AGENT_SPEED = width
        self.AGENT_RADIUS = radius
        self.AGENT_INIT_POS = None
        self.UNIFORM_SPEED = UNIFORM_SPEED
        self.creep_counts = {
            "GOOD": 0,
            "BAD": 0
        }
        self.dx = 0
        self.dy = 0
        self.player = None
        self.creeps = None
        self.assigned_values = None

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

    def _add_creep(self, creep_type, idx, color, sum_color):
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
            (40 + 200*color, 40 + 200*color, 40 + 200*color),
            self.CREEP_RADII[creep_type],
            pos,
            self.rng.choice([-1, 1], 2),
            self.rng.rand() * self.CREEP_SPEED,
            self.CREEP_REWARD[creep_type] * color / sum_color * self.N_CREEPS,
            self.CREEP_TYPES[creep_type],
            self.width,
            self.height,
            self.rng.rand(),
            idx
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
        order = list(range(len(self.creeps.sprites())))
        # self.rng.shuffle(order)
        for i in order:
            c = self.creeps.sprites()[i]
            creep_state = {'type':'creep', 
                           'type_index': c.idx + 1,  
                           'position': [c.pos.x, c.pos.y],
                           'velocity': [c.direction.x * c.speed, c.direction.y * c.speed],
                           'speed': c.speed,
                           'box': [c.rect.top, c.rect.left, c.rect.bottom, c.rect.right]
                          }
            state.append(creep_state)

        return state, None

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return (self.creep_counts['GOOD'] + self.creep_counts['BAD'] == 0) or self.ticks * self.AGENT_SPEED >= self.width + self.height

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        self.assigned_values = self.rng.rand((self.N_CREEPS))
        self.assigned_values.sort()
        self.creep_counts = {"GOOD": 0, "BAD": 0}
        self.AGENT_INIT_POS = self.rng.uniform(self.AGENT_RADIUS, self.height - self.AGENT_RADIUS, size=2)

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

        sum_assigned_values = sum(self.assigned_values)
        for i in range(self.N_CREEPS):
            self._add_creep(0, i, self.assigned_values[i], sum_assigned_values)

        self.score = 0
        self.ticks = 0
        self.lives = -1

        self.screen.fill(self.BG_COLOR)
        self.player.draw(self.screen)
        self.creeps.draw(self.screen)


    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()
        self.player.update(self.dx, self.dy, dt)

        hits = pygame.sprite.spritecollide(self.player, self.creeps, True)
        for creep in hits:
            self.creep_counts[creep.TYPE] -= 1
            self.score += creep.reward

        self.creeps.update(dt)

        self.player.draw(self.screen)
        self.creeps.draw(self.screen)
        self.ticks += dt
        # print(self.creep_counts["GOOD"], self.creep_counts["BAD"])

if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = PacWorld(width=512, height=512, num_creeps=10)
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