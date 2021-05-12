import pygame
import sys
import math

from ..base import PyGameWrapper, Player, Creep, Bomb, Wall

from ..utils import vec2d, percent_round_int
from pygame.constants import K_w, K_a, K_s, K_d, K_SPACE


class BomberMan(PyGameWrapper):
    """
    Need to place bomb to kill all creeps
    Parameters
    ----------
    width : int
        Screen width.
    height : int
        Screen height, recommended to be same dimension as width.
    num_creeps : int (default: 3)
        The number of creeps on the screen at once.
    NO_SPEED : bool (default: false)
        whether the node can move.
    """

    def __init__(self,
                 width=48,
                 height=48,
                 num_creeps=3,
                 NO_SPEED=False,
                 fps=20):

        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s,
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
        self.AGENT_INIT_POS = None
        self.UNIFORM_SPEED = True
        self.creep_counts = {
            "GOOD": 0
        }
        self.BOMB_COLOR = (70, 30, 30)
        self.BOMB_RADIUS = int(radius * 1.1)
        self.BOMB_LIFE = 4
        self.BOMB_RANGE = 2

        self.EXPLODE_COLOR = (120, 220, 180)
        self.EXPLODE_SHAPE = (2 * self.BOMB_RADIUS, 2 * self.BOMB_RADIUS)
        self.explode_pos = []
        self.dx = 0
        self.dy = 0
        self.shoot = 0
        self.player = None
        self.creeps = None
        self.bombs = None
        self.explosion = None
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

                if key == self.actions["up"]:
                    self.dy -= self.AGENT_SPEED

                if key == self.actions["down"]:
                    self.dy += self.AGENT_SPEED


    def _add_bomb(self):
        pos = (self.player.pos.x, self.player.pos.y)

        bomb = Bomb(
            self.BOMB_COLOR,
            self.BOMB_RADIUS,
            pos,
            self.BOMB_LIFE,
            self.BOMB_RANGE,
            self.width,
            self.height
        )
        hits = pygame.sprite.spritecollide(bomb, self.bombs, False)
        if len(hits) == 0:
            self.bombs.add(bomb)
        else:
            bomb.kill()

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
            self.rng.uniform(-1, 1, size=2),
            self.rng.rand() * self.CREEP_SPEED,
            self.CREEP_REWARD[creep_type],
            self.CREEP_TYPES[creep_type],
            self.width,
            self.height,
            self.rng.rand()
        )

        self.creeps.add(creep)

        self.creep_counts[self.CREEP_TYPES[creep_type]] += 1

    def _cal_explode_pos(self, bomb):
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        for bomb_range in range(self.BOMB_RANGE+1):
            for direction in dirs:
                vir_pos = (bomb.pos.x + direction[0]*bomb_range*self.EXPLODE_SHAPE[0], bomb.pos.y + direction[1]*bomb_range*self.EXPLODE_SHAPE[1])
                if vir_pos[0] < self.EXPLODE_SHAPE[0] / 2 or vir_pos[0] >= self.width - self.EXPLODE_SHAPE[0] / 2 or vir_pos[1] < self.EXPLODE_SHAPE[1] / 2 or vir_pos[1] >= self.height - self.EXPLODE_SHAPE[1] / 2:
                    continue
                explosion = Wall(vir_pos, self.EXPLODE_SHAPE[0], self.EXPLODE_SHAPE[1], self.EXPLODE_COLOR)
                self.explosion.add(explosion)

    def explode(self):
        self.explosion.empty()
        for bomb in self.bombs:
            if bomb.life < 1 / self.fps:
                self._cal_explode_pos(bomb)
                bomb.kill()

        hits = pygame.sprite.groupcollide(self.bombs, self.explosion, True, False)
        while len(hits) > 0:
            for bomb in hits.keys():
                self._cal_explode_pos(bomb)
            hits = pygame.sprite.groupcollide(self.bombs, self.explosion, True, False)

    def getGameState(self):
        state = []
        if self.player is not None:
            player_vir_pos = self.real2vir(self.player.pos.x, self.player.pos.y)
            player_vir_vel = [self.player.vel.x / self.fps / self.wall_width, self.player.vel.y / self.fps / self.wall_width]
            player_vir_spd = self.AGENT_SPEED / self.fps / self.wall_width
            player_vir_box = [self.player.rect.left / self.wall_width - 0.5,
                              self.player.rect.top / self.wall_width - 0.5,  
                              self.player.rect.right / self.wall_width - 0.5,
                              self.player.rect.bottom / self.wall_width - 0.5, 
                             ]
            player_state = {'type':'player', 
                            'type_index': [0, -1], 
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
        for c in self.creeps:
            vir_pos = [c.pos.x / self.wall_width - 0.5, c.pos.y / self.wall_width - 0.5]
            vir_vel = [c.direction.x * c.speed / self.fps / self.wall_width, c.direction.y * c.speed / self.fps / self.wall_width]
            vir_spd = c.speed / self.fps / self.wall_width
            vir_box = [c.rect.left / self.wall_width - 0.5,
                       c.rect.top / self.wall_width - 0.5,   
                       c.rect.right / self.wall_width - 0.5,
                       c.rect.bottom / self.wall_width - 0.5,
                       ]
            creep_state = {'type':'creep', 
                           'type_index': [1, -1], 
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

        for b in self.bombs:
            vir_pos = self.real2vir(b.pos.x, b.pos.y)
            vir_box = [b.rect.left / self.wall_width - 0.5,
                       b.rect.top / self.wall_width - 0.5,   
                       b.rect.right / self.wall_width - 0.5,
                       b.rect.bottom / self.wall_width - 0.5,
                       ]
            bomb_state = {'type':'bomb', 
                          'type_index': [2, b.life], 
                          'position': [b.pos.x, b.pos.y],
                          'velocity': [0, 0],
                          'speed': 0,
                          'box': [b.rect.left, b.rect.top, b.rect.right, b.rect.bottom],
                          'norm_position': [vir_pos[0], vir_pos[1]],
                          'norm_velocity': [0, 0],
                          'norm_speed': 0,
                          'norm_box': vir_box,
                         }
            state.append(bomb_state)
        global_state = {'map_shape': self.map_shape, 
                        'bomb_range': [self.EXPLODE_SHAPE[0]*self.BOMB_RANGE, self.EXPLODE_SHAPE[1]*self.BOMB_RANGE], 
                        'norm_bomb_range': [self.EXPLODE_SHAPE[0]*self.BOMB_RANGE//self.wall_width, self.EXPLODE_SHAPE[1]*self.BOMB_RANGE//self.wall_width]}
        return {'local':state, 'global':global_state}

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return (self.creep_counts['GOOD'] == 0) or self.player == None # or self.ticks > 2 * self.N_CREEPS * (self.width + self.height)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        self.creep_counts = {"GOOD": 0}
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

        if self.explosion is None:
            self.explosion = pygame.sprite.Group()
        else:
            self.explosion.empty()

        if self.bombs is None:
            self.bombs = pygame.sprite.Group()
        else:
            self.bombs.empty()

        for i in range(self.N_CREEPS):

            self._add_creep(0, self.CREEP_RADII)

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
        if self.player is None:
            return
        dt = 1.0 / self.fps
        self.screen.fill(self.BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()
        
        if self.shoot > 0:
            self._add_bomb()
            self.player.vel.x = 0
            self.player.vel.y = 0
        else:
            hits_before = pygame.sprite.spritecollide(self.player, self.bombs, False)
            assert len(hits_before) <= 1
            self.player.update(self.dx, self.dy, dt)
            hits_after = pygame.sprite.spritecollide(self.player, self.bombs, False)

            if len(hits_after) > 0 and hits_before != hits_after:
                if self.dx == 0 and self.dy == 0:
                    self.player.update(-self.player.vel.x, -self.player.vel.y, dt)
                else:
                    self.player.update(-self.dx, -self.dy, dt)
                
                self.player.vel.x = 0
                self.player.vel.y = 0
            
        self.creeps.update(dt)
        hits = pygame.sprite.groupcollide(self.creeps, self.bombs, False, False)
        for creep in hits.keys():
            creep.direction.x, creep.direction.y = -creep.direction.x, -creep.direction.y
            creep.update(2*dt)
        
        hits = pygame.sprite.groupcollide(self.creeps, self.bombs, False, False)
        for creep in hits.keys():
            creep.direction.x, creep.direction.y = -creep.direction.x, -creep.direction.y
            creep.update(dt)
            creep.direction.x, creep.direction.y = creep.direction.x, -creep.direction.y
            creep.update(dt)
        
        hits = pygame.sprite.groupcollide(self.creeps, self.bombs, False, False)
        for creep in hits.keys():
            creep.direction.x, creep.direction.y = -creep.direction.x, -creep.direction.y
            creep.update(2*dt)

        hits = pygame.sprite.groupcollide(self.creeps, self.bombs, False, False)
        for creep in hits.keys():
            creep.direction.x, creep.direction.y = -creep.direction.x, -creep.direction.y
            creep.update(dt)

        hits = pygame.sprite.spritecollide(self.player, self.creeps, False)

        self.explode()
        self.bombs.update(dt)
        self.player.draw(self.screen)
        self.bombs.draw(self.screen)
        self.explosion.draw(self.screen)
        self.creeps.draw(self.screen)

        if len(hits) != 0:
            self.player.kill()
            self.player = None
            self.score -= 1
            return

        hits = pygame.sprite.spritecollide(self.player, self.explosion, False)
        if len(hits) != 0:
            self.player.kill()
            self.player = None
            self.score -= 1

        hits = pygame.sprite.groupcollide(self.creeps, self.explosion, True, False)
        for creep in hits.keys():
            self.creep_counts[creep.TYPE] -= 1
            self.score += creep.reward
            # self._add_creep(1)

        if self.creep_counts["GOOD"] == 0:
            self.score += self.rewards["win"]

        self.ticks += self.AGENT_SPEED * dt

if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = BomberMan(width=512, height=512, num_creeps=1, UNIFORM_SPEED=True)
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