import pygame
import math
from ..utils import vec2d
import os
import random

class Bombv(pygame.sprite.Sprite):

    def __init__(self,
                 radius,
                 pos_init,
                 life,
                 explode_range):

        pygame.sprite.Sprite.__init__(self)
        self.TYPE = "Bomb"
        self.radius = radius
        self.pos = vec2d(pos_init)
        self.life = life
        self.explode_range = explode_range

        self.animation = []
        self.load_animations(radius * 2)
        self.image = self.animation[0]
        self.index = 0
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dt):

        self.life -= dt
        if self.life <= -1:
            self.kill()
        self.image = self.animation[int(self.index)]
        self.index += dt
        if self.index >= len(self.animation):
            self.index = self.index % len(self.animation)

    def load_animations(self, scale):
        resize_width = scale
        resize_height = scale
        _dir_ = os.path.dirname(os.path.abspath(__file__))

        b1 = pygame.image.load(os.path.join(_dir_, "images/bomb/1.png"))
        b2 = pygame.image.load(os.path.join(_dir_, "images/bomb/2.png"))
        b3 = pygame.image.load(os.path.join(_dir_, "images/bomb/3.png"))

        b1 = pygame.transform.scale(b1, (resize_width, resize_height))
        b2 = pygame.transform.scale(b2, (resize_width, resize_height))
        b3 = pygame.transform.scale(b3, (resize_width, resize_height))

        self.animation.append(b1)
        self.animation.append(b2)
        self.animation.append(b3)


class Projectile(pygame.sprite.Sprite):

    def __init__(self,
                 radius,
                 pos_init,
                 dir_init,
                 speed,
                 SCREEN_WIDTH,
                 SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)
        self.TYPE = "Projectile"
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.speed = speed
        self.radius = radius
        self.pos = vec2d(pos_init)
        self.direction = vec2d(dir_init)
        self.direction.normalize()
        self.image = self.load_animations(2 * self.radius, self.direction)
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dt):

        dx = self.direction.x * self.speed * dt
        dy = self.direction.y * self.speed * dt

        if self.pos.x + dx > self.SCREEN_WIDTH:
            self.kill()
        elif self.pos.x + dx <= 0:
            self.kill()
        else:
            self.pos.x = self.pos.x + dx

        if self.pos.y + dy > self.SCREEN_HEIGHT:
            self.kill()
        elif self.pos.y + dy <= 0:
            self.kill()
        else:
            self.pos.y = self.pos.y + dy

        self.direction.normalize()

        self.rect.center = ((self.pos.x, self.pos.y))

    def load_animations(self, scale, direction):
        resize_width = scale
        resize_height = scale
        _dir_ = os.path.dirname(os.path.abspath(__file__))

        if direction.x > 0 and direction.y == 0:
            b = pygame.image.load(os.path.join(_dir_, "images/explosion/tile083.png"))
        elif direction.x < 0 and direction.y == 0:
            b = pygame.image.load(os.path.join(_dir_, "images/explosion/tile080.png"))
        elif direction.y > 0 and direction.x == 0:
            b = pygame.image.load(os.path.join(_dir_, "images/explosion/tile047.png"))
        elif direction.y < 0 and direction.x == 0:
            b = pygame.image.load(os.path.join(_dir_, "images/explosion/tile015.png"))
        else:
            raise Exception("invalid direction for bullet")
        b = pygame.transform.scale(b, (resize_width, resize_height))

        return b


class Enemy(pygame.sprite.Sprite):

    def __init__(self,
                 radius,
                 pos_init,
                 dir_init,
                 speed,
                 SCREEN_WIDTH,
                 SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)
        self.TYPE = "Enemy"
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.speed = speed
        self.INIT_SPEED = speed
        self.radius = radius
        self.pos = vec2d(pos_init)
        self.direction = vec2d(dir_init)
        self.direction.normalize()  # normalized

        self.animation = []
        self.load_animations(radius * 2)
        self.image = self.animation[0][0]
        self.index = [0, 0, 0, 0]
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update_image(self, dt):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        index = directions.index((self.direction.x, self.direction.y))
        self.image = self.animation[index][int(self.index[index])]
        self.index[index] += dt
        if self.index[index] >= len(self.animation[index]):
            self.index[index] = self.index[index] % len(self.animation[index])

    def valid(self, walls):
        pos = self.rect.center
        if pos[0] > self.SCREEN_WIDTH - self.radius or pos[0] < self.radius:
            return False
        if pos[1] > self.SCREEN_HEIGHT - self.radius or pos[1] < self.radius:
            return False
        hits = pygame.sprite.spritecollide(self, walls, False)
        if len(hits) > 0:
            return False
        return True 

    def update(self, dt, walls):
        self.speed = self.INIT_SPEED
        dx = self.direction.x * self.speed * dt
        dy = self.direction.y * self.speed * dt

        self.rect.center = (self.pos.x + dx, self.pos.y + dy)
        if not (self.valid(walls) and random.random() < 0.99):
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
            random.shuffle(directions)
            for direction in directions:
                self.rect.center = (self.pos.x + direction[0] * self.speed * dt, self.pos.y + direction[1] * self.speed * dt)
                if self.valid(walls):
                    if direction == (0, 0):
                        self.speed = 0
                    else:
                        self.direction.x = direction[0]
                        self.direction.y = direction[1]
                    break
        self.pos.x, self.pos.y = self.rect.center
        self.update_image(dt)

    def load_animations(self, scale):
        t = 1
        front = []
        back = []
        left = []
        right = []
        resize_width = scale
        resize_height = scale
        _dir_ = os.path.dirname(os.path.abspath(__file__))
        f1 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}f0.png".format(t)))
        f2 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}f1.png".format(t)))
        f3 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}f2.png".format(t)))

        f1 = pygame.transform.scale(f1, (resize_width, resize_height))
        f2 = pygame.transform.scale(f2, (resize_width, resize_height))
        f3 = pygame.transform.scale(f3, (resize_width, resize_height))

        front.append(f1)
        front.append(f2)
        front.append(f3)

        r1 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}r0.png".format(t)))
        r2 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}r1.png".format(t)))
        r3 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}r2.png".format(t)))

        r1 = pygame.transform.scale(r1, (resize_width, resize_height))
        r2 = pygame.transform.scale(r2, (resize_width, resize_height))
        r3 = pygame.transform.scale(r3, (resize_width, resize_height))

        right.append(r1)
        right.append(r2)
        right.append(r3)

        b1 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}b0.png".format(t)))
        b2 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}b1.png".format(t)))
        b3 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}b2.png".format(t)))

        b1 = pygame.transform.scale(b1, (resize_width, resize_height))
        b2 = pygame.transform.scale(b2, (resize_width, resize_height))
        b3 = pygame.transform.scale(b3, (resize_width, resize_height))

        back.append(b1)
        back.append(b2)
        back.append(b3)

        l1 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}l0.png".format(t)))
        l2 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}l1.png".format(t)))
        l3 = pygame.image.load(os.path.join(_dir_, "images/enemy/e{}l2.png".format(t)))

        l1 = pygame.transform.scale(l1, (resize_width, resize_height))
        l2 = pygame.transform.scale(l2, (resize_width, resize_height))
        l3 = pygame.transform.scale(l3, (resize_width, resize_height))

        left.append(l1)
        left.append(l2)
        left.append(l3)

        self.animation.append(front)
        self.animation.append(right)
        self.animation.append(back)
        self.animation.append(left)

class Reward(pygame.sprite.Sprite):

    def __init__(self, pos, radius, reward):
        pygame.sprite.Sprite.__init__(self)
        self.TYPE = "Reward"
        self.pos = vec2d(pos)
        self.radius = radius
        self.reward = reward
        _dir_ = os.path.dirname(os.path.abspath(__file__))
        self.image = pygame.image.load(os.path.join(_dir_, "images/gold.png"))
        self.image = pygame.transform.scale(self.image, (2 * radius, 2 * radius))
        self.rect = self.image.get_rect()
        self.rect.center = pos


class Obstacle(pygame.sprite.Sprite):

    def __init__(self, pos, radius, FIXED=False):
        pygame.sprite.Sprite.__init__(self)
        self.TYPE = "Obstacle"
        self.pos = vec2d(pos)
        self.radius = radius
        _dir_ = os.path.dirname(os.path.abspath(__file__))
        if FIXED:
            self.image = pygame.image.load(os.path.join(_dir_, "images/terrain/block.png"))
        else:
            self.image = pygame.image.load(os.path.join(_dir_, "images/terrain/box.png"))
        self.image = pygame.transform.scale(self.image, (radius * 2, radius * 2))
        self.rect = self.image.get_rect()
        self.rect.center = pos

class Blast(pygame.sprite.Sprite):

    def __init__(self, pos, radius):
        pygame.sprite.Sprite.__init__(self)
        self.TYPE = "Blast"
        self.pos = vec2d(pos)
        self.radius = radius

        self.animation = []
        self.load_animations(2 * radius, 2 * radius)

        self.image = self.animation[0]
        self.index = 0
        self.life = len(self.animation)
        self.rect = self.image.get_rect()
        self.rect.center = pos

    def update(self, dt):
        self.image = self.animation[int(self.index)]
        self.index += dt
        self.life -= dt
        if self.index >= len(self.animation):
            self.kill()

    def load_animations(self, w, h):
        resize_width = w
        resize_height = h
        _dir_ = os.path.dirname(os.path.abspath(__file__))

        b1 = pygame.image.load(os.path.join(_dir_, "images/explosion/1.png"))
        b2 = pygame.image.load(os.path.join(_dir_, "images/explosion/2.png"))
        b3 = pygame.image.load(os.path.join(_dir_, "images/explosion/3.png"))

        b1 = pygame.transform.scale(b1, (resize_width, resize_height))
        b2 = pygame.transform.scale(b2, (resize_width, resize_height))
        b3 = pygame.transform.scale(b3, (resize_width, resize_height))

        self.animation.append(b1)
        self.animation.append(b2)
        self.animation.append(b3)

class Agent(pygame.sprite.Sprite):

    def __init__(self,
                 radius,
                 speed,
                 pos_init,
                 SCREEN_WIDTH,
                 SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)
        self.TYPE = "Agent"
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.pos = vec2d(pos_init)
        self.direction = vec2d((0, 1))
        self.vel = vec2d((0, speed))
        self.animation = []
        self.load_animations(radius * 2)
        self.direction.normalize()

        self.speed = speed
        self.image = self.animation[0][0]
        self.index = [0, 0, 0, 0]
        self.rect = self.image.get_rect()
        self.rect.center = pos_init
        self.radius = radius


    def update_image(self, dt):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        index = directions.index((self.direction.x, self.direction.y))
        self.image = self.animation[index][int(self.index[index])]
        self.index[index] += dt
        if self.index[index] >= len(self.animation[index]):
            self.index[index] = self.index[index] % len(self.animation[index])

    def valid(self, walls):
        pos = self.rect.center
        if pos[0] > self.SCREEN_WIDTH - self.radius or pos[0] < self.radius:
            return False
        if pos[1] > self.SCREEN_HEIGHT - self.radius or pos[1] < self.radius:
            return False
        hits = pygame.sprite.spritecollide(self, walls, False)
        if len(hits) > 0:
            return False
        return True 

    def update(self, dx, dy, dt, walls):
        if dx == 0 and dy == 0:
            pass
        else:
            self.rect.center = (self.pos.x + dx * dt, self.pos.y + dy * dt)
            if self.valid(walls):
                self.pos.x += dx * dt
                self.pos.y += dy * dt
                self.direction.x = dx
                self.direction.y = dy
                self.direction.normalize()
                self.vel.x = dx
                self.vel.y = dy
                self.update_image(dt)
            else:
                self.direction.x = dx
                self.direction.y = dy
                self.vel.x = 0
                self.vel.y = 0
                self.direction.normalize()
                self.update_image(dt)
                self.rect.center = (self.pos.x, self.pos.y)


    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def load_animations(self, scale):
        front = []
        back = []
        left = []
        right = []
        resize_width = scale
        resize_height = scale
        _dir_ = os.path.dirname(os.path.abspath(__file__))
        f1 = pygame.image.load(os.path.join(_dir_, "images/hero/pf0.png"))
        f2 = pygame.image.load(os.path.join(_dir_, "images/hero/pf1.png"))
        f3 = pygame.image.load(os.path.join(_dir_, "images/hero/pf2.png"))

        f1 = pygame.transform.scale(f1, (resize_width, resize_height))
        f2 = pygame.transform.scale(f2, (resize_width, resize_height))
        f3 = pygame.transform.scale(f3, (resize_width, resize_height))

        front.append(f1)
        front.append(f2)
        front.append(f3)

        r1 = pygame.image.load(os.path.join(_dir_, "images/hero/pr0.png"))
        r2 = pygame.image.load(os.path.join(_dir_, "images/hero/pr1.png"))
        r3 = pygame.image.load(os.path.join(_dir_, "images/hero/pr2.png"))

        r1 = pygame.transform.scale(r1, (resize_width, resize_height))
        r2 = pygame.transform.scale(r2, (resize_width, resize_height))
        r3 = pygame.transform.scale(r3, (resize_width, resize_height))

        right.append(r1)
        right.append(r2)
        right.append(r3)

        b1 = pygame.image.load(os.path.join(_dir_, "images/hero/pb0.png"))
        b2 = pygame.image.load(os.path.join(_dir_, "images/hero/pb1.png"))
        b3 = pygame.image.load(os.path.join(_dir_, "images/hero/pb2.png"))

        b1 = pygame.transform.scale(b1, (resize_width, resize_height))
        b2 = pygame.transform.scale(b2, (resize_width, resize_height))
        b3 = pygame.transform.scale(b3, (resize_width, resize_height))

        back.append(b1)
        back.append(b2)
        back.append(b3)

        l1 = pygame.image.load(os.path.join(_dir_, "images/hero/pl0.png"))
        l2 = pygame.image.load(os.path.join(_dir_, "images/hero/pl1.png"))
        l3 = pygame.image.load(os.path.join(_dir_, "images/hero/pl2.png"))

        l1 = pygame.transform.scale(l1, (resize_width, resize_height))
        l2 = pygame.transform.scale(l2, (resize_width, resize_height))
        l3 = pygame.transform.scale(l3, (resize_width, resize_height))

        left.append(l1)
        left.append(l2)
        left.append(l3)

        self.animation.append(front)
        self.animation.append(right)
        self.animation.append(back)
        self.animation.append(left)
