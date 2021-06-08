import pygame
import math
from ..utils import vec2d
import os
class Bomb(pygame.sprite.Sprite):

    def __init__(self,
                 color,
                 radius,
                 pos_init,
                 life,
                 explode_range,
                 SCREEN_WIDTH,
                 SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.radius = radius
        self.pos = vec2d(pos_init)
        self.life = life
        self.explode_range = explode_range

        self.animation = []
        self.load_animations(radius * 2)

        # image = pygame.Surface((radius * 2, radius * 2))
        # image.fill((0, 0, 0))
        # image.set_colorkey((0, 0, 0))

        # pygame.draw.circle(
        #     image,
        #     color,
        #     (radius, radius),
        #     radius,
        #     0
        # )

        self.image = self.animation[0]
        self.index = 0
        # self.image = image.convert()
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dt):

        self.life -= dt
        if self.life <= -1:
            self.kill()
        self.image = self.animation[int(self.index)]
        self.index += 3 * dt
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


class Bullet(pygame.sprite.Sprite):

    def __init__(self,
                 color,
                 radius,
                 pos_init,
                 dir_init,
                 speed,
                 TYPE,
                 SCREEN_WIDTH,
                 SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.TYPE = TYPE
        self.speed = speed
        self.radius = radius
        self.pos = vec2d(pos_init)
        self.direction = vec2d(dir_init)
        self.direction.normalize()  # normalized

        # image = pygame.Surface((radius * 2, radius * 2))
        # image.fill((0, 0, 0))
        # image.set_colorkey((0, 0, 0))

        # pygame.draw.circle(
        #     image,
        #     color,
        #     (radius, radius),
        #     radius,
        #     0
        # )
        self.image = self.load_animations(2 * self.radius, self.direction)
        # self.image = image.convert()
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dt):

        dx = self.direction.x * self.speed * dt
        dy = self.direction.y * self.speed * dt

        if self.pos.x + dx > self.SCREEN_WIDTH - self.radius:
            self.pos.x = self.SCREEN_WIDTH - self.radius
            self.kill()
        elif self.pos.x + dx <= self.radius:
            self.pos.x = self.radius
            self.kill()
        else:
            self.pos.x = self.pos.x + dx

        if self.pos.y + dy > self.SCREEN_HEIGHT - self.radius:
            self.pos.y = self.SCREEN_HEIGHT - self.radius
            self.kill()
        elif self.pos.y + dy <= self.radius:
            self.pos.y = self.radius
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


class Creep(pygame.sprite.Sprite):

    def __init__(self,
                 color,
                 radius,
                 pos_init,
                 dir_init,
                 speed,
                 reward,
                 TYPE,
                 SCREEN_WIDTH,
                 SCREEN_HEIGHT,
                 jitter_speed,
                 idx=0):

        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.TYPE = TYPE
        self.jitter_speed = jitter_speed
        self.speed = speed
        self.reward = reward
        self.radius = radius
        self.pos = vec2d(pos_init)
        self.idx = idx
        self.direction = vec2d(dir_init)
        self.direction.normalize()  # normalized

        if self.idx != 0:
            image = pygame.Surface((radius * 2, radius * 2))
            image.fill((0, 0, 0))
            image.set_colorkey((0, 0, 0))

            pygame.draw.circle(
                image,
                color,
                (radius, radius),
                radius,
                0
            )
            self.image = image.convert()
        else:
            self.animation = []
            self.load_animations(radius * 2, TYPE)
            self.image = self.animation[0][0]
            self.index = [0, 0, 0, 0]
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update_image(self, dx, dy, dt):
        if self.idx != 0:
            return
        if dx != 0 or dy != 0:
            if abs(dx) >= abs(dy):
                dy = 0
                dx /= abs(dx)
            else:
                dx = 0
                dy /= abs(dy)
        else:
            return
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        index = directions.index((dx, dy))
        self.image = self.animation[index][int(self.index[index])]
        self.index[index] += 5 * dt
        if self.index[index] >= len(self.animation[index]):
            self.index[index] = self.index[index] % len(self.animation[index])

    def update(self, dt):

        dx = self.direction.x * self.speed * dt
        dy = self.direction.y * self.speed * dt

        if self.pos.x + dx > self.SCREEN_WIDTH - self.radius:
            self.pos.x = self.SCREEN_WIDTH - self.radius
            self.direction.x = -1 * self.direction.x * \
                (1 + 0.5 * self.jitter_speed)  # a little jitter
        elif self.pos.x + dx <= self.radius:
            self.pos.x = self.radius
            self.direction.x = -1 * self.direction.x * \
                (1 + 0.5 * self.jitter_speed)  # a little jitter
        else:
            self.pos.x = self.pos.x + dx

        if self.pos.y + dy > self.SCREEN_HEIGHT - self.radius:
            self.pos.y = self.SCREEN_HEIGHT - self.radius
            self.direction.y = -1 * self.direction.y * \
                (1 + 0.5 * self.jitter_speed)  # a little jitter
        elif self.pos.y + dy <= self.radius:
            self.pos.y = self.radius
            self.direction.y = -1 * self.direction.y * \
                (1 + 0.5 * self.jitter_speed)  # a little jitter
        else:
            self.pos.y = self.pos.y + dy

        self.direction.normalize()  # normalized
        self.update_image(dx, dy, dt)
        self.rect.center = ((self.pos.x, self.pos.y))

    def load_animations(self, scale, TYPE):
        if TYPE == "GOOD":
            t = 3
        else:
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

    def __init__(self, pos, shape, reward):
        pygame.sprite.Sprite.__init__(self)

        self.pos = vec2d(pos)
        self.shape = int(shape)
        self.reward = reward
        _dir_ = os.path.dirname(os.path.abspath(__file__))
        self.image = pygame.image.load(os.path.join(_dir_, "images/gold.png"))
        self.image = pygame.transform.scale(self.image, (self.shape, self.shape))
        self.rect = self.image.get_rect()
        self.rect.center = pos


class Wall(pygame.sprite.Sprite):

    def __init__(self, pos, w, h, color, FIXED=True, GRASS=False):
        pygame.sprite.Sprite.__init__(self)

        self.pos = vec2d(pos)
        self.w = w
        self.h = h
        self.color = color
        _dir_ = os.path.dirname(os.path.abspath(__file__))
        if GRASS:
            self.image = pygame.image.load(os.path.join(_dir_, "images/terrain/grass.png"))
        else:
            if FIXED:
                self.image = pygame.image.load(os.path.join(_dir_, "images/terrain/block.png"))
            else:
                self.image = pygame.image.load(os.path.join(_dir_, "images/terrain/box.png"))
        self.image = pygame.transform.scale(self.image, (w, h))
        self.rect = self.image.get_rect()
        self.rect.center = pos

    def draw(self, screen):
        pygame.draw.rect(
            screen, self.color, [
                self.pos.x, self.pos.y, self.w, self.h], 0)

class Explosion(pygame.sprite.Sprite):

    def __init__(self, pos, w, h, color, FIXED=True):

        pygame.sprite.Sprite.__init__(self)
        self.pos = vec2d(pos)
        self.w = w
        self.h = h
        self.color = color

        self.animation = []
        self.load_animations(w, h)

        # image = pygame.Surface((radius * 2, radius * 2))
        # image.fill((0, 0, 0))
        # image.set_colorkey((0, 0, 0))

        # pygame.draw.circle(
        #     image,
        #     color,
        #     (radius, radius),
        #     radius,
        #     0
        # )

        self.image = self.animation[0]
        self.index = 0
        # self.image = image.convert()
        self.rect = self.image.get_rect()
        self.rect.center = pos

    def update(self, dt):
        self.image = self.animation[int(self.index)]
        self.index += 10 * dt
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


class Player(pygame.sprite.Sprite):

    def __init__(self,
                 radius,
                 color,
                 speed,
                 pos_init,
                 SCREEN_WIDTH,
                 SCREEN_HEIGHT,
                 UNIFORM_SPEED=False):

        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.UNIFORM_SPEED = UNIFORM_SPEED

        self.pos = vec2d(pos_init)
        self.vel = vec2d((0, 0))
        self.last_vel = vec2d((speed, 0))
        self.animation = []
        self.load_animations(radius * 2)

        # image = pygame.Surface([radius * 2, radius * 2])
        # image.set_colorkey((0, 0, 0))

        # pygame.draw.circle(
        #     image,
        #     color,
        #     (radius, radius),
        #     radius,
        #     0
        # )

        self.speed = speed
        self.image = self.animation[0][0]
        self.index = [0, 0, 0, 0]
        self.rect = self.image.get_rect()
        self.rect.center = pos_init
        self.radius = radius


    def update_image(self, dx, dy, dt):
        if dx != 0:
            dx /= abs(dx)
        elif dy != 0:
            dy /= abs(dy)
        else:
            return
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        index = directions.index((dx, dy))
        self.image = self.animation[index][int(self.index[index])]
        self.index[index] += 10 * dt
        if self.index[index] >= len(self.animation[index]):
            self.index[index] = self.index[index] % len(self.animation[index])

    
    def update(self, dx, dy, dt):
        if self.UNIFORM_SPEED:
            if dx == 0 and dy == 0:
                new_x = self.pos.x + self.vel.x * dt
                new_y = self.pos.y + self.vel.y * dt
                self.update_image(self.vel.x, self.vel.y, dt)
            else:
                new_x = self.pos.x + dx * dt
                new_y = self.pos.y + dy * dt
                self.vel.x = dx
                self.vel.y = dy
                self.update_image(dx, dy, dt)
        else:
            self.vel.x += dx
            self.vel.y += dy

            new_x = self.pos.x + self.vel.x * dt
            new_y = self.pos.y + self.vel.y * dt
            self.update_image(dx, dy, dt)

        # if its not against a wall we want a total decay of 50
        if new_x >= self.SCREEN_WIDTH - 1.2 * self.radius:
            # self.pos.x = self.SCREEN_WIDTH - self.radius * 2
            self.vel.x = 0.0
        elif new_x <= 1.2 * self.radius:
            # self.pos.x = 0.0
            self.vel.x = 0.0
        else:
            self.pos.x = new_x
            if not self.UNIFORM_SPEED:
                self.vel.x = self.vel.x * 0.975

        if new_y >= self.SCREEN_HEIGHT - 1.2 * self.radius:
            # self.pos.y = self.SCREEN_HEIGHT - self.radius * 2
            self.vel.y = 0.0
        elif new_y <= 1.2 * self.radius:
            # self.pos.y = 0.0
            self.vel.y = 0.0
        else:
            self.pos.y = new_y
            if not self.UNIFORM_SPEED:
                self.vel.y = self.vel.y * 0.975

        self.rect.center = (self.pos.x, self.pos.y)
        if self.vel.x != 0 or self.vel.y != 0:
            self.last_vel.x = self.vel.x
            self.last_vel.y = self.vel.y

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

class Substitute(pygame.sprite.Sprite):

    def __init__(self,
                 pos_init):

        pygame.sprite.Sprite.__init__(self)
        self.shape = 20
        self.pos = pos_init
        image = pygame.Surface((self.shape, self.shape))
        self.rect = image.get_rect()
        self.rect.center = pos_init


class Agent(pygame.sprite.Sprite):

    def __init__(self,
                 radius,
                 color,
                 speed,
                 pos_init,
                 SCREEN_WIDTH,
                 SCREEN_HEIGHT,
                 UNIFORM_SPEED=False):

        pygame.sprite.Sprite.__init__(self)
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.UNIFORM_SPEED = UNIFORM_SPEED

        self.pos = vec2d(pos_init)
        self.direction = vec2d((0, 1))
        self.animation = []
        self.load_animations(radius * 2)
        self.direction.normalize()

        # image = pygame.Surface([radius * 2, radius * 2])
        # image.set_colorkey((0, 0, 0))

        # pygame.draw.circle(
        #     image,
        #     color,
        #     (radius, radius),
        #     radius,
        #     0
        # )

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
        self.index[index] += 10 * dt
        if self.index[index] >= len(self.animation[index]):
            self.index[index] = self.index[index] % len(self.animation[index])

    
    def update(self, dx, dy, dt):
        if dx == 0 and dy == 0:
            pass
        else:
            self.pos.x += dx * dt
            self.pos.y += dy * dt
            self.direction.x = dx
            self.direction.y = dy
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
