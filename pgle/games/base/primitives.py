import pygame
import math
from ..utils import vec2d

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
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dt):

        self.life -= dt
        if self.life <= -1:
            self.kill()



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
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

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

        self.rect.center = ((self.pos.x, self.pos.y))


class Wall(pygame.sprite.Sprite):

    def __init__(self, pos, w, h, color):
        pygame.sprite.Sprite.__init__(self)

        self.pos = vec2d(pos)
        self.w = w
        self.h = h
        self.color = color
        image = pygame.Surface([w, h])
        image.fill(color)
        self.image = image.convert()

        self.rect = self.image.get_rect()
        self.rect.center = pos

    def draw(self, screen):
        pygame.draw.rect(
            screen, self.color, [
                self.pos.x, self.pos.y, self.w, self.h], 0)


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

        image = pygame.Surface([radius * 2, radius * 2])
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            color,
            (radius, radius),
            radius,
            0
        )

        self.speed = speed
        self.image = image.convert()
        self.rect = self.image.get_rect()
        self.rect.center = pos_init
        self.radius = radius

    def update(self, dx, dy, dt):
        if self.UNIFORM_SPEED:
            if dx == 0 and dy == 0:
                new_x = self.pos.x + self.vel.x * dt
                new_y = self.pos.y + self.vel.y * dt
            else:
                new_x = self.pos.x + dx * dt
                new_y = self.pos.y + dy * dt
                self.vel.x = dx
                self.vel.y = dy

        else:
            self.vel.x += dx
            self.vel.y += dy

            new_x = self.pos.x + self.vel.x * dt
            new_y = self.pos.y + self.vel.y * dt

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