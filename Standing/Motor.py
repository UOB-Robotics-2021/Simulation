import pymunk
from pymunk.pygame_util import *
from pymunk.vec2d import Vec2d

import pygame
from pygame.locals import *

import math
from PIL import Image

space = pymunk.Space()
space.gravity = 0, -900
b0 = space.static_body

size = w, h = 1000, 1000
fps = 30
steps = 10

BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
WHITE = (255, 255, 255)

class App:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(size)
        self.draw_options = DrawOptions(self.screen)
        self.running = True
        self.gif = 0
        self.images = []

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                mot.rate = -10
            elif keys[pygame.K_DOWN]:
                mot.rate = 10
            else:
                mot.rate = 0

            self.draw()
            self.clock.tick(fps)

            for i in range(steps):
                space.step(1/fps/steps)

        pygame.quit()

    def draw(self):
        self.screen.fill(GRAY)
        space.debug_draw(self.draw_options)
        pygame.display.update()

        text = f'fpg: {self.clock.get_fps():.1f}'
        pygame.display.set_caption(text)

class Segment:
    def __init__(self, p0, v, radius=10):
        self.body = pymunk.Body()
        self.body.position = p0
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.density = 0.1
        shape.elasticity = 0.5
        shape.filter = pymunk.ShapeFilter(group=1)
        shape.color = (0, 255, 0, 0)
        space.add(self.body, shape)

class PivotJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0), collide=True):
        joint = pymunk.PinJoint(b, b2, a, a2)
        joint.collide_bodies = collide
        space.add(joint)

p = 100, 120
v = 60,10

p1 = 200, 120
arm = Segment(p1, v)
PivotJoint(b0, arm.body, p1)
mot = pymunk.SimpleMotor(b0, arm.body, 0)
space.add(mot)

App().run()
