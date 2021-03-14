import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import json

pygame.init()
screen = pygame.display.set_mode((600,600))
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0.0, 900)
draw_options = pymunk.pygame_util.DrawOptions(screen)

def loadConfig(configFile):

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)

class Swing():

    def __init__(self, space, swingConfig):
        self.space = space
        self.objects = self.generateSwing(swingConfig)

    def generateSwing(self, config):
        # specifies the top of the swing as defined by topPosition
        top = pymunk.Body(10,1000000, pymunk.Body.STATIC)
        top.position = Vec2d(*config['topPosition'])
        top_shape = pymunk.Poly.create_box(top, (20,20))

        self.space.add(top, top_shape)

        joints = [] # list of [body, shape]
        pivots = []

        joints.append([top, top_shape])

        for i, j in zip([[50, 100],[100, 200]], config['jointMasses']):
            '''
            Iterate through the list of coordinates as specified by jointLocations,
            relative to the top of the swing
            '''
            point = pymunk.Body(j, 100)
            point.position = top.position + Vec2d(*i)
            point_shape = pymunk.Segment(point, (0,0), (0,0), 5)
            # if the first joint, join to the top, otherwise join to the preceding joint
            if len(joints) == 0:
                pivot = pymunk.PinJoint(top, point, (0,0))
            else:
                pivot = pymunk.PinJoint(joints[-1][0], point) # selects the body component of the preceding joint
            pivot.collide_bodies = False
            joints.append([point, point_shape])
            pivots.append(pivot)

            self.space.add(point, point_shape)
            self.space.add(pivot)

        return {'rod' : joints, 'top' : [top, top_shape], 'pivots' : pivots}

    def getJointByNumber(self, num):
        return self.objects['rod'][num][0]

    def getPivotByNumber(self, num):
        return self.objects['pivots'][num]

    def moveDirection(self, middleIndex, endIndex):
        '''
        Calculates the vector which dictates the direction in which the middle mass should move.
        '''
        return (self.getJointByNumber(endIndex).position - self.getJointByNumber(middleIndex).position)
            
    def moveBody(self, middleIndex, endIndex, dirMultiplier=0.1, minLength=10, maxLength=150):
        '''
        The middle mass is middleIndex, and top or bottom pivot is endIndex. The speed at which the mass moves
        is controlled with dirMultiplier, and minLength and maxLength sets the limits on how far the mass can
        move, but maxLength may not be needed.

        The code which has been commented out is used to remove and remake the rods, but currently throws an error.
        Fixing the error might make for better performance.

        Further comments: Spamming the down key causes the swing to go crazy and start spinning lots, this was meant
        to be stopped by applying a minLength but this does not seem to currently work. Additionally, the seat can
        spin around the person, so this could potentially be looked into as it is unphysical.
        '''
        moveDistance = self.moveDirection(middleIndex, endIndex) * dirMultiplier
        tempPos = self.getJointByNumber(middleIndex).position + moveDistance
        distance = (abs(self.getJointByNumber(endIndex).position) - abs(tempPos) * -1)**0.5

        if distance > minLength and distance < maxLength:
            #space.remove(self.objects['pivots'], endIndex)
            self.getJointByNumber(middleIndex).position += moveDistance
            #pivot = pymunk.PinJoint(self.getJointByNumber(middleIndex), self.getJointByNumber(endIndex))
            #self.objects['pivots'][endIndex] = pivot
            #space.add(pivot)

    def moveUp(self):
        swing.moveBody(1, 0)

    def moveDown(self):
        swing.moveBody(1, -1)

    def render(self, screen):
        pass

    def update(self):
        self.eventListener()

    def eventListener(self):
        pass


class LearningArea():

    def __init__(self, configFile):
        pass

config = loadConfig('Standing\\config_standsquat.json')

swing = Swing(space, config['swingConfig'])

data = []

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        print("Moving mass UP")
        swing.moveUp()
    elif keys[pygame.K_DOWN]:
        print("Moving mass DOWN")
        swing.moveDown()
    elif keys[pygame.K_LEFT]:
        print("Pushing legs LEFT")
        swing.getJointByNumber(-1).apply_impulse_at_local_point(swing.getJointByNumber(-1).mass*Vec2d(-10,0))
    elif keys[pygame.K_RIGHT]:
        print("Pushing legs LEFT")
        swing.getJointByNumber(-1).apply_impulse_at_local_point(swing.getJointByNumber(-1).mass*Vec2d(10,0))
    data.append((pygame.time.get_ticks(), swing.getJointByNumber(-1).velocity.x, swing.getJointByNumber(-1).velocity.y))
    space.step(1/60)
    screen.fill((255,255,255))
    space.debug_draw(draw_options)
    pygame.display.flip()

    clock.tick(60)

data = pd.DataFrame(data, columns=['tick', 'vx', 'vy'])
data.to_csv('data.csv')
plt.plot(data)
