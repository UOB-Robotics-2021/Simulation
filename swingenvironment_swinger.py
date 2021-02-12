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

class Person():

    def __init__(self, space, pos, mass=5):
        self.space = space
        self.pos = Vec2d(*pos)
        self.mass = mass
        self.objects = self.generatePerson()
        self.legs = self.objects['body'][1]
        self.knee_motor = self.objects['pivots'][1]

    def generatePerson(self):
        body = pymunk.Body(0.75*self.mass, 100000000000000) # assumes the body from the quads up make up 75% of mass
        body.position = self.pos
        
        legs = pymunk.Body(0.25*self.mass, 100)
        legs.position = self.pos

        torso_shape = pymunk.Segment(body, (0,0), (0, -30), 3)
        bottom_shape = pymunk.Segment(body, (0,0), (20, 0), 3)
        self.space.add(body, torso_shape, bottom_shape)
        
        legs_shape = pymunk.Segment(legs, (20, 0), (20, 20), 3)
        self.space.add(legs, legs_shape)
        
        knee_joint = pymunk.PinJoint(legs, body, (20, 0), (20,0))
        knee_motor = pymunk.SimpleMotor(legs, body, 0)
        knee_joint.collide_bodies = False
        self.space.add(knee_joint)
        self.space.add(knee_motor)
        
        return {'body' : [(body, torso_shape, legs_shape), (legs, legs_shape)], 'pivots' : [knee_joint, knee_motor]}

    def update(self):
        self.limitRotation()
    
    def limitRotation(self, limits=(np.pi/4, -np.pi/2)):
        # prevents legs from rotating too far, +pi/4 is 45deg clockwise, -pi/2 is 90 deg anticlockwise
        if self.legs[0].angle > limits[0]:
            self.legs[0].angle = limits[0]
        elif self.legs[0].angle < limits[1]:
            self.legs[0].angle = limits[1]
            

class Swing():

    def __init__(self, space, swingConfig):
        self.space = space
        self.objects = self.generateSwing(swingConfig)
        self.seat = self.getJointByNumber(-1)
        self.pos = self.seat.position

    def generateSwing(self, config):
        # specifies the top of the swing as defined by topPosition
        top = pymunk.Body(10, 1000000, pymunk.Body.STATIC)
        top.position = Vec2d(*config['topPosition'])
        top_shape = pymunk.Poly.create_box(top, (20,20))

        self.space.add(top, top_shape)

        joints = [] # list of [body, shape]
        pivots = []
        for i, j in zip(config['jointLocations'], config['jointMasses']):
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

    def render(self, screen):
        pass

    def update(self):
        self.eventListener()

    def eventListener(self):
        pass


class LearningArea():

    def __init__(self, space, configFile):
        self.space = space
        self.configFile = configFile
        self.swing = Swing(self.space, self.configFile)
        self.person = None

        if self.configFile['person']:
            self.person = Person(space, self.swing.pos, mass=configFile['person'])
            self.seat = pymunk.PinJoint(self.swing.seat, self.person.objects['body'][0][0])
            self.seat.collide_bodies = False
            self.space.add(self.seat)
    
    def update(self):
        self.eventHandler()
        self.swing.update()
        if self.person:
            self.person.update()

    def eventHandler(self):
        keys = pygame.key.get_pressed()
        if self.person:
            if keys[pygame.K_UP]:
                self.person.knee_motor.rate = -np.pi
            elif keys[pygame.K_DOWN]:
                self.person.knee_motor.rate = np.pi
        else:
            if keys[pygame.K_UP]:
                self.swing.getJointByNumber(-1).apply_impulse_at_local_point(self.swing.getJointByNumber(-1).mass*Vec2d(10,0))
            elif keys[pygame.K_DOWN]:
                self.swing.getJointByNumber(-1).apply_impulse_at_local_point(self.swing.getJointByNumber(-1).mass*Vec2d(-10,0))

config = loadConfig('config_swinger.json')

la = LearningArea(space, config['swingWithPerson'])

data = []

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    la.update()
    space.step(1/60)
    screen.fill((255,255,255))
    space.debug_draw(draw_options)
    pygame.display.flip()

    clock.tick(60)
data = pd.DataFrame(data, columns=['tick', 'vx', 'vy'])
data.to_csv('data.csv')