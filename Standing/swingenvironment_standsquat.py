# Import modules
import pymunk
import pygame
import json

from pymunk.pygame_util import *
from pymunk.vec2d import Vec2d

from pygame.locals import *

import numpy as np

import pandas as pd
import math
import matplotlib.pyplot as plt

# Custom functions
def loadConfig(configFile):

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)

config = loadConfig('Standing\\config_standsquat.json')

# Set-up environment
space = pymunk.Space()
space.gravity = 0, 0
b0 = space.static_body

size = w, h = 600, 500
fps = 30
steps = 10

BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
WHITE = (255, 255, 255)

# Classes from Pymunk
class PinJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0)):
        joint = pymunk.PinJoint(b, b2, a, a2)
        space.add(joint)

class PivotJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0), collide=True):
        joint = pymunk.PinJoint(b, b2, a, a2)
        joint.collide_bodies = collide
        space.add(joint)

class Segment:
    def __init__(self, p0, v, m=10, radius=2):
        self.body = pymunk.Body()
        self.body.position = p0
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.mass = m
        shape.density = 0.1
        shape.elasticity = 0.5
        shape.filter = pymunk.ShapeFilter(group=1)
        shape.color = (0, 255, 0, 0)
        space.add(self.body, shape)


class Circle:
    def __init__(self, pos, radius=20):
        self.body = pymunk.Body()
        self.body.position = pos
        shape = pymunk.Circle(self.body, radius)
        shape.density = 0.01
        shape.friction = 0.5
        shape.elasticity = 1
        shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        space.add(self.body, shape)

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
                self.do_event(event)
 
            self.draw()
            self.clock.tick(fps)

            for i in range(steps):
                space.step(1/fps/steps)

        pygame.quit()

    def do_event(self, event):
        if event.type == QUIT:
            self.running = False
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: 
            upper_leg.body.apply_impulse_at_local_point([10000,0], upper_leg.body.position)
        if event.type == KEYDOWN:
            if event.key in (K_q, K_ESCAPE):
                self.running = False

            elif event.key == K_p:
                pygame.image.save(self.screen, 'joint.png')

            elif event.key == K_g:
                self.gif = 60

    def draw(self):
        self.screen.fill(GRAY)
        space.debug_draw(self.draw_options)
        pygame.display.update()

        text = f'fpg: {self.clock.get_fps():.1f}'
        pygame.display.set_caption(text)
        self.make_gif()

    def make_gif(self):
        if self.gif > 0:
            strFormat = 'RGBA'
            raw_str = pygame.image.tostring(self.screen, strFormat, False)
            image = Image.frombytes(
                strFormat, self.screen.get_size(), raw_str)
            self.images.append(image)
            self.gif -= 1
            if self.gif == 0:
                self.images[0].save('joint.gif',
                                    save_all=True, append_images=self.images[1:],
                                    optimize=True, duration=1000//fps, loop=0)
                self.images = []

# Code to generate figure
class Stickman:
    def __init__(self, config, x, y, theta=0, scale=1):
        # In the json file, the format for limbs is --> "limb": [angle, length, mass].
        # The head has format --> "head": [radius, mass]
        self.offset = Vec2d(x, y) # Allows you to move the stickman around
        self.config = config
        anklePosition = self.config["anklePosition"] + self.offset

        self.footVector = self.dirVec("foot", theta, scale)
        self.foot = Segment(anklePosition, self.footVector, self.limbMass("foot"))

        lowerLegVector = self.dirVec("lowerLeg", theta, scale)
        self.lowerLeg = Segment(anklePosition, lowerLegVector, self.limbMass("lowerLeg"))
        self.ankle = PivotJoint(self.foot.body, self.lowerLeg.body, (0,0))

        kneePosition = self.vectorSum(anklePosition, lowerLegVector)

        upperLegVector = self.dirVec("upperLeg", theta, scale)
        self.upperLeg = Segment(kneePosition, upperLegVector, self.limbMass("upperLeg"))
        self.knee = PivotJoint(self.lowerLeg.body, self.upperLeg.body, lowerLegVector)

        pelvisPosition = self.vectorSum(kneePosition, upperLegVector)
        
        torsoVector = self.dirVec("torso", theta, scale)
        self.torso = Segment(pelvisPosition, torsoVector, self.limbMass("torso"))
        self.pelvis = PivotJoint(self.upperLeg.body, self.torso.body, upperLegVector)

        shoulderPosition = self.vectorSum(pelvisPosition, torsoVector)

        upperArmVector = self.dirVec("upperArm", theta, scale)
        self.upperArm = Segment(shoulderPosition, upperArmVector, self.limbMass("upperArm"))
        self.shoulder = PivotJoint(self.torso.body, self.upperArm.body, torsoVector)

        elbowPosition = self.vectorSum(shoulderPosition, upperArmVector)

        self.lowerArmVector = self.dirVec("lowerArm", theta, scale)
        self.lowerArm = Segment(elbowPosition, self.lowerArmVector, self.limbMass("lowerArm"))
        self.elbow = PivotJoint(self.upperArm.body, self.lowerArm.body, upperArmVector)

        headRadius = config["head"][0]
        headPosition = shoulderPosition + (headRadius * Vec2d(np.sin(theta * np.pi/180), -np.cos(theta * np.pi/180)))
        self.head = Circle(headPosition, headRadius)
        self.headJoint = PivotJoint(self.torso.body, self.head.body, torsoVector + (headRadius * Vec2d(np.sin(theta * np.pi/180), -np.cos(theta * np.pi/180))))


    def dirVec(self, limb, rotation, scale):
        angle = self.config[limb][0] + rotation
        return scale * self.config[limb][1] * Vec2d(np.cos(angle * np.pi/180), np.sin(angle * np.pi/180))
    
    def limbMass(self, limb):
        return self.config[limb][2]

    def vectorSum(self, v1, v2):
        return [(v1[0]+v2[0]), (v1[1]+v2[1])]

# Code for swing
class Swing():

    def __init__(self, space, swingConfig, theta):
        self.space = space
        self.theta = theta
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

        for i, j in zip(config['jointLocations'], config['jointMasses']):
            '''
            Iterate through the list of coordinates as specified by jointLocations,
            relative to the top of the swing
            '''
            point = pymunk.Body(j, 100)
            point.position = (top.position + Vec2d(*i)) * Vec2d(np.cos(theta * np.pi/180), np.sin(theta * np.pi/180))
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

theta = 0

man = Stickman(config=config["squatStandConfig"], x=285, y=350, theta=theta, scale=0.7)
swing = Swing(space, config['swingConfig'], theta=0)

holdHand = PinJoint(man.lowerArm.body, swing.getJointByNumber(1), man.lowerArmVector)
holdFoot = PinJoint(man.foot.body, swing.getJointByNumber(-1), man.footVector)

data = []

App().run()

data = pd.DataFrame(data, columns=['tick', 'vx', 'vy'])
data.to_csv('data.csv')
plt.plot(data)