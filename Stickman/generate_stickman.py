# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:09:39 2021

@author: remib
"""

#Import modules
import pymunk
import pygame
import json

from pymunk.pygame_util import *
from pymunk.vec2d import Vec2d

from pygame.locals import *

import numpy as np

#Custom functions
def loadConfig(configFile):

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)

def vector_sum(a1, a2):
    return [(a1[0]+a2[0]), (a1[1]+a2[1])]

config = loadConfig('config_stickman.json')["squatStandConfig"]

#Set-up environment
space = pymunk.Space()
space.gravity = 0, config["g"]
b0 = space.static_body

size = w, h = config["screenSize"]
fps = 30
steps = 10

BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
WHITE = (255, 255, 255)

#Classes 

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
        shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        shape.color = (0, 255, 0, 0)
        space.add(self.body, shape)


class Circle:
    def __init__(self, pos, radius=20):
        self.body = pymunk.Body()
        self.body.position = pos
        shape = pymunk.Circle(self.body, radius)
        shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        
        shape.density = 0.01
        shape.friction = 0.5
        shape.elasticity = 1
        space.add(self.body, shape)

class App:
    def __init__(self, man):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(size)
        self.draw_options = DrawOptions(self.screen)
        self.running = True
        self.gif = 0
        self.images = []
        self.stick_figure = man

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
            print(dir(lower_leg.body))
            print(foot.body.position)
            #print(dir(leg.body))
            #leg.body.apply_impulse_at_local_point([10000,0], leg.body.position)
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

def CircleOrientation(p_i, v, r):
    norm = (v[0]**2 + v[1]**2)**0.5
    A = r/norm
    dp = [v[0]*A, v[1]*A]
    p = [(p_i[0]-dp[0]), (p_i[1]-dp[1])]
    
#Code to generate figure
class Stickman:
    def __init__(self, x, y, theta=0, scale=1):
        # In the json file, the format for limbs is --> "limb": [angle, length, mass].
        # The head has format --> "head": [radius, mass]
        self.offset = Vec2d(x, y) # Allows you to move the stickman around
        
        anklePosition = config["anklePosition"] + self.offset

        footVector = self.dirVec("foot", theta, scale)
        self.foot = Segment(anklePosition, footVector, self.limbMass("foot"))

        lowerLegVector = self.dirVec("lowerLeg", theta, scale)
        self.lowerLeg = Segment(anklePosition, lowerLegVector, self.limbMass("lowerLeg"))
        self.ankle = PivotJoint(self.foot.body, self.lowerLeg.body, (0,0))

        kneePosition = vector_sum(anklePosition, lowerLegVector)

        upperLegVector = self.dirVec("upperLeg", theta, scale)
        self.upperLeg = Segment(kneePosition, upperLegVector, self.limbMass("upperLeg"))
        self.knee = PivotJoint(self.lowerLeg.body, self.upperLeg.body, lowerLegVector)

        pelvisPosition = vector_sum(kneePosition, upperLegVector)
        
        torsoVector = self.dirVec("torso", theta, scale)
        self.torso = Segment(pelvisPosition, torsoVector, self.limbMass("torso"))
        self.pelvis = PivotJoint(self.upperLeg.body, self.torso.body, upperLegVector)

        shoulderPosition = vector_sum(pelvisPosition, torsoVector)

        upperArmVector = self.dirVec("upperArm", theta, scale)
        self.upperArm = Segment(shoulderPosition, upperArmVector, self.limbMass("upperArm"))
        self.shoulder = PivotJoint(self.torso.body, self.upperArm.body, torsoVector)

        elbowPosition = vector_sum(shoulderPosition, upperArmVector)
        
        lowerArmVector = self.dirVec("lowerArm", theta, scale)
        self.lowerArm = Segment(elbowPosition, lowerArmVector, self.limbMass("lowerArm"))
        self.elbow = PivotJoint(self.upperArm.body, self.lowerArm.body, upperArmVector)
        neckPosition = vector_sum(pelvisPosition, torsoVector)
        
        #Add neck
        neckVector = self.dirVec("neck", theta, scale)
        self.neck = Segment(neckPosition, neckVector, 10)
        self.neckJoint = PivotJoint(self.upperArm.body, self.neck.body)
        
        #Add head
        headPosition = vector_sum(neckPosition, neckVector)
        headRadius = config["head"][0]
        
        #Ensure head's center of gravity in line with neck
        norm = (neckVector[0]**2 + neckVector[1]**2)**0.5
        A = headRadius/norm
        dp = [neckVector[0]*A, neckVector[1]*A]
        headJointPosition = vector_sum(neckPosition, neckVector)
        headPosition = [(headJointPosition[0]+dp[0]), (headJointPosition[1]+dp[1])]
        
        self.head = Circle(pos=headPosition, radius=headRadius)
        self.headJoint = PinJoint(self.neck.body, self.head.body, neckVector)

    def dirVec(self, limb, rotation, scale):
        angle = config[limb][0] + rotation
        return scale * config[limb][1] * Vec2d(np.cos(angle * np.pi/180), np.sin(angle * np.pi/180))
    
    def limbMass(self, limb):
        return config[limb][2]

man = Stickman(x=150, y=250, theta=0, scale=1)
App(man).run()