# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:09:39 2021

@author: remib
"""

#Import modules
import pymunk
import pygame
import json
import os

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

dir_name = os.path.basename(os.getcwd())
if dir_name == "Stickman":
    config = loadConfig('config_stickman.json')["squatStandConfig"]
else:
    config = loadConfig('Stickman//config_stickman.json')["squatStandConfig"]

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

class SimpleMotor:
    def __init__(self, b, b2, rate):
        joint = pymunk.SimpleMotor(b, b2, rate)
        space.add(joint)
        
    def __setattr__(self, name, value):
         self.__dict__[name] = value

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
    def __init__(self, p0, v, m=10, radius=2, body=0):

        if body == 0:
            self.body = pymunk.Body()
            self.body.position = p0
            shape = pymunk.Segment(self.body, (0, 0), v, radius)
        else:
            print(body)
            shape = pymunk.Segment(body, (0, 0), v, radius)
        shape.mass = m
        shape.density = 0.1
        shape.elasticity = 0.5
        shape.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        shape.color = (0, 255, 0, 0)
        
        if body == 0:
            space.add(self.body, shape)
        else:
            space.add(shape)


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
        m=0
        while self.running:
            m=m+1
            #print(self.stick_figure.upperLeg.body.angle)
            for event in pygame.event.get():
                self.do_event(event)
 
            self.draw()
            self.clock.tick(fps)
            
            #self.stick_figure.apply_constraints(m)
            
            for i in range(steps):
                space.step(1/fps/steps)

        pygame.quit()

    def do_event(self, event):
        if event.type == QUIT:
            self.running = False
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: 
            self.stick_figure.stand()
        elif keys[pygame.K_DOWN]:
            self.stick_figure.squat()
        elif keys[pygame.K_RIGHT]:
            self.stick_figure.stop_motion()
        elif keys[pygame.K_i]: #i key:
            self.stick_figure.print_info()

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
        
        #Generate foot
        self.anklePosition = config["anklePosition"] + self.offset
        #self.footVector = self.dirVec("foot", theta, scale)
        #self.foot = Segment(self.anklePosition, self.footVector, self.limbMass("foot"))

        #Generate lower leg 
        self.lowerLegVector = self.dirVec("lowerLeg", theta, scale)
        self.lowerLeg = Segment(self.anklePosition, self.lowerLegVector, self.limbMass("lowerLeg"))
        
        #self.ankle = PivotJoint(self.foot.body, self.lowerLeg.body, (0,0))

        #Generate upper leg
        self.kneePosition = vector_sum(self.anklePosition, self.lowerLegVector)
        self.upperLegVector = self.dirVec("upperLeg", theta, scale)
        self.upperLeg = Segment(self.kneePosition, self.upperLegVector, self.limbMass("upperLeg"))
        self.knee = PivotJoint(self.lowerLeg.body, self.upperLeg.body, self.lowerLegVector)
        
        #Generate torso
        self.pelvisPosition = vector_sum(self.kneePosition, self.upperLegVector)
        self.torsoVector = self.dirVec("torso", theta, scale)
        self.torso = Segment(self.pelvisPosition, self.torsoVector, self.limbMass("torso"))
        self.pelvis = PivotJoint(self.upperLeg.body, self.torso.body, self.upperLegVector)
       
        #Generate upper arm
        self.shoulderPosition = vector_sum(self.pelvisPosition, self.torsoVector)
        self.upperArmVector = self.dirVec("upperArm", theta, scale)
        self.upperArm = Segment(self.shoulderPosition, self.upperArmVector, self.limbMass("upperArm"))
        self.shoulderArm = PivotJoint(self.torso.body, self.upperArm.body, self.torsoVector)

        #Generate lower arm
        self.elbowPosition = vector_sum(self.shoulderPosition, self.upperArmVector)
        
        self.lowerArmVector = self.dirVec("lowerArm", theta, scale)
        self.lowerArm = Segment(self.elbowPosition, self.lowerArmVector, self.limbMass("lowerArm"))
        self.handPosition = vector_sum(self.elbowPosition, self.lowerArmVector)
        self.elbow = PivotJoint(self.upperArm.body, self.lowerArm.body, self.upperArmVector)
        self.neckPosition = vector_sum(self.pelvisPosition, self.torsoVector)
        
        #Add neck
        self.neckVector = self.dirVec("neck", theta, scale)
        self.neck = Segment(self.neckPosition, self.neckVector, 10)
        self.neckJoint = PivotJoint(self.upperArm.body, self.neck.body)
        
        #Generate head
        headPosition = vector_sum(self.neckPosition, self.neckVector)
        headRadius = config["head"][0]
        
        #Ensure head's center of gravity in line with neck
        norm = (self.neckVector[0]**2 + self.neckVector[1]**2)**0.5
        A = headRadius/norm
        dp = [self.neckVector[0]*A, self.neckVector[1]*A]
        self.headJointPosition = vector_sum(self.neckPosition, self.neckVector)
        self.headPosition = [(self.headJointPosition[0]+dp[0]), (self.headJointPosition[1]+dp[1])]
        
        self.head = Circle(pos=headPosition, radius=headRadius)
        self.headJoint = PinJoint(self.neck.body, self.head.body, self.neckVector)

        self.stopped_at = 0

    def dirVec(self, limb, rotation, scale):
        angle = config[limb][0] + rotation
        return scale * config[limb][1] * Vec2d(np.cos(angle * np.pi/180), np.sin(angle * np.pi/180))
    
    def limbMass(self, limb):
        return config[limb][2]
    
    def squat(self, max_force=100):
        """
        Applies a counter-clockwise torque to stickman's upper leg (pelvis and knee) to cause squatting.
        
        Parameters:
            max_force(int) - maximum force to be applied
        
        """
        if self.upperLeg.body.position[1] > self.torso.body.position[1]:
            print("Squat")
            
            #Apply force to pelvis and opposite force to knee to cause anti-clockwise torque of upper leg
            f1 = [-max_force*np.abs(np.cos(self.upperLeg.body.angle)), -max_force*np.abs(np.sin(self.upperLeg.body.angle))]
            f2 = [-f1[0], -f1[1]]
            self.upperLeg.body.apply_impulse_at_local_point(f1, self.pelvisPosition)
            self.upperLeg.body.apply_impulse_at_local_point(f2, self.kneePosition)
        else:
            print("Sitting restricted")
            self.stop_motion()
        """
        #Apply force to elbow and opposite force to hand to cause clockwise torque of upper leg
        f1 = [max_force*np.sin(self.lowerArm.body.angle), max_force*np.cos(self.upperArm.body.angle)]
        f2 = [-f1[0], -f1[1]]
        self.lowerArm.body.apply_impulse_at_local_point(f1, self.elbowPosition)
        self.lowerArm.body.apply_impulse_at_local_point(f2, self.handPosition)
        """

    def stand(self, max_force=1000):
        """
        Make stickman stand using counter-clockwise upper leg torque and clockwise upper arm torque.
        
        Parameters:
            max_force(int) - maximum force to be applied
        
        """
        
        if self.upperLeg.body.position[0] > self.torso.body.position[0]:
            #Apply force to pelvis and opposite force to knee to cause clockwise torque of upper leg
            f1 = [-max_force*np.cos(self.upperLeg.body.angle), -max_force*np.sin(self.upperLeg.body.angle)]
            f2 = [-f1[0], -f1[1]]
            print(f1, f2)
            self.upperLeg.body.apply_impulse_at_local_point(f1, self.pelvisPosition)
            self.upperLeg.body.apply_impulse_at_local_point(f2, self.kneePosition)
        else:
            print("Standing restricted")
            self.stop_motion()
        """
        #Apply force to elbow and opposite force to hand to cause clockwise torque of upper leg
        f1 = [-max_force*np.sin(self.lowerArm.body.angle), -max_force*np.cos(self.upperArm.body.angle)]
        f2 = [f1[0], f1[1]]
        self.lowerArm.body.apply_impulse_at_local_point(f1, self.elbowPosition)
        self.lowerArm.body.apply_impulse_at_local_point(f2, self.handPosition)
        """
    def stop_motion(self):
        """
        Stops stickman moving by setting velocity of all its parts to 0
        """
        #print("Stop motion")
        
        self.foot.body.velocity = pymunk.Vec2d(0, 0)
        self.lowerLeg.body.velocity = pymunk.Vec2d(0, 0)
        self.upperLeg.body.velocity = pymunk.Vec2d(0, 0)
        self.torso.body.velocity = pymunk.Vec2d(0, 0)
        self.upperArm.body.velocity = pymunk.Vec2d(0, 0)
        self.lowerArm.body.velocity = pymunk.Vec2d(0, 0)
        self.neck.body.velocity = pymunk.Vec2d(0, 0)
        self.head.body.velocity = pymunk.Vec2d(0, 0)
        
        #self.upperLeg.body.sleep_with_group(self.upperLeg.body, 10)
        
    def apply_constraints(self, i=0):
        """
        Stops motion if joints breach angle range
        """
 
        if self.upperLeg.body.angle < -1.1*config["maxKneeAngle"] and (self.stopped_at==0 or i < (self.stopped_at+10)):
            print("Stop motion", i)
            self.stop_motion()
            if i > (self.stopped_at+20): 
                self.stopped_at = i
        elif self.upperLeg.body.position[1] < self.torso.body.position[1] and (self.stopped_at==0 or i < (self.stopped_at+10)):
            print("Stop squatting")
            self.stop_motion()
            if i > (self.stopped_at+20): 
                self.stopped_at = i
        elif self.upperLeg.body.position[0] < self.torso.body.position[0]:
            self.stop_motion()
            print("stop standingg")
        elif self.upperLeg.body.angle > 0:
            print("Stop motion")
            self.stop_motion()
    
            
    def print_info(self, max_force=10):
        print(self.upperLeg.body.angle)
man = Stickman(x=150, y=250, theta=0, scale=1)
App(man).run()
