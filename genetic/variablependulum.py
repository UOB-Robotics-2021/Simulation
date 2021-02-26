# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:47:28 2021
@author: remib
"""

#Import modules
import pymunk
import pymunk.pygame_util
import pygame
import numpy as np
from neuralnetwork import NeuralNetwork

#Get pendulum config details
import json

def loadConfig(configFile):

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)

config = loadConfig('config.json')["variablePendulumConfig"]
GRAY = (220, 220, 220)
space = pymunk.Space()
space.gravity = [0, config["g"]]


class VariablePendulum():
    
    def __init__(self, space, configFile):
        self.config = configFile
        self.space = space
        self.objects = self.generateModel()
        
        self.neuralnetwork = NeuralNetwork(2, 3, 3, 10)
        self.neuralnetwork.loadNN('variablePendulum')
        self.data = []
    
    def update(self):
        action = self.neuralnetwork.forward([abs(self.angle()), self.body.angular_velocity])
        self.extendRope(np.argmax(action))
        self.data.append((self.angle(), self.body.angular_velocity, np.argmax(action)))

    def generateModel(self):
        #Create objects
        moment_of_inertia = 0.25*self.config["flywheelMass"]*self.config["flywheelRadius"]**2
        
        self.body = pymunk.Body(mass=self.config["flywheelMass"], moment=moment_of_inertia)
        self.body.position = self.config["flywheelInitialPosition"]
        self.circle = pymunk.Circle(self.body, radius=self.config["flywheelRadius"])
        self.circle.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        self.circle.friction = 90000
        #Create joints
        self.joint = pymunk.PinJoint(self.space.static_body, self.body, self.config["pivotPosition"])
        
        self.top = self.body.position[1] + self.config["flywheelRadius"]

        self.space.add(self.body, self.circle, self.joint)

    def angle(self):
        y = self.config['pivotPosition'][1] - self.body.position.y
        x = self.body.position.x - self.config['pivotPosition'][0]
        angle = np.arctan(x/y)
        return angle

    def extendRope(self, direction):
        if direction == 0:
            self.joint.distance = self.config["minPendulumLength"]
        elif direction == 1:
            self.joint.distance = self.config["maxPendulumLength"]
        else:
            pass
        
    def angVel(self):  # return angular velocity
        v = [self.body.velocity.x, self.body.velocity.y]
        y = self.config['pivotPosition'][1] - self.body.position.y
        x = self.body.position.x - self.config['pivotPosition'][0]
        r = [x, y]# vector for position of pendulum
        L = np.cross(r, v) 
        return L

#Initialize Simulation
pygame.init()
screen_size = 600, 500
screen = pygame.display.set_mode(screen_size)
draw_options = pymunk.pygame_util.DrawOptions(screen)
running = True
clock = pygame.time.Clock()

vp = VariablePendulum(space, config)

#Main Simulation loop
running = True 
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    #print(body.position)
    clock.tick(60)
    vp.update()
    screen.fill(GRAY)
    space.debug_draw(draw_options)
    pygame.display.update()
    #space.step(0.01)
    space.step(0.01)

pygame.quit()