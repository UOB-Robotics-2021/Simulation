# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:59:31 2021

@author: Kieron
"""

import numpy as np
import pymunk
from pymunk.vec2d import Vec2d
import pygame
from neuralnetwork import NeuralNetwork

'''
Skeleton Model 
if followed, genetic_environment should, in theory, be able to run a machine learning simulation
on the model provided you define a learning environment and feed an appropriate config file

e.g. learningenvironment = LearningEnvironment(space, Model, loadConfig('config.json')['modelConfig'])

class Model():
    def __init__(self, space, configFile, numActions):
        initialise stuff here
    
    def generateModel(self):
        build the model
    
    def destroyModel(self):
        clear the model to allow for reset
    
    def update(self):
        set up the inputs for the step function; determine an action via the neural network
    
    def step(self, action):
        change the system in some way in response to the action determined by the neural network
    
    def calculateFitness(self):
        determine the fitness of the model at the end of the simulation (could be changed?)

'''

class VariablePendulum():
    
    def __init__(self, space, configFile, numActions=500):
        self.num_actions = numActions
        self.config = configFile
        self.space = space
        self.colour = (np.random.uniform(0,255),np.random.uniform(0,255),np.random.uniform(0,255),255)
        self.objects = self.generateModel()

        self.initial_amplitude = abs(self.angle())
        self.maximum_amplitude = abs(self.angle())
        self.fitness = 0
        self.action_step = 0
        self.num_reversals = 0
        self.last_action = 0
        self.prev_angle = 0
        self.prev_dir = 2
        self.steps_to_done = 0
        self.done = False
        self.complete = False
        self.movable = True
        
        self.neuralnetwork = NeuralNetwork(4, 3, *self.config['hiddenLayers'])

    def update(self):
        if not self.done:
            dtheta = self.angle()-self.prev_angle
            ddtheta = self.angle()-dtheta
            action = self.neuralnetwork.forward([abs(self.angle()), abs(dtheta), abs(ddtheta), self.maximum_amplitude])
            self.step(action)
            self.checkMovable()

    def generateModel(self):
        #Create objects
        moment_of_inertia = 0.25*self.config["flywheelMass"]*self.config["flywheelRadius"]**2
        
        self.body = pymunk.Body(mass=self.config["flywheelMass"], moment=moment_of_inertia)
        self.body.position = self.config["flywheelInitialPosition"]
        self.circle = pymunk.Circle(self.body, radius=self.config["flywheelRadius"])
        self.circle.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        self.circle.friction = 90000
        self.circle.color = self.colour
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
        if direction != self.prev_dir and direction != 2: self.num_reversals += 1
        self.prev_dir = direction
        if direction == 0:
            if self.movable:
                self.joint.distance = self.config["minPendulumLength"]
                self.last_action = self.action_step
                self.movable = False
        elif direction == 1:
            if self.movable:
                self.joint.distance = self.config["maxPendulumLength"]
                self.last_action = self.action_step
                self.movable = False
        else:
            pass
    
    def checkMovable(self):
        if self.action_step - self.last_action > self.config['actionDelay']:
            self.movable = True

    def step(self, action):
        self.action_step += 1
        self.prev_angle = self.angle()
        
        amplitude = abs(self.angle())
        if amplitude > self.maximum_amplitude:
            self.maximum_amplitude = amplitude
        self.extendRope(np.argmax(action))
        
        if self.action_step >= self.num_actions:
            self.done = True
        if self.body.position.y < self.config['pivotPosition'][1]:
            self.done = True
            self.complete = True

    def calculateFitness(self):
        if self.complete:
            return (self.maximum_amplitude-self.initial_amplitude)+200/self.num_reversals
        else:
            return (self.maximum_amplitude-self.initial_amplitude)
    
    def destroyModel(self):
        self.space.remove(self.body, self.circle, self.joint)

class SmoothVariablePendulum():
    
    def __init__(self, space, configFile, numActions=500):
        self.num_actions = numActions
        self.config = configFile
        self.space = space
        self.colour = (np.random.uniform(0,255),np.random.uniform(0,255),np.random.uniform(0,255),255)
        self.objects = self.generateModel()
        

        self.initial_amplitude = abs(self.angle())
        self.maximum_amplitude = abs(self.angle())
        self.fitness = 0
        self.action_step = 0
        self.num_reversals = 0
        self.last_action = 0
        self.prev_angle = 0
        self.prev_dir = 2
        self.steps_to_done = 0
        self.done = False
        self.complete = False
        self.movable = True
        
        self.neuralnetwork = NeuralNetwork(6, 5, *self.config['hiddenLayers'])

    def update(self):
        if not self.done:
            dtheta = self.angle()-self.prev_angle
            ddtheta = self.angle()-dtheta
            action = self.neuralnetwork.forward([abs(self.angle()), abs(dtheta), abs(ddtheta), self.maximum_amplitude, self.body.angular_velocity, self.joint.distance])
            self.step(action)
            self.checkMovable()

    def generateModel(self):
        #Create objects
        moment_of_inertia = 0.25*self.config["flywheelMass"]*self.config["flywheelRadius"]**2
        
        self.body = pymunk.Body(mass=self.config["flywheelMass"], moment=moment_of_inertia)
        self.body.position = self.config["flywheelInitialPosition"]
        self.circle = pymunk.Circle(self.body, radius=self.config["flywheelRadius"])
        self.circle.filter = pymunk.ShapeFilter(categories=0b1,mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)
        self.circle.friction = 90000
        self.circle.color = self.colour
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
        if direction != self.prev_dir and direction != 2: self.num_reversals += 1
        self.prev_dir = direction
        if direction == 0:
            if self.movable:
                self.joint.distance -= self.config['squattingSpeed']
                if self.joint.distance < self.config['minPendulumLength']:
                    self.joint.distance = self.config["minPendulumLength"]
        elif direction == 1:
            if self.movable:
                self.joint.distance += self.config['squattingSpeed']
                if self.joint.distance > self.config['maxPendulumLength']:
                    self.joint.distance = self.config["maxPendulumLength"]
        elif direction == 2:
            self.body.angular_velocity += 1
            if self.body.angular_velocity > 5:
                self.body.angular_velocity = 5
        elif direction == 3:
            self.body.angular_velocity -= 1
            if self.body.angular_velocity < -5:
                self.body.angular_velocity = -5
        else:
            pass
    
    def checkMovable(self):
        if self.action_step - self.last_action > self.config['actionDelay']:
            self.movable = True

    def step(self, action):
        self.action_step += 1
        self.prev_angle = self.angle()
        
        amplitude = abs(self.angle())
        if amplitude > self.maximum_amplitude:
            self.maximum_amplitude = amplitude
        self.extendRope(np.argmax(action))
        
        if self.action_step >= self.num_actions:
            self.done = True
        if self.body.position.y < self.config['pivotPosition'][1]:
            self.done = True
            self.complete = True

    def calculateFitness(self):
        if self.complete:
            return (self.maximum_amplitude-self.initial_amplitude)+200/self.num_reversals
        else:
            return (self.maximum_amplitude-self.initial_amplitude)
    
    def destroyModel(self):
        self.space.remove(self.body, self.circle, self.joint)


'''
ALL BELOW IS WIP; NONE ARE VERIFIED TO RUN AS EXPECTED
TODO : Add functions for machine learning implementation
    - fitness functions
    - step functions
'''

class Swing():

    def __init__(self, space, swingConfig, numActions = 500):
        self.space = space
        self.objects = self.generateSwing(swingConfig)
        self.seat = self.getJointByNumber(-1)
        self.pos = self.seat.position
        self.config = swingConfig
        self.num_actions = numActions
        self.colour = (np.random.uniform(0,255),np.random.uniform(0,255),np.random.uniform(0,255),255)
        # self.swing = Swing(self.space, self.config)
        
        self.initial_amplitude = abs(self.angle())
        self.maximum_amplitude = abs(self.angle())
        self.fitness = 0
        self.action_step = 0
        self.num_reversals = 0
        self.last_action = 0
        self.prev_angle = 0
        self.prev_dir = 2
        self.steps_to_done = 0
        self.done = False
        self.complete = False
        self.movable = True
        
        self.neuralnetwork = NeuralNetwork(5, 4, *self.config['hiddenLayers'])

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
        self.joints = joints
        self.pivots = pivots
        

        return {'rod' : joints, 'top' : [top, top_shape], 'pivots' : pivots}
    
    def angle(self):
        y = self.config['pivotPosition'][1] - self.seat.position.y
        x = self.seat.position.x - self.config['pivotPosition'][0]
        angle = np.arctan(x/y)
        return angle

    def getJointByNumber(self, num):
        return self.objects['rod'][num][0]
    
    def checkMovable(self):
        if self.action_step - self.last_action > self.config['actionDelay']:
            self.movable = True

    def update(self):
        if not self.done:
            dtheta = self.angle()-self.prev_angle
            ddtheta = self.angle()-dtheta
            action = self.neuralnetwork.forward([abs(self.angle()), abs(dtheta), abs(ddtheta), self.maximum_amplitude, self.seat.angular_velocity])
            self.step(action)
            self.checkMovable()

    def eventHandler(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.getJointByNumber(-1).apply_impulse_at_local_point(self.getJointByNumber(-1).mass*Vec2d(10,0))
        elif keys[pygame.K_DOWN]:
            self.getJointByNumber(-1).apply_impulse_at_local_point(self.getJointByNumber(-1).mass*Vec2d(-10,0))

    def destroyModel(self):
        for i in self.objects['rod']:
            self.space.remove(*i)
        for i in self.objects['top']:
            self.space.remove(i)
        for i in self.objects['pivots']:
            self.space.remove(i)
        # while n < len(self.joints):
        #     self.space.remove(self.joints[n])
        #     n+=1
        # while m < len(self.pivots):
        #     self.space.remove(self.joints[m])
        #     m+=1
        
        # for i in self.objects['rod']:
        #     for j in i:
        #         self.space.remove(j)
            # self.space.remove(i)
        # self.space.remove(self.objects)
        # self.objects.pop('rod')
        # self.objects.pop('pivots')
    
    def applyimpulse(self, direction):
        if direction != self.prev_dir and direction != 2: self.num_reversals += 1
        self.prev_dir = direction
        if direction == 0: #down
            if self.movable:
                self.getJointByNumber(-1).apply_impulse_at_local_point(self.getJointByNumber(-1).mass*Vec2d(10,0))
        elif direction == 1:
            if self.movable:
                self.getJointByNumber(-1).apply_impulse_at_local_point(self.getJointByNumber(-1).mass*Vec2d(-10,0))
        else:
            pass
        
        
    def step(self, action):
        self.action_step += 1
        self.prev_angle = self.angle()
        
        amplitude = abs(self.angle())
        if amplitude > self.maximum_amplitude:
            self.maximum_amplitude = amplitude
        self.applyimpulse(np.argmax(action))
        
        if self.action_step >= self.num_actions:
            self.done = True
        if self.seat.position.y < self.config['pivotPosition'][1]:
            self.done = True
            self.complete = True
    
    def calculateFitness(self):
        if self.complete:
            return (self.maximum_amplitude-self.initial_amplitude)+200/self.num_reversals
        else:
            return (self.maximum_amplitude-self.initial_amplitude)
        
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
        
        legs_shape = pymunk.Segment(legs, (20, 0), (20, 20), 3)
        
        knee_joint = pymunk.PinJoint(legs, body, (20, 0), (20,0))
        knee_motor = pymunk.SimpleMotor(legs, body, 0)
        knee_joint.collide_bodies = False
        
        self.space.add(body, torso_shape, bottom_shape, legs, legs_shape, knee_joint, knee_motor)
        
        return {'body' : [(body, torso_shape, legs_shape), (legs, legs_shape)], 'pivots' : [knee_joint, knee_motor]}

    def update(self):
        self.limitRotation()
    
    def limitRotation(self, limits=(np.pi/4, -np.pi/2)):
        # prevents legs from rotating too far, +pi/4 is 45deg clockwise, -pi/2 is 90 deg anticlockwise
        if self.legs[0].angle > limits[0]:
            self.legs[0].angle = limits[0]
        elif self.legs[0].angle < limits[1]:
            self.legs[0].angle = limits[1]

class SittingSwinger(Swing):
    
    def __init__(self, space, configFile):
        self.person = Person(space, self.swing.pos, mass=configFile['person'])
        self.seat = pymunk.PinJoint(self.swing.seat, self.person.objects['body'][0][0])
        self.seat.collide_bodies = False
        self.space.add(self.seat)
    
    def eventHandler(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.person.knee_motor.rate = -np.pi
        elif keys[pygame.K_DOWN]:
            self.person.knee_motor.rate = np.pi
    
    # def generateModel(self):
    #     build the model
    
    
