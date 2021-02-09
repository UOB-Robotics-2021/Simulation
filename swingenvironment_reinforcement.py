import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import json
import random

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

    def __init__(self):
        pass

    def generatePerson(self):
        torso = pymunk.Body(10, 10)

class Swing():

    def __init__(self, space, swingConfig, numActions=1200):
        self.space = space
        self.swingConfig = swingConfig
        self.objects = self.generateSwing(swingConfig)
        self.seat = self.getJointByNumber(-1)

        self.actionStep = 0
        self.actions = self.generateActions(numActions)
        self.cont = True

        self.prev_y = 450
        self.maximum_y = 450
        self.steps_to_max = 500
        self.steps_above_prev_max = 0


    def generateSwing(self, config):
        # specifies the top of the swing as defined by topPosition
        top = pymunk.Body(10,1000000, pymunk.Body.STATIC)
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

    def destruct(self):
        for i in self.objects['rod']:
            self.space.remove(*i)
        for i in self.objects['pivots']:
            self.space.remove(i)
        self.space.remove(*self.objects['top'])

    def generateActions(self, numActions):
        actionList = []
        for _ in range(numActions):
            actionList.append(random.randrange(-1, 2))
        return actionList

    def step(self, action):
        if self.cont:
            self.applyImpulse(action)
            self.actionStep += 1
        if self.seat.position.y < self.maximum_y:
            self.maximum_y = self.seat.position.y
            self.steps_to_max = self.actionStep
        if self.seat.position.y < self.prev_y:
            self.steps_above_prev_max += 1

    def getJointByNumber(self, num):
        return self.objects['rod'][num][0]

    def applyImpulse(self, magnitude):
        self.seat.apply_impulse_at_local_point(Vec2d(magnitude, 0))

    def render(self, screen):
        pass

    def update(self):
        if self.actionStep + 1 == len(self.actions):
            self.cont = False
            self.destruct()
        else:
            self.step(100*self.actions[self.actionStep])

    def eventListener(self):
        pass

    def calculateFitness(self):
        return np.sqrt(1.0/self.maximum_y)**2+self.steps_above_prev_max

    def mutate(self, actions, mutationRate=0.07):
        newactions = []
        for i in self.actions:
            rand = np.random.rand()
            if rand < mutationRate:
                newactions.append(random.randrange(-1, 2))
            else:
                newactions.append(i)
        return newactions

    def reproduce(self):
        child = Swing(self.space, self.swingConfig)
        child.actionStep = 0
        child.actions = self.mutate(self.actions)
        child.steps_above_prev_max = 0
        child.prev_y = self.maximum_y
        return child

class Population():

    def __init__(self, populationSize, configFile, space):
        self.configFile = configFile
        self.space = space
        self.populationSize = populationSize
        self.population = self.generatePopulation(populationSize)

        self.generation = 1

    def generatePopulation(self, size):
        population = []
        for _ in range(size):
            population.append(Swing(self.space, self.configFile))
        return population

    def update(self):
        if self.checkAllFinished():
            self.naturalSelection()
        for i in self.population:
            i.update()

    def selectParent(self):
        fitnessSum = np.sum([i.calculateFitness() for i in self.population])

        randNum = random.uniform(0, fitnessSum)
        runningSum = 0
        for i in self.population:
            runningSum += i.calculateFitness()
            if runningSum > randNum:
                return i

    def naturalSelection(self):
        nextGen = []
        for _ in self.population:
            parent = self.selectParent()
            nextGen.append(parent.reproduce())

        self.population = nextGen
        self.generation += 1

    def checkAllFinished(self):
        count = 0
        for i in self.population:
            if i.cont:
                continue
            else:
                count += 1
        if count < self.populationSize:
            return False
        else:
            return True

class LearningArea():

    def __init__(self, configFile):
        pass

config = loadConfig('config_reinforcement.json')

#swing = Swing(space, config['swingConfig'])
pop = Population(50, config['swingConfig'], space)


running = True
while running:
    pygame.display.set_caption(f'Generation: {pop.generation}')
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        print("applying impulse to +ve x at the final joint")
        #swing.getJointByNumber(-1).apply_impulse_at_local_point(swing.getJointByNumber(-1).mass*Vec2d(10,0))
    elif keys[pygame.K_DOWN]:
        print("applying impulse to -ve x at the final joint")
        #swing.getJointByNumber(-1).apply_impulse_at_local_point(swing.getJointByNumber(-1).mass*Vec2d(-10,0))

    space.step(1/60)
    pop.update()
    screen.fill((255,255,255))
    space.debug_draw(draw_options)
    pygame.display.flip()

    clock.tick(60)
