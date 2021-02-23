# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:10:11 2021

@author: Kieron
"""

import pygame
import numpy as np
import pymunk
import pymunk.pygame_util
import json
import random
import copy
from models import VariablePendulum
from neuralnetwork import NeuralNetwork
from datetime import datetime

space = pymunk.Space()
space.gravity = [0, 900]

def loadConfig(configFile):

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)

class LearningEnvironment():
    
    def __init__(self, space, model, configFile):
        self.space = space
        self.model = model
        self.config = configFile
        self.population_size = self.config['populationSize']
        self.num_actions = self.config['initialNumSteps']
        self.inittime = datetime.now()

        self.population = self.generatePopulation(self.population_size)
        self.generation = 1
        self.prev_average_fitness = 0
        self.prev_champ_fitness = 0
        
        self.font = pygame.font.SysFont('Arial', 20)

    def update(self):
        count = 0
        for i in self.population:
            if i.done:
                count += 1
            i.update()
        if count == len(self.population):
            self.generateNextPopulation()

    def render(self, screen):
        text = [f'Time Since Start: {(datetime.now()-self.inittime)}', f'Generation: {self.generation}', f'Population Size: {len(self.population)}', 
                f'Previous Avg. Fitness: {round(self.prev_average_fitness, 4)}', f'Prev Champion Fitness: {round(self.prev_champ_fitness, 4)}', f'Num. Steps: {self.num_actions}']
        cnt = 0
        for i in text:
            render = self.font.render(i, True, (0,0,0))
            screen.blit(render, (0, cnt*20))
            cnt+=1
    
    def generatePopulation(self, populationSize):
        pop = []
        for _ in range(populationSize):
            pop.append(self.model(self.space, self.config, self.num_actions))
        return pop
    
    def generateNextPopulation(self):
        nextGen = []
        running_sum = 0
        
        # find the individual with the highest fitness score and replicate their NN
        champion = self.population[0]
        for i in self.population:
            running_sum += i.calculateFitness()
            if i.calculateFitness() > champion.calculateFitness():
                champion = i
        champion = copy.deepcopy(champion)
        champ_nn = champion.neuralnetwork
        
        champ_fitness = champion.calculateFitness()
        avg_fitness = running_sum/self.population_size
        
        # every 10 generations, allow the pendulums to have more time to swing. This encourages
        # rapid generation of suitable individuals early on
        if self.generation % 10 == 0:
            self.num_actions += self.config['numStepIncrement']

        # allow ~75% of the next population to be children of current models
        for _ in range(int(self.population_size // (4/3))):
            child = self.model(self.space, self.config, self.num_actions)
            parent_nn = self.selectParent()
            child.neuralnetwork = parent_nn
            child.neuralnetwork.mutate(self.config['mutationRate'])
            nextGen.append(child)

        # the remaining ~25% are new random specimens to encourage variation in the population
        for _ in range(self.population_size - len(nextGen)):
            child = self.model(self.space, self.config, self.num_actions)
            child.circle.colour = (255,0,0,255)
            nextGen.append(child)

        for i in self.population:
            i.destroyModel()
        champion.destroyModel()

        nextGen[0].neuralnetwork = champ_nn # assign the champion neural network, not mutated, so that the strongest individual continues to the next generation

        self.prev_average_fitness = avg_fitness
        self.prev_champ_fitness = champ_fitness
        self.population = nextGen
        self.generation += 1
        return nextGen
    
    def selectParent(self):
        # demands that the probability of reproducing is proportional to an individual's fitness
        fitnessSum = np.sum([i.calculateFitness() for i in self.population])

        randNum = random.uniform(0, fitnessSum)
        runningSum = 0
        for i in self.population:
            runningSum += i.calculateFitness()
            if runningSum > randNum:
                return i.neuralnetwork
    
    def destroyPopulation(self):
        for i in self.population:
            i.destroyModel()

pygame.init()
screen_size = 600, 500
screen = pygame.display.set_mode(screen_size)
draw_options = pymunk.pygame_util.DrawOptions(screen)
running = True
clock = pygame.time.Clock()
le = LearningEnvironment(space, VariablePendulum, loadConfig('config.json')['variablePendulumMLConfig'])
pygame.display.set_caption(f'Machine Learning')

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    clock.tick(30)
    screen.fill((255,255,255))
    le.update()
    le.render(screen)
    space.debug_draw(draw_options)
    pygame.display.flip()
    
    space.step(1/30)