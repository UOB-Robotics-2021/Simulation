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
from models import Swing
from neuralnetwork import produceChildNetwork, saveNN
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
        self.champion_nn = None
        
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
                f'Previous Avg. Fitness: {round(self.prev_average_fitness, 4)}', f'Prev Champion Fitness: {round(self.prev_champ_fitness, 4)}', f'Num. Steps: {self.num_actions}',
                f'Num. Done: {len([i for i in self.population if i.done])}/{self.population_size}']
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
        avg_fitness = np.mean([i.calculateFitness() for i in self.population])
        
        parents, champion = self.bestPerformers(self.config['selectionPercentile'])
        champion_nn = copy.deepcopy(champion.neuralnetwork)
        self.champion_nn = champion_nn
        
        # every 10 generations, allow the pendulums to have more time to swing. This encourages
        # rapid generation of suitable individuals early on
        if self.generation % self.config['gensBeforeIncrement'] == 0:
            self.num_actions += self.config['numStepIncrement']

        if self.generation < self.config['gensBeforeNatural']:
            # allow ~75% of the next population to be children of current models
            for _ in range(int(self.population_size // (4/3))):
                child = self.model(self.space, self.config, self.num_actions)
                parents = self.selectParents(parents)
                parent_one_colour = np.array([i for i in parents[0].colour]) # tracking colour to show the 'sexual reproduction' method is working
                parent_two_colour = np.array([i for i in parents[1].colour])
                parentnets = [i.neuralnetwork for i in parents]
                child.neuralnetwork = produceChildNetwork(*parentnets)
                child.neuralnetwork.mutate(self.config['mutationRate'])
                child.colour = (parent_one_colour+parent_two_colour)/2
                #child.circle.color = (parent_one_colour+parent_two_colour)/2
                nextGen.append(child)

            # the remaining ~25% are new random specimens to encourage variation in the population
            for _ in range(self.population_size - len(nextGen)):
                child = self.model(self.space, self.config, self.num_actions)
                nextGen.append(child)
        else:
            for _ in range(self.population_size):
                child = self.model(self.space, self.config, self.num_actions)
                parents = self.selectParents(parents)
                parent_one_colour = np.array([i for i in parents[0].colour])
                parent_two_colour = np.array([i for i in parents[1].colour])
                parentnets = [i.neuralnetwork for i in parents]
                child.neuralnetwork = produceChildNetwork(*parentnets)
                child.neuralnetwork.mutate(self.config['mutationRate'])
                child.colour = (parent_one_colour+parent_two_colour)/2
                #child.circle.color = (parent_one_colour+parent_two_colour)/2
                nextGen.append(child)

        # ensure the champion is always carried through; stops the population getting worse
        nextGen[-1].neuralnetwork = champion_nn
        #nextGen[-1].circle.color = (255,0,0,255)
        self.destroyPopulation()

        self.prev_average_fitness = avg_fitness
        self.prev_champ_fitness = champion.calculateFitness()
        self.population = nextGen
        self.generation += 1
        return nextGen
    
    def selectParents(self, list_of_parents):
        # selects two parents weighted by their fitness; may be better to stop double selection of the same parent
        parent_fitness = [i.calculateFitness() for i in list_of_parents]
        parents = random.choices(list_of_parents, weights=parent_fitness, k=2)
        return parents
    
    def bestPerformers(self, percentile=0.05):
        fitnesses = [i.calculateFitness() for i in self.population]
        indices = np.argsort(fitnesses) # sorts the fitnesses and returns their ordered indices
        top_percentile = indices[-int(percentile*self.population_size):]

        parents = [self.population[i] for i in top_percentile]

        champion = parents[0]
        for i in parents:
            if i.calculateFitness() > champion.calculateFitness():
                champion = i
        
        return parents, champion
    
    def destroyPopulation(self):
        for i in self.population:
            i.destroyModel()

pygame.init()
screen_size = 600, 500
screen = pygame.display.set_mode(screen_size)
draw_options = pymunk.pygame_util.DrawOptions(screen)
running = True
clock = pygame.time.Clock()
# le = LearningEnvironment(space, VariablePendulum, loadConfig('config.json')['variablePendulumMLConfig'])
le = LearningEnvironment(space, Swing, loadConfig('config.json')['swingConfig'])
pygame.display.set_caption('Machine Learning')

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
