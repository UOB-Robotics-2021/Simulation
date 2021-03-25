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
import os
import pandas as pd
import matplotlib.pyplot as plt
from models import VariablePendulum
from stickman import Stickman
from neuralnetwork import produceChildNetwork, saveNN
from datetime import datetime

space = pymunk.Space()
space.gravity = [0, 900]
space.damping = 0.9


def loadConfig(configFile):

    with open(f'{configFile}', 'r') as cfg:
        config = cfg.read()

    return json.loads(config)


class LearningEnvironment():

    def __init__(self, model, configFile):
        self.space = space
        self.space.gravity = [0, 900]
        self.model = model
        self.config = configFile
        self.population_size = self.config['MLConfig']['populationSize']
        self.num_actions = self.config['MLConfig']['initialNumSteps']

        self.inittime = datetime.now()

        self.population = self.generatePopulation(self.population_size)
        self.generation = 1

        self.prev_average_fitness = 0
        self.prev_champ_fitness = 0

        self.champion_nn = None
        self.MLdata = [] # generation, champfitness, avgfitness
        self.energydata = [] # time, kinetic energy, potential energy
        self.phasedata = [] # theta, dtheta
        self.pumpdata = []

        self.font = pygame.font.SysFont('Arial', 20)

    def update(self):
        count = 0
        self.space.step(1/30)
        for i in self.population:
            if i.done:
                count += 1
            i.update()
        if count == len(self.population):
            self.generateNextPopulation()

    def render(self, screen):
        text = [f'Time Since Start: {(datetime.now()-self.inittime)}',
                f'Generation: {self.generation}',
                f'Population Size: {len(self.population)}',
                f'Previous Avg. Fitness: {round(self.prev_average_fitness, 4)}',
                f'Prev Champion Fitness: {round(self.prev_champ_fitness, 4)}',
                f'Num. Steps: {self.num_actions}',
                f'Num. Done: {len([i for i in self.population if i.done])}/{self.population_size}']
        cnt = 0
        for i in text:
            render = self.font.render(i, True, (0, 0, 0))
            screen.blit(render, (0, cnt*20))
            cnt += 1

    def generatePopulation(self, populationSize):
        pop = []
        for _ in range(populationSize):
            pop.append(self.model(self.space, self.config['ModelConfig'], self.num_actions))
        return pop

    def generateNextPopulation(self):
        nextGen = []
        best = self.bestPerformers(self.config['MLConfig']['selectionPercentile'])
        champion = self.population[0]
        for i in self.population:
            if i.calculateFitness() > champion.calculateFitness():
                champion = i
        champion_nn = copy.deepcopy(champion.neuralnetwork)
        self.champion_nn = champion_nn  # save this for extraction

        avg_fitness = np.mean([i.calculateFitness() for i in self.population if i.calculateFitness() != 0])

        if self.config['MLConfig']['randomIndividuals']:
            # allow ~75% of the next population to be children of current models
            while len(nextGen) < (int(self.population_size // 4/3)):
                for i in self.generateChild([i.neuralnetwork for i in self.selectParents(best, 2)]):
                    nextGen.append(i)

            # the remaining ~25% are new random specimens to encourage variation in the population
            for _ in range(self.population_size - len(nextGen)):
                child = self.model(self.space, self.config['ModelConfig'], self.num_actions)
                nextGen.append(child)

        else:
            while len(nextGen) < self.population_size:
                for i in self.generateChild([i.neuralnetwork for i in self.selectParents(best, 2)]):
                    nextGen.append(i)

        # ensure the champion is always carried through; stops the population getting worse
        if self.config['MLConfig']['elitistEnabled']:
            nextGen[-1].neuralnetwork = copy.deepcopy(champion_nn)

        self.MLdata.append((self.generation, champion.calculateFitness(), avg_fitness))
        self.energydata = champion.energydata
        self.phasedata = champion.phasedata
        self.pumpdata = champion.pumpdata
        self.misdata = [champion.num_reversals, champion.initial_amplitude, champion.maximum_amplitude]
        self.destroyPopulation()

        self.prev_average_fitness = avg_fitness
        self.prev_champ_fitness = champion.calculateFitness()
        self.population = nextGen
        self.generation += 1
        return nextGen

    def selectParents(self, list_of_parents, num=2):
        # selects two parents weighted by their fitness; may be better to stop double selection of the same parent
        parent_fitness = [i.calculateFitness() for i in list_of_parents]
        parents = random.choices(list_of_parents, weights=parent_fitness, k=num)
        return parents

    def bestPerformers(self, percentile=0.05):
        fitnesses = [i.calculateFitness() for i in self.population]
        indices = np.argsort(fitnesses)  # sorts the fitnesses and returns their ordered indices
        top_percentile = [i for i in indices[-int(percentile*self.population_size):] if i != 0]  # sorts indices in order  descending fitness

        parents = [self.population[i] for i in top_percentile]

        return parents

    def generateChild(self, parents):
        children = []
        if self.config['MLConfig']['crossoverMethod'] == 'mean':
            child = self.model(self.space, self.config['ModelConfig'], self.num_actions)
            child.neuralnetwork = produceChildNetwork(*parents, self.config['MLConfig']['crossoverMethod'])[0]
            child.neuralnetwork.mutate(self.config['MLConfig']['mutationRate'])
            children.append(child)

        elif self.config['MLConfig']['crossoverMethod'] == 'uniform':
            child = self.model(self.space, self.config['ModelConfig'], self.num_actions)
            child2 = self.model(self.space, self.config['ModelConfig'], self.num_actions)

            child.neuralnetwork = produceChildNetwork(*parents, self.config['MLConfig']['crossoverMethod'])[0]
            child.neuralnetwork.mutate(self.config['MLConfig']['mutationRate'])

            child2.neuralnetwork = produceChildNetwork(*parents, self.config['MLConfig']['crossoverMethod'])[1]
            child2.neuralnetwork.mutate(self.config['MLConfig']['mutationRate'])

            children.append(child)
            children.append(child2)

        elif self.config['MLConfig']['crossoverMethod'] == 'no':
            child = self.model(self.space, self.config['ModelConfig'], self.num_actions)
            child.neuralnetwork = copy.deepcopy(parents[0])
            child.neuralnetwork.mutate(self.config['MLConfig']['mutationRate'])
            children.append(child)
        return children

    def destroyPopulation(self):
        for i in self.population:
            i.destroyModel()


pygame.init()
screen_size = 600, 500
renderScreen = pygame.display.set_mode(screen_size)
draw_options = pymunk.pygame_util.DrawOptions(renderScreen)
running = True
clock = pygame.time.Clock()
le = LearningEnvironment(VariablePendulum, loadConfig('config.json'))
pygame.display.set_caption('Machine Learning')

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        le.destroyPopulation()
        le = LearningEnvironment(VariablePendulum, loadConfig('config.json')['variablePendulumMLConfig'])
    if le.generation > 50:
        running = False
    clock.tick(30)
    renderScreen.fill((255, 255, 255))
    le.update()
    le.render(renderScreen)
    le.space.debug_draw(draw_options)
    pygame.display.flip()

pygame.quit()
data = pd.DataFrame(le.MLdata, columns=['gen', 'champ', 'avg'])
data.to_csv(f'data/mean/{len(os.listdir("data/mean"))+1}.csv')

#saveNN(le.champion_nn, 'variablePendulum')
