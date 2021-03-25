# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:10:17 2021

@author: Kieron
"""

import numpy as np
import os
import copy


def sigmoid(layer):
    return 1/(1+np.exp(-layer))


def softmax(layer):
    exponential = np.exp(layer)
    return exponential/exponential.sum()


class NeuralNetwork():

    def __init__(self, observation_size, action_size, n_layers, n_neurons):
        self.layers = []
        self.bias = []
        self.outputs = np.random.rand(action_size, n_neurons)

        for i in range(n_layers):
            entry_size = n_neurons if i != 0 else observation_size
            weight = np.random.rand(n_neurons, entry_size)*2 - 1
            self.layers.append(weight)
            self.bias.append(np.random.rand(n_neurons, 1)*2-1)

    def forward(self, inputs):
        inputs = np.array(inputs).reshape((-1, 1))

        for layer, bias in zip(self.layers, self.bias):
            inputs = np.matmul(layer, inputs)
            inputs += bias
            inputs = sigmoid(inputs)

        output_layer = np.matmul(self.outputs, inputs)
        output_layer = output_layer.reshape(-1)
        return softmax(output_layer)

    def mutate(self, mutationRate=0.05):
        mutated_layers = []
        mutated_bias = []
        for layer in self.layers:
            mutation_probability_matrix = np.random.rand(layer.shape[0], layer.shape[1])
            mutations = np.where(mutation_probability_matrix < mutationRate, (np.random.rand()-0.5)/2, 0)

            mutated_layers.append(layer + mutations)
        for bias in self.bias:
            mutation_probability_matrix = np.random.rand(bias.shape[0], bias.shape[1])
            mutations = np.where(mutation_probability_matrix < mutationRate, (np.random.rand()-0.5)/2, 0)

            mutated_bias.append(bias + mutations)

        mutation_probability_matrix = np.random.rand(self.outputs.shape[0], self.outputs.shape[1])
        mutations = np.where(mutation_probability_matrix < mutationRate, (np.random.rand()-0.5)/2, 0)

        self.outputs += mutations
        self.layers = mutated_layers
        self.bias = mutated_bias


def produceChildNetwork(parent_one, parent_two, method='mean'):
    parent_one = copy.deepcopy(parent_one)
    parent_one = copy.deepcopy(parent_two)
    layers = []
    bias = []
    outputs = []
    if method == 'mean':
        for i, j in zip(parent_one.layers, parent_two.layers):
            layers.append(i/2+j/2)
        for i, j in zip(parent_one.bias, parent_two.bias):
            bias.append(i/2+j/2)
        outputs = parent_one.outputs/2 + parent_two.outputs/2

        neuralnetwork = NeuralNetwork(1, 1, 1, 1)
        neuralnetwork.layers = layers
        neuralnetwork.bias = bias
        neuralnetwork.outputs = outputs

        return [neuralnetwork]

    elif method == 'uniform':
        layers2 = []
        bias2 = []
        outputs2 = []
        for i, j in zip(parent_one.layers, parent_two.layers):
            if np.random.randint(0, 2) == 0:
                layers.append(i)
                layers2.append(j)
            else:
                layers.append(j)
                layers2.append(i)
        for i, j in zip(parent_one.bias, parent_two.bias):
            if np.random.randint(0, 2) == 0:
                bias.append(i)
                bias2.append(j)
            else:
                bias.append(j)
                bias2.append(j)
        for i, j in zip(parent_one.outputs, parent_two.outputs):
            if np.random.randint(0, 2) == 0:
                outputs.append(i)
                outputs2.append(j)
            else:
                outputs.append(j)
                outputs2.append(i)
        outputs = np.array(outputs)
        outputs2 = np.array(outputs)

        neuralnetwork = NeuralNetwork(1, 1, 1, 1)
        neuralnetwork.layers = layers
        neuralnetwork.bias = bias
        neuralnetwork.outputs = outputs
        neuralnetwork2 = NeuralNetwork(1, 1, 1, 1)
        neuralnetwork2.layers = layers2
        neuralnetwork2.bias = bias2
        neuralnetwork2.outputs = outputs2

        return [neuralnetwork, neuralnetwork2]


def loadNN(nnName):
    layerFiles = os.listdir(f'{nnName}\\layers')
    biasFiles = os.listdir(f'{nnName}\\bias')

    layers = []
    bias = []
    output = np.load(f'{nnName}\\layers\\{layerFiles[-1]}')
    for i in range(len(layerFiles)-1):
        layers.append(np.load(os.path.join(nnName, f'layers\\{i}.npy')))
    for j in range(len(biasFiles)):
        bias.append(np.load(os.path.join(nnName, f'bias\\{j}.npy')))

    neuralnetwork = NeuralNetwork(1, 1, 1, 1)
    neuralnetwork.layers = layers
    neuralnetwork.bias = bias
    neuralnetwork.outputs = output
    return neuralnetwork


def saveNN(neuralnetwork, nnName):
    neuralnetwork = copy.deepcopy(neuralnetwork)
    cnt = 0
    for i in neuralnetwork.layers:
        np.save(f'{nnName}\\layers\\{cnt}', i)
        cnt += 1
    np.save(f'{nnName}\\layers\\{cnt}', neuralnetwork.outputs)
    cnt = 0
    for i in neuralnetwork.bias:
        np.save(f'{nnName}\\bias\\{cnt}', i)
        cnt += 1
