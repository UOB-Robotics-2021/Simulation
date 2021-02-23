# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:10:17 2021

@author: Kieron
"""

import numpy as np

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
        self.layers = mutated_layers
        self.bias = mutated_bias