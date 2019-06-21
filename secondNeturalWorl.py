# Hanwei Wang 21-06-2019
# start with building neuron with 2 inputs, 1 weight and a activation function

import numpy as np

def sigmod(x): # external function, not in body of class
    # f(x) = 1/(1 + e^(-x))
    return 1/(1 + np.exp(-x))

class Neuron():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        # weight inputs, add bias, then use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmod(total)

weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])
print(n.feedForward(x), 'in between 0 -- 1')


