# Hanwei Wang 21-06-2019

import numpy as np
import math

class NeturalNetwork():
    def __init__(self, x, y):
        self.input = x
        self.weight1_array = np.random.rand(self.input.shape[1], 4)  # 4 cells in mid layer
        self.weight2_array = np.random.rand(4,1) # row, column
        self.y = y
        self.output = np.zeros(y.shape)

    def feedForward(self):
        # forward feed function
        self.layer1_array = self.sigmoid(np.dot(self.input, self.weight1_array))
        self.output_array = self.sigmoid(np.dot(self.layer1, self.weight2_array))

    def backPropgation(self):
        # backward propgation function
        # application of the chain rule to find the derivative of the loss function with respect to the weight layer 1 and 2
        self.


    def sigmoid(self):
        pass





