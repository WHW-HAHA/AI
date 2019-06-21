# Hanwei Wang 21-06-2019
# start with building neuron with 2 inputs, 1 weight and a activation function

import numpy as np

def sigmoid(x): # external function, not in body of class
    # f(x) = 1/(1 + e^(-x))
    return 1/(1 + np.exp(-x))

def dev_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)

def mse_loss(y_true, y_pred):
    return ((y_true- y_pred)**2).mean()


class Neuron():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        # weight inputs, add bias, then use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class neuralNetwork():
    '''
    Just for understand, should use matrix
    '''
    def __init__(self):
        # weights
        self.weights = np.random.rand(6,1)

        # biases
        self.biases =  np.random.rand(3,1)

        # Neuron
        # self.h1 = Neuron(weights, bias)
        # self.h2 = Neuron(weights, bias)
        # self.o1 = Neuron(weights, bias)

    def feedForward(self, x):
        h1 = sigmoid(self.weights[0] * x[0] + self.weights[1] * x[1] + self.biases[0])
        h2 = sigmoid(self.weights[2] * x[0] + self.weights[3] * x[1] + self.biases[1])
        o1 = sigmoid(self.weights[4] * h1 +  self.weights[5] * h2 + self.biases[2])

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        eprochs = 1000 # number of times to loop through the entire dataset

        for eproch in range(eprochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.weights[0] * x[0] + self.weights[1] * x[1] + self.biases[0]
                h1 = sigmoid(sum_h1)

                sum_h2 = self.weights[2] * x[0] + self.weights[3] * x[1] + self.biases[1]
                h2 = sigmoid(sum_h2)

                sum_o1 = self.weights[4] * h1 + self.weights[5] * h2 + self.biases[2]
                o1 = sigmoid(sum_o1)

                y_pred = o1
                d_L_d_ypred = -2 * (y_true - y_pred)





# <--- instantiation --->
netWork = neuralNetwork()
x = np.array([2, 3]) # inputs
print(netWork.feedForward(x), 'in between 0 and 1')






