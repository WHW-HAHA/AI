# Hanwei Wang 21-06-2019
# start with building neuron with 2 inputs, 1 weight and a activation function

import numpy as np
import matplotlib.pyplot as plt

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
        return o1

    def train(self, data, all_y_trues):
        Eproch = []
        Loss = []
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

                # calculate the partial devatives
                # d_var1_d_var2 is partial var1 / partial var2
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * dev_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * dev_sigmoid(sum_o1)
                d_ypred_d_b3 = dev_sigmoid(sum_o1)

                d_ypred_d_h1 = self.weights[4] * dev_sigmoid(sum_o1)
                d_ypred_d_h2 = self.weights[5] * dev_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * dev_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * dev_sigmoid(sum_h1)
                d_h1_d_b1 = dev_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * dev_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * dev_sigmoid(sum_h2)
                d_h2_d_b2 = dev_sigmoid(sum_h2)

                # updates weights and biases
                # gradient decreasing method
                # Neuron h1
                self.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.biases[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.weights[3] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.biases[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.weights[4] -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.weights[5] -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.biases[2] -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                if eproch % 10 == 0: # mod 1000 ==> 100
                    y_preds = np.apply_along_axis(self.feedForward, 1, data) # along y
                    loss = mse_loss(all_y_trues, y_preds)
                    Eproch.append(eproch)
                    Loss.append(loss)
                    print("Epoch %d loss: %.3f"%(eproch, loss))
        plt.plot(Eproch, Loss)
        plt.xlabel('IterNo')
        plt.ylabel('Loss')
        plt.title('Loss vs Iteration')
        plt.show()



# <--- instantiation --->

data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, 6],
])

all_y_true = np.array([
    1,
    0,
    0,
    1,
])

network = neuralNetwork()
network.train(data, all_y_true)






