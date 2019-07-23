'''
Hanwei Wang 23-7-2019
'''

import numpy as np
import pandas as pd
import matplotlib as plt
import unittest

data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)


# re-encode the data in one-hot format
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis = 1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis = 1)

# normalize the continuous features by mean and std
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

# splitting data to train and test data data set
test_data = data[-21*24: ]
train_data = data[ :-21*24]

# separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis = 1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis = 1), test_data[target_fields]

# hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]
print('Length input layer:', train_features.iloc[0, :].shape)
print('Length target layer:', train_targets.iloc[0, :].shape)

# class NeuralNetwork
class NeuralNetwork():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_to_hidden = np.random.normal(0, self.input_nodes**-0.5,
                                                        (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_to_output = np.random.normal(0, self.hidden_nodes**-0.5,
                                                         (self.output_nodes, self.hidden_nodes))

        self.lr = learning_rate
        self.activation_function = lambda x: 1/(1+np.exp(-x)) # sigmoid function

    def train(self, features, targets):
        '''
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''
        features = np.array(features, ndmin= 2).T
        targets = np.array(targets, ndmin= 2).T

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            hidden_inputs = np.dot(self.weights_input_to_hidden, X)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
            final_outputs = final_inputs

            error = y - final_outputs    # 3 * 1
            error_term = error

            hidden_error = np.dot(error_term.T, self.weights_hidden_to_output)
            hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)

            # delta_weights_h_o += np.dot(error_term, hidden_outputs.T)
            # delta_weights_i_h += np.dot(hidden_error_term, X )

            delta_weights_h_o += error_term *  hidden_outputs.T
            delta_weights_i_h += hidden_error_term * X

        self.weights_hidden_to_output += self.lr * delta_weights_h_o/ n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/ n_records

    def run(self, features):
        features = np.array(features, ndmin=2).T
        hidden_inputs = np.dot(self.weights_input_to_hidden, features)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs  = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs

        return final_outputs

class TestMethods(unittest.TestCase):

    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')

    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1 / (1 + np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        network.train(inputs, targets)
        print(network.weights_hidden_to_output)
        print(network.weights_input_to_hidden)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, np.array([[0.37275328, -0.03172939]])))
    #     self.assertTrue(np.allclose(network.weights_input_to_hidden,
    #                                 np.array([[0.10562014, 0.39775194, -0.29887597],
    #                                           [-0.20185996, 0.50074398, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

# Test inputs
inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3],
                       [-0.2, 0.5, 0.2]]) # shape 2x3
test_w_h_o = np.array([[0.3, -0.1]])      # shape 1x2


suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)



















