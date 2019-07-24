'''
Hanwei Wang 23-7-2019
'''

import numpy as np
import pandas as pd
import matplotlib as plt
import unittest
import sys

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
        features = np.array(features, ndmin= 2)
        targets = np.array(targets, ndmin= 2)

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            hidden_inputs = np.dot(self.weights_input_to_hidden, X.T)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
            final_outputs = final_inputs

            error = y - final_outputs
            error_term = error

            hidden_error = np.dot(self.weights_hidden_to_output.T, error_term) # shape should be hidden_nodes x 1 --> 16 x 1 or 2 x 1
            hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs) # 2 x 1

            # delta_weights_h_o += np.dot(error_term, hidden_outputs.T)
            # delta_weights_i_h += np.dot(hidden_error_term, X)
            delta_weights_h_o += error_term * hidden_outputs.T  # shape should like self.w_h_o --> 1 x 2 or 1 x 16
            delta_weights_i_h += hidden_error_term * X        # shape should like self.w_i_h --> 2 x 3 or 16 x 56

        self.weights_hidden_to_output += self.lr * delta_weights_h_o/ n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/ n_records

    def run(self, features):
        features = np.array(features, ndmin=2)
        hidden_inputs = np.dot(self.weights_input_to_hidden, features.T)
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
        # self.assertTrue(np.allclose(network.weights_hidden_to_output, np.array([[0.37275328, -0.03172939]])))
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

# use MSE to the evaluate the performance of the net work
from NN import NeuralNetwork
def MSE(y, Y):
    return np.mean((y-Y)**2)

# Tune the model
iteration = 5000 # different rate for 0--2000, 2001-- 4000, 4001--5000
learning_rate = 0.1
hidden_nodes = 16
output_nodes = 1
N_i = train_features.shape[1]
network = NeuralNetwork(input_nodes=N_i, hidden_nodes = hidden_nodes, output_nodes = output_nodes, learning_rate = learning_rate)
losses = {'train':[], 'validation': []}

for e in range(iteration):
    if (e > 2000):
        network.lr = 0.05
    elif (e > 4000):
        network.lr = 0.01
    else:
        network.lr = 0.001

    # generate the batch by random selection from the dataset
    batch = np.random.choice(train_features.index, size = 128)
    for record, target in zip(train_features.ix[batch].values, train_targets.ix[batch]['cnt']):
        # use cnt as test targets
        train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
        sys.stdout.write("\rProgress: " + str(100 * e/ float(iteration))[:4]
                         + "% ... Training loss: " + str(train_loss)[:5]
                         + " ... Validation loss: " + str(val_loss)[:5])
        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

# visualization tool

plt.plot(losses['train'], label = 'Training loss')
plt.plot(losses['validation'], label = 'validation loss')
plt.legend()
_ = plt.ylim()


fig, ax = fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
































