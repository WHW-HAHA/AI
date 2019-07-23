'''
Hanwei Wang 23-7-2019
'''

import numpy as np
import pandas as pd
import matplotlib as plt

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

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]
print('Length input layer:', train_features.iloc[0, :].shape)
print('Length target layer:', train_targets.iloc[0, :].shape)

#









