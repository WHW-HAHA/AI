'''
Hanwei Wang
Hanwei_Wang_94@outlook.com
'''

import time
import sys
import numpy as np

class SentimentNetwork():
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        np.random.seed(1)
        self.pre_process