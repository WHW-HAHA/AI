'''
Hanwei Wang
Hanwei_Wang_94@outlook.com
'''

import time
import sys
import numpy as np

class SentimentNetwork():
    def __init__(self, reviews, review_vocab, labels, hidden_nodes = 10, learning_rate = 0.1):
        np.random.seed(1)
        self.pre_process_data(reviews, labels)
        self.review_vocab = review_vocab
        self.init_network(input_nodes= len(self.review_vocab), hidden_nodes= hidden_nodes, output_nodes=1, learning_rate = learning_rate)

    def pre_process_data(self, labels):
        review_vocab = set(self.review_vocab)
        self.review_vocab = list(review_vocab)
        label_vocab = set()
        self.lanbel_vocab = list(self.lanbel_vocab)
        self.review_vocab_size = len(self.review_vocab)
        self.lanbel_vocab_size = len(self.lanbel_vocab)



    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        pass
