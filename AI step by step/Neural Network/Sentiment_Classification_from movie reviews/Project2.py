'''
Hanwei Wang

Hanwei_Wang_94@outlook.com
'''
import numpy as np
from collections import Counter

class projec2():

    def __init__(self, total_counts):
        if isinstance(total_counts, dict):
            self.vocab = total_counts.keys()
            self.voc_len = len(self.vocab)
            self.word2index = {}
            for i, word in enumerate(self.vocab):
                self.word2index[word] = i
            self.input_counts = Counter()
            self.output_counts = Counter()
            self.layer0 = np.zeros((1, self.voc_len))

    def update_input_layer(self, review):
        self.layer0 *= 0
        for word in review.split(' '):
            self.layer0[0][self.word2index[word]] += 1 # assign first rwo here

    def get_target_for_label(self, label):
        if label == 'POSITIVE':
            return 1
        else:
            return 0
        # layer1 = np.zeros((1, self.voc_len))


