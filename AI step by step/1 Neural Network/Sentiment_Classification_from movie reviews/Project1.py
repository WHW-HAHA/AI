'''
Hanwei Wang

Hanwei_Wang_94@outlook.com
'''
from collections import Counter
import numpy as np

class project1():
    def __init__(self, labels, reviews):
        self.labels = labels
        self.reviews = reviews

    def pretty_printer_review_and_label(self, i):
        print(self.labels[i] + "\t:\t" + self.reviews[i][:80] + "...")

    def Validation(self):
        self.positive_counts = Counter()
        self.negative_counts = Counter()
        self.total_counts = Counter()
        # Count the most popular words in the review
        for i in range(len(self.labels)):
            if self.labels[i] == 'POSITIVE':
                for word in self.reviews[i].split(' '):
                    self.positive_counts[word] += 1
                    self.total_counts[word] += 1
            else:
                for word in self.reviews[i].split(' '):
                    self.negative_counts[word] += 1
                    self.total_counts[word] += 1
        # pos_neg_ratios

    def picking_valuable_words(self, many):
        self.pos_neg_ratios = Counter()
        for term, cnt in self.total_counts.most_common():
            if (cnt > many):
                pos_neg_ratio = self.positive_counts[term]/float(self.negative_counts[term] + 1)
                self.pos_neg_ratios[term] = pos_neg_ratio

        for word, ratio in self.pos_neg_ratios.most_common():
            if (ratio >1):
                self.pos_neg_ratios[word] = np.log(ratio)
            else:
                self.pos_neg_ratios[word] = -np.log(1/(ratio + 0.01))











