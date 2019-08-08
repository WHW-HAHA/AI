'''
Hanwei Wang
Hanwei_Wang_94@outlook.com
'''
with open('labels.txt', 'r') as label_content:
    labels = label_content.read().upper().split('\n')
    label_content.close()

with open('reviews.txt', 'r') as review_content:
    reviews = review_content.read().lower().split('\n')
    review_content.close()


'''
Step 1
'''
from Project1 import project1
Step1 = project1(labels= labels, reviews=reviews)
Step1.pretty_printer_review_and_label(10)
Step1.Validation()

###
# for word, count in Part1.positive_counts.most_common()[:10]:
#     print(word,':', count, '\n')
# for word, count in Part1.negative_counts.most_common()[:10]:
#     print(word,':', count, '\n')
# The top words in the most common list are 'the', 'are' ..., which won't help us

Step1.picking_valuable_words(many = 100)
print('Top Positive:', Step1.pos_neg_ratios.most_common()[:20])
print('Top Negative', list(reversed(Step1.pos_neg_ratios.most_common()))[:20])

'''
Step 2: convert words in review and label to numerical data 
'''

from Project2 import project2
Step2 = project2(total_counts=Step1.total_counts)

'''
Step 3: Construct neural network
'''

from Nural_Network import SentimentNetwork
Network = SentimentNetwork(reviews = reviews, review_vocab=Step2.vocab, labels = labels)