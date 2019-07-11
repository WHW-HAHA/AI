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
Project 1
'''
from Project1 import project1
Part1 = project1(labels= labels, reviews=reviews)
Part1.pretty_printer_review_and_label(10)
Part1.Validation()

###
# for word, count in Part1.positive_counts.most_common()[:10]:
#     print(word,':', count, '\n')
# for word, count in Part1.negative_counts.most_common()[:10]:
#     print(word,':', count, '\n')
# The top words in the most common list are 'the', 'are' ..., which won't help us

Part1.picking_valuable_words(many = 100)
print('Top Positive:', Part1.pos_neg_ratios.most_common()[:20])
print('Top Negative', list(reversed(Part1.pos_neg_ratios.most_common()))[:20])