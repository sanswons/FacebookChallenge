from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from Cleaning import *
from sklearn.model_selection import GridSearchCV

# Split Data
msk = np.random.rand(len(posts)) < 0.8
train_data = posts[msk]
test_data = posts[~msk]
total_data_length = len(train_data) + len(test_data)

#Alternate feature vector representations
'''
# create word vectors
vectorizer = CountVectorizer()
train_matrix = vectorizer.fit_transform(train_data['Clean Description'])
test_matrix = vectorizer.transform(test_data['Clean Description'])

# Convert the meta features to a numpy array.
print('Adding more features')
no_of_likes = np.asarray(posts['has_likes'])
no_of_shares = np.asarray(posts['has_shares'])
has_picture_labels = np.asarray(posts['has_picture_label'])

# Concatenate the features together.
train_matrix = np.column_stack([has_picture_labels[:len(train_data)], no_of_shares[:len(train_data)],
                                no_of_likes[:len(train_data)], train_matrix.todense()])
test_matrix = np.column_stack([has_picture_labels[len(train_data):], no_of_shares[len(train_data):],
                               no_of_likes[len(train_data):], test_matrix.todense()])

'''

# print(train_matrix)
