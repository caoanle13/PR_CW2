# 13,164 images in data set
# 1,360 pedestrians

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.io import loadmat
camId = loadmat('cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
filelist = loadmat('cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()
gallery_idx = loadmat('cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
labels = loadmat('cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
train_idx = loadmat('cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
query_idx = loadmat('cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()

train_idx -= 1
query_idx -= 1


import json
with open('feature_data.json', 'r') as f:
    features = np.array(json.load(f))

X_train = features[train_idx]
y_train = labels[train_idx]
X_query = features[query_idx]
y_query = labels[query_idx]
X_gallery =  features[gallery_idx]
y_gallery = labels[gallery_idx]


"""A function which takes parameters:
neighbors_labels: An np array of the classes of the neighbors for each query point (rows)
y_query: The classes of the query points  

It returns the rank accuracy (based on Q&A defintion: If one neighbor has the same class
as the query point then it is given a 1, it is given a 0. These are then summed up across 
all the query points and then divided by the number of query points to get the percentage 
of query points that were represented correctly. 
 
of 
"""
def rank_calculation(neighbors_labels, y_query, rank):
    correct = 0
    i = 0
    for x in neighbors_labels:
        if len(x) > 0 and y_query[i] in x[0:rank]:
            correct += 1
        i += 1
    average_accuracy = correct/len(y_query)
    average_error = 1 - average_accuracy
    return average_accuracy, average_error

"""
NN_for_testing is a function which takes parameters:
neighbors_labels: An np array of the classes of the neighbors for each query point (rows) 
list_of_indices: A list of the indices of the neighbors of each query point (rows)
y_query: The classes of the query points 
k: The number of nearest neighbors (k) used with the k-NN algorithm

It return a list of the classes of the nearest neighbors who have the same class
as the query points AND are taken by different camera angles to the query points.
This is needed for the testing phase (see training instructions pdf)
"""
def NN_for_testing(neighbors_labels, list_of_indices, y_query, k):
    i = 0
    true_neighbors = []
    delete_idx = []
    if k == 1:
        for label in neighbors_labels:
            if label == y_query[i]:
                    if camId[query_idx[i]] == camId[gallery_idx[[list_of_indices[i]]]]:
                        true_neighbors.append(np.delete(label, i))
            i += 1
    
    else:
        for neighbors in neighbors_labels:
            j = 0
            for label in neighbors:
                if label == y_query[i]:
                    if camId[query_idx[i]] == camId[gallery_idx[[list_of_indices[i][j]]]]:
                        delete_idx.append(j)                
                j += 1
            true_neighbors.append(np.delete(neighbors, delete_idx))
            delete_idx.clear()
            i += 1
    
    return true_neighbors


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier

k = 10 #Number of nearest neighbors 
rank = 1
classifier = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')

classifier.fit(X_gallery, y_gallery)

list_of_indices = classifier.kneighbors(X_query, return_distance = False) #This used to be X_query as the first parametewr

neighbors_labels = y_gallery[list_of_indices]

true_neighbors = NN_for_testing(neighbors_labels, list_of_indices, y_query, k)

accuracy, error = rank_calculation(true_neighbors, y_query, rank)

print(f"The average accuracy on the test set is {accuracy}\nThe average error is {error}")



    
    
    
    
    
    
    
    
    
    
    
    
    