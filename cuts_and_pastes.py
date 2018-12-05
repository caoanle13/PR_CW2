#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:47:35 2018

@author: mohyeldinaboualam
"""
#M_pca = 150
"""
validation_classes = []
y_history = []
# Making the validation set
import random
while len(validation_classes) < 100:
    y = random.randint(1,1467)
    if y in y_train and y not in y_history:
        validation_classes.append(y)
    y_history.append(y)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = M_pca)
X_train_pca = pca.fit_transform(X_train)
X_query_pca = pca.transform(X_query)
explained_variance = pca.explained_variance_ratio_

total_variance = 0
for var in explained_variance:
    total_variance += var

print(f"The data contribution carried by {M_pca} is {total_variance}")

"""
"""
correct = 0
wrong = 0
i = 0
accuracies = []
for x in temp_neighbors:
    if len(x) > 0:
        if np.sum(x)/len(x) == x[0]:
            correct += 1
        else:
            wrong += 1
    if (correct + wrong)!=0:
        accuracy_score = (correct/(correct + wrong))
        accuracies.append(accuracy_score)
    correct = 0
    wrong = 0
    i+= 1
accuracies = np.array(accuracies)
average_accuracy = np.average(accuracies)
average_error = 1 - average_accuracy


print(f"The average accuracy on the test set is {average_accuracy}\nThe average error is {average_error}")


correct = 0
wrong = 0
i = 0
accuracies = []
for x in neighbor_labels:
    for element in x:
        if element == y_query[i]:
            correct += 1
        else:
            wrong += 1
    accuracy_score = (correct/(correct + wrong))
    accuracies.append(accuracy_score)
    correct = 0
    wrong = 0
    i+= 1
accuracies = np.array(accuracies)
average_accuracy = np.average(accuracies)
average_error = 1 - average_accuracy

print(f"The average accuracy on the test set is {average_accuracy}\nThe average error is {average_error}")




# A finction which takes in the list of nearest k neighbors and calculates the rank accuracy (based on Q&A defintion)
def rank_calculation(list_of_neighbors, y_query):
    correct = 0
    i = 0
    for x in list_of_neighbors:
        if len(x) > 0 and y_query[i] in x:
            correct += 1
        i += 1
    average_accuracy = correct/len(y_query)
    average_error = 1 - average_accuracy
    return average_accuracy, average_error
"""

"""
i = 0
temp_neighbors = []
delete_idx = []
for neighbors in neighbor_labels:
    j = 0
    for label in neighbors:
         if label == y_query[i]:
            if camId[query_idx[i]] == camId[gallery_idx[[list_of_indices[i][j]]]]:
                print(list_of_indices[i][j])
                print("Both conditions satisfied")
                delete_idx.append(j)
                
         j += 1
    temp_row = np.delete(neighbors, delete_idx)
    delete_idx = []
    temp_neighbors.append(temp_row)
    i += 1

temp_neighbors = np.array(temp_neighbors)
"""