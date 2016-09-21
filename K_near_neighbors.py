#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv('Breast_Cancer.txt')
df.replace('?', -99999, inplace=True)
#print(df.head())
df.drop(['id'], 1, inplace=True)
#print(df.head())

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

examples = np.array([[4,2,2,3,4,1,1,3,1], [8,5,3,10,5,3,6,12,4],[4,2,5,3,4,3,1,2,2]])
examples = examples.reshape(len(examples), -1)

predictions = clf.predict(examples)
print(predictions)

# here we are going to implement our own K-nearest neighbor algorithm
import warnings
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('fivethirtyeight')

# define our own K-nearest-neighbors
def K_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to the value that is less than the total voting groups!')
    distances = []
    for group in data:
        for feature in data[group]:
            # slow way, hard coded and only works for 2 dimensions
            #eucl_dist = sqrt( (feature[0] - predict[0])**2 + (feature[1] - predict[1])**2 )
            
            # intermediate way using numpy way
            eucl_dist = np.sqrt( np.sum(np.array(feature) - np.array(predict))**2 )
            
            # the fastest way using linear algebra from numpy
            eucl_dist = np.linalg.norm(np.array(feature) - np.array(predict))
            
            distances.append([eucl_dist, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    #print(votes)
    vote_result = Counter(votes).most_common(1)[0][0]
    #print(Counter(votes).most_common(1))
    return vote_result
    
dataset = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}
new_features = [5,7]

# the display should be used in notebook!!!
for i in dataset:
    for ii in dataset[i]:
        #plt.scatter(ii[0], ii[1], s=100, color=i)
        pass
        
# put the above for loops into a single line ==> just put the for loop at the end of the inner statement
#[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

# now test it out
result = K_nearest_neighbors(dataset, new_features, 3)
print(result)

# now we are going to use the function we defined for the analysis of breast cancer dataset
import random

# somehow the data contains the strains, we need to get rid of them by casting
num_data = df.astype(float).values.tolist()
print(num_data[1:5])

# random shuffle the dataset
random.shuffle(num_data)
print(num_data[1:5])

# split the dataset for training and testing
train_set = {2:[], 4:[]}
test_set  = {2:[], 4:[]}
test_size = 0.2

train_data = num_data[:-int(test_size*len(num_data))]
test_data = num_data[-int(test_size*len(num_data)):]

# now do out training, you could write them in one line, but it's easy to view for the later usage
for i in train_data:
    train_set[i[-1]].append([i[:-1]])
        
# for testing dataset
for j in test_data:
    test_set[j[-1]].append(j[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = K_nearest_neighbors(train_set, data, 5)
        if vote == group:
            correct += 1
        total+=1
        
print("Accuracy is ", correct/total)
