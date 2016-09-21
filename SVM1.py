#!/usr/bin/python

import pandas as pd 
import numpy as np
from sklearn import svm, cross_validation, preprocessing

df = pd.read_csv('Breast_Cancer.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

# need to implement the svm class here
import matplotlib.pyplot as plt

class Support_Vector_Machine():
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    #training
    def fit(self, data):
        pass
    
    def predict(self, features):
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features), np.array(self.w)) + self.b)
        return classification
        
        




