#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:40:39 2020

@author: flora
"""

import pandas as pd
import numpy as np
from sklearn import tree
import random
import matplotlib.pyplot as plt

random.seed(123)
train_df = pd.read_csv("health_train.csv")
test_df = pd.read_csv("health_test.csv")
train_X = np.array(train_df.iloc[:, :-1])
train_y = np.array(train_df.iloc[:, -1])
test_X = np.array(test_df.iloc[:, :-1])
test_y = np.array(test_df.iloc[:, -1])

'''Part a '''
def randomForest(train_X, train_y, test_X, test_y, n_trees, max_feat, acc):
    train_row = train_X.shape[0]  
    test_row = test_X.shape[0]  
    train_pred = [0 for i in range(train_row)]
    test_pred = [0 for i in range(test_row)]

    for i in range(n_trees):
        print('Iteration', i, '...' )
        '''Boostrap the dataset '''
        sample_idx = np.random.choice(train_row, train_row, replace = True)
        sampled_train_X, sampled_train_y = train_X[sample_idx, :], train_y[sample_idx]
        clf = tree.DecisionTreeClassifier(min_samples_split = 2, max_features = max_feat).fit(sampled_train_X, sampled_train_y)
        train_pred += clf.predict(train_X)
        test_pred += clf.predict(test_X)
        cutoff = (i+1) / 2
        final_train_pred = [0 if pred < cutoff else 1 for pred in train_pred]
        final_test_pred = [0 if pred < cutoff else 1 for pred in test_pred]
        
    if acc:
        final_train_acc = sum(final_train_pred == train_y)/len(train_y) * 100
        final_test_acc = sum(final_test_pred == test_y)/len(test_y) * 100
        return final_train_acc, final_test_acc
    else:
        final_train_err = sum(final_train_pred != train_y)/len(train_y) * 100
        final_test_err = sum(final_test_pred != test_y)/len(test_y) * 100
        return final_train_err, final_test_err


'''Part b '''
random_feature_set = [50, 100, 150, 200, 250]
n_trees = 100
train_error_lst = []
test_error_lst = []
for max_feat in random_feature_set:
    train_error, test_error = randomForest(train_X, train_y, test_X, test_y, n_trees, max_feat, acc = False)
    train_error_lst.append(train_error)
    test_error_lst.append(test_error)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(random_feature_set, train_error_lst, color = 'blue', label = 'train error')
ax.plot(random_feature_set, test_error_lst, color = 'orange', label = 'test error')
plt.legend()
plt.xlabel('Size of feature set')
plt.ylabel('Error rate (%)')
plt.title('Feature set size VS Error rate')
plt.show


    
'''Part c '''
num_of_trees = [10, 20, 40, 80, 100]
max_feat = 250
train_acc_lst = []
test_acc_lst = []
for n_trees in num_of_trees:
    train_acc, test_acc = randomForest(train_X, train_y, test_X, test_y, n_trees, max_feat, acc = True)
    train_acc_lst.append(train_acc)
    test_acc_lst.append(test_acc)
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(num_of_trees, train_acc_lst, color = 'blue', label = 'train accuracy')
ax.plot(num_of_trees, test_acc_lst, color = 'orange', label = 'test accuracy')
plt.legend()
plt.xlabel('Number of trees')     
plt.ylabel('Accuracy rate (%)')
plt.title('Number of trees VS Accuracy rate')
plt.show