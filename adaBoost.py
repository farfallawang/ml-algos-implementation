#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:07:52 2020

@author: flora
"""
import pandas as pd
import numpy as np
from sklearn import tree
import collections
import random
import matplotlib.pyplot as plt

random.seed(123)
train_df = pd.read_csv("cancer_train.csv")
test_df = pd.read_csv("cancer_test.csv")
train_X = np.array(train_df.iloc[:, :-1])
train_y = np.array(train_df.iloc[:, -1])
test_X = np.array(test_df.iloc[:, :-1])
test_y = np.array(test_df.iloc[:, -1])
# Set 0 labels to -1
train_y[train_y == 0] = -1
test_y[test_y == 0] = -1

n_trees = 100
global train_row, test_row
train_row = train_X.shape[0]  
test_row = test_X.shape[0]  
train_err_lst = []
test_err_lst = []

''' Training model '''
def train(train_X, train_y, n_trees):
    old_weights = [1/train_row for i in range(train_row)]
    final_train_y = [0 for i in range(train_row)]
    final_test_y = [0 for i in range(test_row)]
    cumulative_alpha = 0
    for wl in range(1, n_trees + 1):
        clf = tree.DecisionTreeClassifier(max_depth = 1).fit(train_X, train_y, sample_weight = old_weights)
        y_train_pred = clf.predict(train_X)
        y_test_pred = clf.predict(test_X)
        epsilon = sum(old_weights * (y_train_pred != train_y))        
        alpha = 1/2 * np.log((1 - epsilon)/epsilon + 1e-5)
        if alpha < 0:
            print("Error: alpha is less than 0")
            break
        
        updates = [np.exp(-alpha * train_y[i] * y_train_pred[i]) for i in range(train_row)]
        new_weights = [i * j for i, j in zip(old_weights, updates)]
        new_weights_normalized = [weight/sum(new_weights) for weight in new_weights]
        old_weights = new_weights_normalized #new_weights_normalized
        #sample_idx = np.random.choice(train_row, train_row, p = new_weights_normalized)
        #train_X, train_y = train_X[sample_idx, :], train_y[sample_idx]
        
        final_train_y += alpha * y_train_pred
        final_test_y += alpha * y_test_pred
        cumulative_alpha += alpha
        
        train_err = round(sum(np.sign(final_train_y) != train_y)/len(train_y), 2)
        test_err = round(sum(np.sign(final_test_y) != test_y)/len(test_y), 2)
        train_err_lst.append(train_err)
        test_err_lst.append(test_err)
        
        if wl % 25 == 0:
            margin = [final_train_y[i] * train_y[i] /cumulative_alpha for i in range(train_row)]
            fig = plt.figure()
            n_bins = 100
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # plot the cumulative histogram
            n, bins, patches = ax.hist(margin, n_bins, range = (-1,1), density=True, histtype='step',
                                       cumulative=True, label='Empirical')
            plt.xlabel('Margin')
            plt.title('Cumulative margin distribution with ' + str(wl) + ' weak learners')
            fig

        #print("Train err", train_err, "Test error", test_err, "\n")
        
    return np.sign(final_train_y), np.sign(final_test_y)

''' Training model '''
final_train_y, final_test_y = train(train_X, train_y, n_trees)


''' Plot error '''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(n_trees), train_err_lst, color='blue', label = 'train err')
ax.plot(range(n_trees), test_err_lst, color='orange', label = 'test err')
leg = plt.legend()
plt.xlabel('Number of weak learners')
plt.ylabel('Error rate')
plt.show



