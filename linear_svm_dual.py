#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:08:34 2020

@author: mengdie
"""

from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

np.random.seed(1)

class SVM():
    
    def __init__(self, c):
        self.c = c
        self.cv = 10
        self.epsilon = 1e-5
        self.X_shuffled = {}
        self.y_shuffled = {}
        
    def X_y_split(self, train_valid_df, test_df):
        ncol = df.shape[1]
        self.train_valid_X = train_valid_df[:, 0: ncol - 1]
        self.train_valid_y = train_valid_df[:, ncol - 1]
        
        self.X_test = test_df[:, 0: ncol - 1]
        self.y_test = test_df[:, ncol - 1]
        #return train_valid_X, train_valid_y, test_X, test_y
        
    def shuffle(self, X, y):
        nrow = X.shape[0]
        idx = np.arange(nrow)
        np.random.shuffle(idx)
        batch_size = (nrow - 1) // self.cv + 1
        for i in range(self.cv):
            sub = idx[i*batch_size : (i+1)*batch_size] 
            self.X_shuffled[i] = X[sub, :] 
            self.y_shuffled[i] = y[sub]
           
    def get_next_train_valid(self, itr): 
        #self.train_test_split(df)
        self.shuffle(self.train_valid_X, self.train_valid_y)
        X_valid = self.X_shuffled[itr]
        y_valid = self.y_shuffled[itr]
        X_train = np.concatenate([self.X_shuffled[key] for key in self.X_shuffled.keys() if key != itr], axis = 0)
        y_train = np.concatenate([self.y_shuffled[key] for key in self.y_shuffled.keys() if key != itr], axis = 0)
        
        return X_train, y_train, X_valid, y_valid
        
    def svmfit(self, X, y, c):
        nrow = X.shape[0]
        ncol = X.shape[1]
        
        P = matrix(np.multiply(np.matmul(X, X.T), np.outer(y.T, y)))
        q = matrix(np.ones(nrow) * -1)
        
        A = matrix(y, (1, nrow))
        b = matrix(0.0)
        
        tmp1 = np.diag(np.ones(nrow) * -1)
        tmp2 = np.identity(nrow)
        G = matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(nrow)
        tmp2 = np.ones(nrow) * c
        h = matrix(np.hstack((tmp1, tmp2)))
        
        result = qp(P, q, G, h, A, b)
        alpha = np.ravel(result['x'])
        
        # Use support vectors to calculate weight and use free support vectors to calculate intercept:
        weight = np.zeros(ncol)
        intercept = 0.0
        cnt = 0
        for i in range(nrow):
            if alpha[i] > self.epsilon:
                weight += alpha[i] * y[i] * X[i]
                intercept += y[i]
                intercept -= np.sum(alpha * y * np.dot(X[i], weight))
#            if alpha[i] < c:
#                intercept += y[i] - np.dot(X[i], weight)
                cnt += 1
        return weight, intercept / cnt

        
    def predict(self, X, weight, intercept):
        return np.sign(np.dot(X, weight) + intercept)
    
    def evaluate(self, y_true, y_predicted):
        return sum(y_true == y_predicted) / len(y_true)
    
    def k_fold_cv(self, train_data, test_data, k, c):
        self.X_y_split(train_data, test_data)
        X_train, y_train, X_valid, y_valid = self.get_next_train_valid(k)
        weight, intercept = self.svmfit(X_train, y_train, c)
        y_train_predicted = self.predict(X_train, weight, intercept)
        y_valid_predicted = self.predict(X_valid, weight, intercept)
        y_test_predicted = self.predict(self.X_test, weight, intercept)
        
        train_acc = self.evaluate(y_train, y_train_predicted)
        valid_acc = self.evaluate(y_valid, y_valid_predicted)
        test_acc = self.evaluate(self.y_test, y_test_predicted)
        
        return train_acc, valid_acc, test_acc
    
''' Prep data '''
df = np.array(pd.read_csv("hw2data.csv", header = None))
train_rows = int(df.shape[0] * 0.8)
train_valid_idx = np.random.choice(df.shape[0], train_rows, replace = False)
test_idx = [i for i in range(df.shape[0]) if i not in train_valid_idx]

train_valid_df = df[train_valid_idx, :]
test_df = df[test_idx, :]

''' CV Training '''
c_lst = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
cv = 10
train_acc_lst, valid_acc_lst, test_acc_lst = [], [], []
for c in c_lst:
    train_acc = 0
    valid_acc = 0
    test_acc = 0
    for k in range(cv):
        svm = SVM(c)
        train_accuracy, valid_accuracy, test_accuracy = svm.k_fold_cv(train_valid_df, test_df, k, c)
        train_acc += train_accuracy
        valid_acc += valid_accuracy
        test_acc += test_accuracy
    train_acc_lst.append(train_acc / cv)
    valid_acc_lst.append(valid_acc / cv)
    test_acc_lst.append(test_acc / cv)
        
'''Plot error rates'''
x = np.log(c_lst)
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x, train_acc_lst, s=10, c='b', marker="s", label='Training error')
ax1.scatter(x, valid_acc_lst, s=10, c='r', marker="o", label='Validation error')
ax1.scatter(x, test_acc_lst, s=10, c='g', marker="+", label='Test error')
plt.legend(loc = 'upper left');
plt.xlabel("C value after log transform")
plt.ylabel("Accuracy rate (in percent)")
plt.show() 
