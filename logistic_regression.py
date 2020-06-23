#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:51:03 2020

@author: mengdie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
np.random.seed(1)

feature = np.array(pd.read_csv("HW1-data/IRISFeat.csv", header = None))
label = np.array(pd.read_csv("HW1-data/IRISlabel.csv", header = None))

class LogisticRegression():
    
    def __init__(self, X, y, cv):
        self.X = X
        self.y = y
        self.cv = cv
        self.dimension = X.shape[1]
        self.size = y.shape[0]
        self.X_shuffled = {}
        self.y_shuffled = {}
        
    def shuffle(self):
        idx = np.arange(self.size)
        np.random.shuffle(idx)
        batch_size = (self.size - 1) // self.cv + 1
        for i in range(self.cv):
            sub = idx[i*batch_size : (i+1)*batch_size] 
            self.X_shuffled[i] = self.X[sub, :] 
            self.y_shuffled[i] = self.y[sub]
           
    def get_next_train_valid(self, itr): 
        self.shuffle()
        X_valid = self.X_shuffled[itr]
        y_valid = self.y_shuffled[itr]
        X_train = np.concatenate([self.X_shuffled[key] for key in self.X_shuffled.keys() if key != itr], axis = 0)
        y_train = np.concatenate([self.y_shuffled[key] for key in self.y_shuffled.keys() if key != itr], axis = 0)
        
        return X_train, y_train, X_valid, y_valid
        
    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x)) #negative x!
    
    def train(self, X_train, y_train):
        nrow = X_train.shape[0]
        intercept = np.ones((nrow, 1))
        X_train = np.concatenate((intercept, X_train), axis = 1)
        max_iter = 1000
        #epsilon = 1e-10
        w = np.zeros(X_train.shape[1]) #np.random.rand(self.dimension + 1)
        step_size = 0.01 
        itr = 0
        
        while itr < max_iter:
#            h = lambda i: self.sigmoid(np.matmul(X_train[i], w)) 
#            gradient = np.zeros(self.dimension + 1)
#            for i in range(nrow):
#                gradient += X_train[i] * (h(i) - y_train[i])                
            h = self.sigmoid(np.matmul(X_train, w))   # (n, d) * (d, 1) = (n,1)  
            gradient = np.dot(X_train.T, (h - y_train[:, 0])) / nrow #(d, n) * (n, 1) = (d, 1)
            w -= step_size * gradient #/ nrow  #divide by num of samples
#            diff = np.abs(np.linalg.norm(w_new) - np.linalg.norm(w_prev))
#            print("Iter:", itr, "w difference", diff)
#            if diff < epsilon:
#                break
#            w_prev = w_new
            itr += 1
        
        return w[1:len(w)], w[0]
        
    def predict_prob(self, X_valid, model_weights, model_intercept):
        nrow = X_valid.shape[0]
        intercept = np.ones((nrow, 1))
        #X_valid = np.concatenate((intercept, X_valid), axis = 1)
        y_predict_class = np.expand_dims(np.matmul(X_valid, model_weights), axis = 1) + intercept * model_intercept  #need sigmoid transformation?   
        
        return self.sigmoid(y_predict_class)
    
    def predict(self, X_valid, model_weights, model_intercept, threshold):
        y_predict_class = self.predict_prob(X_valid, model_weights, model_intercept)
        return y_predict_class >= threshold
    
    def evaluate(self, y_predict_class, y_valid):        
        return sum(y_predict_class[:,0] != y_valid[:, 0]) / float(y_predict_class.shape[0]) * 100
        
c = LogisticRegression(feature, label, 5)

'''Calculate train, validation error and confusion matrix in each fold '''
train_error_lst = []
valid_error_lst = []
confusion_matrix_lst = []
for i in range(5):
    X_train, y_train, X_valid, y_valid = c.get_next_train_valid(0)
    model_weights, model_intercept = c.train(X_train, y_train)
    y_train_predict = c.predict(X_train, model_weights, model_intercept, 0.5)
    y_valid_predict = c.predict(X_valid, model_weights, model_intercept, 0.5)
    train_error_lst.append(c.evaluate(y_train_predict, y_train))
    valid_error_lst.append(c.evaluate(y_valid_predict, y_valid))
    confusion_matrix_lst.append(confusion_matrix(y_valid, y_valid_predict))

'''Plot error rates'''
x = [1, 2, 3, 4, 5] 
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x, train_error_lst, s=10, c='b', marker="s", label='Training error')
ax1.scatter(x, valid_error_lst, s=10, c='r', marker="o", label='Validation error')
plt.legend(loc = 'upper left');
plt.xlabel("Iteartion")
plt.ylabel("Error rate (in percent)")
plt.show() 



