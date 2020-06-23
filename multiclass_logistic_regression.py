#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:51:03 2020

@author: mengdie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
np.random.seed(1)


class MultiClassLogisticRegression():
    
    def __init__(self, cv = None):
        #self.X = X
        #self.y = y
        self.cv = cv
        #self.dimension = X.shape[1]
        #self.size = y.shape[0]
        self.X_shuffled = {}
        self.y_shuffled = {}
        
    def X_y_split(self, train_valid_df, test_df):
        ncol = train_valid_df.shape[1]
        train_valid_X = train_valid_df[:, 1: ncol]
        train_valid_y = train_valid_df[:, 0]
        
        X_test = test_df[:, 1: ncol]
        y_test = test_df[:, 0]
        return train_valid_X, train_valid_y, X_test, y_test
    
    def softmax(self, x):  
        e = np.exp(x - np.max(x)) #prevent overflow
        
        if e.ndim == 1:
            return e / np.sum(e, axis = 0)
        else:  
            return e / np.array([np.sum(e, axis = 1)]).T
        
    
    def train(self, X_train, y_train):
        nrow = X_train.shape[0]
        intercept = np.ones((nrow, 1))
        X_train = np.concatenate((intercept, X_train), axis = 1)
        max_iter = 1000
        #epsilon = 1e-10
        self.w = np.zeros(X_train.shape[1]) #np.random.rand(self.dimension + 1)
        step_size = 0.01 
        itr = 0
        
        while itr < max_iter:               
            #h = self.sigmoid(np.matmul(X_train, w))   # (n, d) * (d, 1) = (n,1)  
            h = self.softmax(np.matmul(X_train, self.w)) 
            gradient = np.dot(X_train.T, (h - y_train)) / nrow #(d, n) * (n, 1) = (d, 1)
            self.w -= step_size * gradient #/ nrow  #divide by num of samples
            itr += 1
            
        return self.w
        
    def predict_prob(self, X_valid):
        nrow = X_valid.shape[0]
        intercept = np.ones((nrow, 1))
        X_valid = np.concatenate((intercept, X_valid), axis = 1)
        y_predict_class = np.matmul(X_valid, self.w) #need softmax transformation
        
        return self.softmax(y_predict_class)
    
    def predict(self, X_valid):
        y_predict_class = self.predict_prob(X_valid)
        
        return y_predict_class
    
    def evaluate(self, y_predict_class, y_valid):        
        return sum(y_predict_class != y_valid) / len(y_predict_class) 

train_df = np.array(pd.read_csv("mnist_train.csv", header = None))
test_df = np.array(pd.read_csv("mnist_test.csv", header = None))

       
lr = MultiClassLogisticRegression()
X_train, y_train, X_test, y_test = lr.X_y_split(train_df, test_df)

model_weights = lr.train(X_train, y_train)
y_test_pred = lr.predict(X_test)
test_acc = lr.evaluate(y_test_pred, y_test)
confusion_matrix(y_test, y_test_pred)

''' Save model weights'''
np.savetxt("p6_weights.txt", model_weights)

