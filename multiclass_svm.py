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
from sklearn.metrics import confusion_matrix

np.random.seed(1)

class SVM():
    
    def __init__(self, c, sigma):
        self.c = c
        self.sigma = sigma
        self.cv = 10
        self.nclass = 10
        self.epsilon = 1e-3
        self.X_shuffled = {}
        self.y_shuffled = {}
        
    def X_y_split(self, train_valid_df, test_df):
        ncol = train_valid_df.shape[1]
        train_valid_X = train_valid_df[:, 0: ncol - 1]
        train_valid_y = train_valid_df[:, ncol - 1]
        
        X_test = test_df[:, 0: ncol - 1]
        y_test = test_df[:, ncol - 1]
        return train_valid_X, train_valid_y, X_test, y_test
            
    def gaussian_kernel(self, x, y):
        return np.exp(-linalg.norm(x-y)**2 / (2 * (self.sigma ** 2)))
    
    def multiclass_svm_train(self, X, y, c):
        nrow = X.shape[0]
        
        self.sv = []
        self.sv_y = []
        self.alpha = []
        self.intercept = []
        
        sv_lst, sv_y_lst, alpha_lst, intercept_lst = [], [], [], []
        
        end = self.nclass + 1
        for i in range(1, end):
            new_y = np.ones(nrow)
            new_y[y != i] = -1
            intercept, sv, sv_y, alpha = self.rbf_svm_train(X, new_y, c)
            self.sv.append(sv)
            self.sv_y.append(sv_y)
            self.alpha.append(alpha)
            self.intercept.append(intercept)
            sv_lst.append(sv)
            sv_y_lst.append(sv_y)
            alpha_lst.append(alpha)
            intercept_lst.append(intercept)
        
        return sv_lst, sv_y_lst, alpha_lst, intercept_lst
    
    def rbf_svm_train(self, X, y, c):
        nrow = X.shape[0]
        ncol = X.shape[1]
        
        # Rewrite P using kernel function
        G = np.zeros([nrow, nrow])
        for i in range(nrow):
            for j in range(nrow):
                G[i, j] = self.gaussian_kernel(X[i], X[j]) * (y[i] * y[j])
        
        P = matrix(G)
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
                
        weight = np.zeros(ncol)
        intercept = 0.0
        cnt = 0
        for i in range(nrow):
            if alpha[i] > self.epsilon:
                weight += alpha[i] * y[i] * X[i]
                intercept += y[i]
                intercept -= np.sum(alpha * y * np.dot(X[i], weight))
                cnt += 1
        
        # find support vectors and its corresponding y and alpha
        sv = alpha > self.epsilon
        
        return intercept / cnt, X[sv], y[sv], alpha[sv]
            
    def predict(self, X):
        nrow = X.shape[0]
        y_predicted = np.zeros([nrow, self.nclass])
        for i in range(nrow):
            for cl in range(self.nclass):
                res = 0
                for alpha, sv_y, sv in zip(self.alpha[cl], self.sv_y[cl], self.sv[cl]):
                        res += alpha * sv_y * self.gaussian_kernel(X[i], sv)
                y_predicted[i, cl] = res + self.intercept[cl]
         
        y_pred = np.zeros(nrow) 
        for i in range(nrow): 
            tmp = list(y_predicted[i])
            y_pred[i] = tmp.index(max(y_predicted[i])) + 1  
        return y_pred
    
    def evaluate(self, y_true, y_predicted):
        return sum(y_true == y_predicted) / len(y_true)
    
    
''' Training '''
train_df = np.array(pd.read_csv("mfeat_train.csv", header = 0, index_col = 0))
test_df = np.array(pd.read_csv("mfeat_test.csv", header = 0, index_col = 0))
c = 10
sigma = 10
svm = SVM(c, sigma)
X_train, y_train, X_test, y_test = svm.X_y_split(train_df, test_df)
sv_lst, sv_y_lst, alpha_lst, intercept_lst = svm.multiclass_svm_train(X_train, y_train, c)
y_test_pred = svm.predict(X_test)
test_acc = svm.evaluate(y_test, y_test_pred) #94.16%
print("Test accuracy", test_acc)

confusion_matrix(y_test, y_test_pred)

''' Save all support vector info and corresponding alpha'''
import pickle

with open('p6.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump([sv_lst, sv_y_lst, alpha_lst, intercept_lst], filehandle)
