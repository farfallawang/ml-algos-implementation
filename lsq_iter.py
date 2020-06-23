#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:23:26 2020

@author: mengdie
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(111)

A = np.random.rand(20, 10)
b = np.random.rand(20, 1)
mu = 1 / (np.linalg.norm(A) ** 2)

iterations = 500
epsilon = 1e-10

def lsq_iter(A, b):
    w_prev = np.zeros((10, 1))
    w_hat = lsq(A, b) 
    diff_lst = []
    
    for i in range(iterations):
        tmp = np.matmul(A.T, (np.matmul(A, w_prev) - b)) #(10, 20) * (20, 1)
        w_new = w_prev - mu * tmp 
        diff_lst.append(np.abs(np.linalg.norm(w_hat) - np.linalg.norm(w_new)))
        #print("Iter", i)
        if np.linalg.norm(tmp) < epsilon:
            break
        w_prev = w_new
        
    return w_new, diff_lst

def lsq(A, b):
    tmp = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)) 
    
    return np.matmul(tmp, b)

w_hat = lsq(A, b)  
w, diff_lst = lsq_iter(A, b)    

plt.plot(diff_lst)
plt.xlabel('Number of Iteration ')
plt.ylabel('Loss: ||w_k - w*||')
plt.show()