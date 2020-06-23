#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:27:54 2020

@author: flora
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, datasets
import random
import matplotlib.pyplot as plt
from time import time
import numpy as np

random.seed(123)

class FeedforwardNN(nn.Module):
    
    def __init__(self, input_size): 
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim = 1)
        return x
    
    
''' Process data '''    
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                              ])  #
trainset = datasets.MNIST(root='./data', train = True, download=False, transform = transform)
testset = datasets.MNIST(root='./data', train = False, download=False, transform = transform)

train_bs = 32
test_bs = 1000
train_dl = DataLoader(trainset, batch_size = train_bs)
test_dl = DataLoader(testset, batch_size = test_bs)


''' Training stage '''
input_size = 28 * 28
model = FeedforwardNN(input_size)
opt = optim.SGD(model.parameters(), lr = 0.001)

train_losses = []
train_acc = []

time_start = time()
epsilon = 1e-5
last_loss = 0
delta = 1e5
epoch = 0

while delta > epsilon:
    model.train()
    train_loss = 0
    correct = 0.0
    for x_train, y_train in train_dl:
        # Flatten MNIST images into a long vector
        x_train = x_train.view(x_train.shape[0], -1)
        
        train_pred = model(x_train)
        
        loss = nn.NLLLoss(reduction = 'sum')(train_pred, y_train)  #nn.CrossEntropyLoss()
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        train_loss += loss.item()
        
        train_pred_int = train_pred.data.max(1, keepdim = True)[1]
        correct += train_pred_int.eq(y_train.data.view_as(train_pred_int)).sum()
        
  
    avg_loss = train_loss / len(trainset)
    avg_acc = correct / len(trainset) * 100
    
    delta = np.abs(avg_loss - last_loss)
    last_loss = avg_loss
    epoch += 1  
    
    train_losses.append(avg_loss)
    train_acc.append(avg_acc)
    
    print('Train Epoch: {} \t Loss:{:.4f} \t delta:{:0.4f} \t Acc:{:.0f}%'.format(
            epoch, avg_loss, delta, avg_acc))
        
print("\nTime to convergence (in minutes) = ", (time()-time_start)/60)

## Save model
#torch.save(model, './mnist-fc.pt')

fig = plt.figure()
plt.plot(range(epoch), train_losses, color='blue')
plt.xlabel('Number of epochs')
plt.ylabel('Negative log likelihood loss')
fig


fig2 = plt.figure()
plt.plot(range(epoch), train_acc, color='orange')
plt.xlabel('Number of epochs')
plt.ylabel('Training accuracy')
fig2


''' Testing stage '''
#import torch
#model = torch.load('./mnist-fc.pt')
test_loss = 0.0
correct = 0
with torch.no_grad():
    for x_test, y_test in test_dl:
        # Flatten data
        x_test = x_test.view(x_test.shape[0], -1)
        test_pred_prob = model(x_test)
        test_loss += nn.NLLLoss()(test_pred_prob, y_test).item()  #nn.CrossEntropyLoss
        test_pred = test_pred_prob.data.max(1, keepdim = True)[1]
        correct += test_pred.eq(y_test.data.view_as(test_pred)).sum()
test_loss /= len(test_dl.dataset)
print('Test set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dl.dataset), 
        100. * correct / len(test_dl.dataset)))

    

