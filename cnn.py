#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:36:23 2020

@author: flora
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 00:40:43 2020

@author: flora
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, datasets
import random
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from time import time
import numpy as np

random.seed(123)

class CNN(nn.Module):
    
    def __init__(self, input_size): 
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size = 3, stride = 1)
        self.fc1 = nn.Linear(13 * 13 * 20 , 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim = 1)

        return x
    
     
''' Process data '''    
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
trainset = datasets.MNIST(root='./data', train = True, download=False, transform = transform)
testset = datasets.MNIST(root='./data', train = False, download=False, transform = transform)

test_bs = 1000
test_dl = DataLoader(testset, batch_size = test_bs)

input_size = 28 * 28
model = CNN(input_size)

def train(bs, opt):
    train_dl = DataLoader(trainset, batch_size = bs)

    ''' Training stage '''    
    train_losses = []
    train_acc = []
    epsilon = 1e-5
    last_loss = 0
    epoch = 0
    delta = 1e5    
    
    while delta > epsilon:
        model.train()
        train_loss = 0
        correct = 0.0
        for x_train, y_train in train_dl:        
            train_pred = model(x_train)
            loss = nn.NLLLoss(reduction = 'sum')(train_pred, y_train)  #nn.CrossEntropyLoss()
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            train_loss += loss.item()
            train_pred_int = train_pred.data.max(1, keepdim = True)[1]
            correct += train_pred_int.eq(y_train.data.view_as(train_pred_int)).sum()
        
        avg_loss = train_loss / len(trainset)
        train_losses.append(avg_loss)
        
        delta = np.abs(avg_loss - last_loss)
        last_loss = avg_loss
        epoch += 1  
        
        avg_acc = correct / len(trainset) * 100
        train_acc.append(avg_acc)
    
        print('Train Epoch: {} \t Loss:{:.4f} \t delta:{:0.4f} \t Acc:{:.0f}%'.format(
            epoch, avg_loss, delta, avg_acc))

    return epoch, train_losses, train_acc

train_bs = 32
opt = optim.SGD(model.parameters(), lr = 0.01)
epoch, train_losses, train_acc = train(train_bs, opt)

#Save model
torch.save(model, './mnist-cnn.pt')

'''Plot '''
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

# Load model and perform test 

''' Testing stage '''

model = torch.load('./mnist-cnn.pt')
test_loss = 0
correct = 0
with torch.no_grad():
    for x_test, y_test in test_dl:
        test_pred_prob = model(x_test)
        test_loss += nn.NLLLoss()(test_pred_prob, y_test).item()  #nn.CrossEntropyLoss
        test_pred = test_pred_prob.data.max(1, keepdim = True)[1]
        correct += test_pred.eq(y_test.data.view_as(test_pred)).sum()
test_loss /= len(test_dl.dataset)
print('Test set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dl.dataset), 
        100. * correct / len(test_dl.dataset)))

    