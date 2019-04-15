#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15th April 2019
@author: Akihiro Inui
Multi-class logistic regression
"""

import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits


# Load dataset
iris = load_digits()

# X: training, y: test
X = iris.data
y = iris.target

#  Convert to tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

# Define architecture
net = nn.Linear(X.size()[1], 10)

# Define cost function (softmax cross entropy)
loss_function = nn.CrossEntropyLoss()

# Define optimizer (sigmoid)
optimizer = optim.SGD(net.parameters(), lr=0.01)

# List of loss
losses = []

# Iterate over 100 times
for epoc in range(100):

    # Delete gradient from previous iteration
    optimizer.zero_grad()

    # Make prediction
    y_pred = net(X)

    # Calculate mean squared error
    loss = loss_function(y_pred, y)
    loss.backward()

    # Update gradient
    optimizer.step()

    # Keep loss record
    losses.append(loss.item())

# # Plot loss history
plt.plot(losses)
plt.show()
