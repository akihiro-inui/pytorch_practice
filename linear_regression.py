#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14th April 2019
@author: Akihiro Inui

Example:
    Derive optimal coefficient for
    y = 1 + 2x1 + 3x2
"""

import torch
from matplotlib import pyplot as plt


# True coefficients
w_true = torch.Tensor([1, 2, 3])

# Prepare data
X = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1)

# Calculate y
y = torch.mv(X, w_true) + torch.randn(100) * 0.5

# Initialize weights (set requires_grad True to enable derivative)
w = torch.randn(3, requires_grad=True)

# Learning rate
lr = 0.1

# Define loss function
losses = []

# Iterate 100 times
for epoc in range(100):
    # Delete previous gradient calculated by backward
    w.grad = None

    # Make prediction (mv: matrix x vector)
    y_pred = torch.mv(X, w)

    # Calculate loss with mean squared error
    loss = torch.mean((y - y_pred)**2)
    loss.backward()

    # Update gradient
    w.data = w.data - lr * w.grad.data

    # Make log of the loss
    losses.append(loss.item())

# # Plot loss history
plt.plot(losses)
plt.show()

# Print predicted coefficients
print("Predicted coefficients are {coeff_1} {coeff_2} {coeff_3}".format(coeff_1=w[0], coeff_2=w[1], coeff_3=w[2]))
