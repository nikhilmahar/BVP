# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:54:08 2022

@author: NIKHIL
"""

## let's import the relevant libraries
import torch
import torch.nn as nn
from time import perf_counter
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import requests
import os

## check if GPU is available and use it; otherwise use CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# N is a Neural Network - This is exactly the network used by Lagaris et al. 1997
N = nn.Sequential(nn.Linear(1, 10), nn.Sigmoid(), nn.Linear(10,2, bias=False))

# Initial condition
f0 = 1
k0=1

# The Psi_t function
Psi_t = lambda x: x*k0+f0 + (x**2) * N(x)

# The right hand side function
#f = lambda x, Psi: 2*x
def f(x):
  return 2*x
# The loss function
def loss(x):
    x.requires_grad = True
    outputs = Psi_t(x)
    Psi_t_x = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs),
                                  create_graph=True)[0]
    #print(Psi_t_x)
    Psi_t_xx = torch.autograd.grad(Psi_t_x, x, grad_outputs=torch.ones_like(Psi_t_x),
                                  create_graph=True)[0]
    #print(Psi_t_xx)
    print(torch.mean((Psi_t_xx - f(x)) ** 2))                              
    return torch.mean((Psi_t_xx - f(x)) ** 2)

# # Optimize (same algorithm as in Lagaris)
# optimizer = torch.optim.LBFGS(N.parameters())

# # The collocation points used by Lagaris
# x = 2*torch.rand(1000,1)
# #print(x)
# # Run the optimizer
# def closure():
#     optimizer.zero_grad()
#     l = loss(x)
#     print(l)
#     l.backward()
#     return l
    
# for i in range(2):
#     optimizer.step(closure)

# # Let's compare the result to the true solution
# xx = np.linspace(0, 2, 100)[:, None]
# with torch.no_grad():
#     yy = Psi_t(torch.Tensor(xx)).numpy()
# yt = (xx**2)/2+xx+1
# fig, ax = plt.subplots(dpi=100)
# ax.plot(xx, yt, label='True')
# ax.plot(xx, yy, '--', label='Neural network approximation')
# ax.set_xlabel('$x$')
# ax.set_ylabel('$\Psi(x)$')
# plt.legend(loc='best');

# We need to reinitialize the network
N = nn.Sequential(nn.Linear(1, 100,bias=False), nn.Sigmoid(), nn.Linear(100,100,bias=False),nn.Sigmoid(),nn.Linear(100,1, bias=False))

# Let's see now if a stochastic optimizer makes a difference
adam = torch.optim.Adam(N.parameters(), lr=0.001)

# The batch size you want to use (how many points to use per iteration)
n_batch = 50

# The maximum number of iterations to do
max_it = 1000

for i in range(max_it):
    # Randomly pick n_batch random x's:
    x = 2 * torch.rand(n_batch, 1)
    #print(x)
    # Zero-out the gradient buffers
    adam.zero_grad()
    # Evaluate the loss
    l = loss(x)
    # Calculate the gradients
    l.backward()
    # Update the network
    adam.step()
    # Print the iteration number
    if i % 100 == 99:
        print(i+1)
        #print(l)
        
# Let's compare the result to the true solution
xx = np.linspace(0, 2, 100)[:, None]
with torch.no_grad():
    yy = Psi_t(torch.Tensor(xx)).numpy()
yt  = (xx**3)/3+xx+1

fig, ax = plt.subplots(dpi=100)
ax.plot(xx, yt, label='True')
ax.plot(xx, yy, '--', label='Neural network approximation')
ax.set_xlabel('$x$')
ax.set_ylabel('$\Psi(x)$')
plt.legend(loc='best');