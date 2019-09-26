from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import cPickle
import os
from scipy.io import loadmat


def softmax(o):
    """
    softmax function
    """
    y = np.exp(o)
    y /= y.sum(1).reshape((-1, 1))
    return y

############
#Part 2
############
def forward(x,W,b):
    """
    o_i = Sigma_j w_ji x_j + b_i
    """
    # dim of x: N * 784
    # dim of W: 10 * 784
    # dim of b: N * 10
    o = np.dot(x,W.T) + b
    return softmax(o)
    

def loss_fn(y, t):
    """
    loss function
    """
    return -np.sum(t*np.log(y)) # sum of the negative log-probabilities

########
#Part 3(a)
########
# Proof!!!!!!!!!!
    
def loss_grad(y,t):
    """
    gradient of the loss function
    """
    return y-t


########
#Part 3(b)
########
def get_gradient(x,t,b,W):
    """
    compute the bias gradient and the weight gradient
    """
    y = forward(x,W,b)
    lossGradient = loss_grad(y,t)
    weight_grad = np.dot( (lossGradient.T), x )
    bias_grad = np.sum(lossGradient,axis=0)
    return y,weight_grad,bias_grad

M = loadmat("mnist_all.mat")


### check the gradient at several coordinates using finite differences
np.random.seed(0)    
size = 100
x = np.zeros((size,784))
t = np.zeros((size,10))

# randomly select 100 images from the data set and construct x and t matrix
for i in range(size):
    digit = int(np.random.rand(1)*10)
    number = int(np.random.rand(1) * M["train"+str(digit)].shape[0])
    t[i,digit] = 1
    x[i,:] = M["train"+str(digit)][number].reshape((1,784))/255.0

# randomly construct weight and bias matrix    
W = np.random.rand(10,784)
b = np.random.rand(size,10)

diff = 0
y1, weight_grad, bias_grad = get_gradient(x,t,b,W) # compute the gradient
C1 = loss_fn(y1,t) # compute the cost

### gradient check for weight gradient
h_ = [0.1,0.001,0.0001,0.00001,0.000001]
for h in h_:

    for m in range(5):
        np.random.seed(m)
        i = int(np.random.rand(1)*W.shape[0]) # randomly choose the coordinates
        j = int(np.random.rand(1)*W.shape[1])
        
        W2 = np.copy(W)
        W2[i,j] += h # increase one element 
        y2 = forward(x,W2,b)
        C2 = loss_fn(y2,t)
        
        gradient_approx = (C2 - C1)/h # approximate the gradient 
        
        total = abs(gradient_approx) + abs(weight_grad[i,j])
        if total != 0:
            # compute relative error
            diff += abs(gradient_approx - weight_grad[i,j])/total

    diff /= 5.0
    # print the result
    print("The relative error for weight gradient when h = %.6f is %.8f%%" % (h,diff*100))

    
    
    
### gradient check for bias gradient
print("\n")
for h in h_:

    for m in range(5):
        np.random.seed(m)
        j = int(np.random.rand(1)*b.shape[1]) # randomly choose the coordinates
        
        b2 = np.copy(b)
        b2[:,j] += h 
        y2 = forward(x,W,b2)
        C2 = loss_fn(y2,t)
        
        gradient_approx = (C2 - C1)/h # approximate the gradient 
        
        total = abs(gradient_approx)+abs(bias_grad[j])
        if total != 0:
            # compute relative error
            diff += abs(gradient_approx - bias_grad[j])/total
    
    diff /= 5.0
    # print the result
    print("The relative error for bias gradient when h = %.6f is %.8f%%" % (h, diff*100))





