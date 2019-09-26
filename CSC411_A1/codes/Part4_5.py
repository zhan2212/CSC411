from __future__ import division
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


def grad_descent_Momentum(f, df, x, t, xTest, tTest, init_W, alpha, momentum):
    """
    the gradient descent function with momentum
    """
    performanceM = array([])
    performanceMTest = array([])
    EPS = 1e-5   #EPS = 10**(-5)
    prev_W = init_W-10*EPS
    W = init_W.copy()
    max_iter = 1500
    iter  = 0
    p = 0
    while norm(W - prev_W) >  EPS and iter < max_iter:
        prev_W = W.copy()
        p = momentum * p - alpha * df(x, t, W)
        W += p
        performanceM = np.append(performanceM,performance(W,x,t))
        performanceMTest = np.append(performanceMTest,performance(W,xTest,tTest))
        if iter % 50 == 0:
            print "Iter", iter
            print "Cost: ", f(x,t,W)
            print "Gradient: ", sum(df(x, t, W))
            print "Train Performance: ", performance(W,x,t)
            print "Test Perofrmance:", performance(W,xTest,tTest)
        iter += 1
    return W, performanceM, performanceMTest


def grad_descent(f, df, x, t, xTest, tTest, init_W, alpha):
    """
    the gradient descent function without momentum
    """
    performanceM = array([])
    performanceMTest = array([])
    EPS = 1e-5   #EPS = 10**(-5)
    prev_W = init_W-10*EPS
    W = init_W.copy()
    max_iter = 1500
    iter  = 0
    while norm(W - prev_W) >  EPS and iter < max_iter:
        prev_W = W.copy()
        W -= alpha*df(x, t, W)
        performanceM = np.append(performanceM,performance(W,x,t))
        performanceMTest = np.append(performanceMTest,performance(W,xTest,tTest))
        if iter % 50 == 0:
            print "Iter", iter
            print f(x,t,W)
            print "Gradient: ", sum(df(x, t, W)),
            print "Train Performance:", performance(W,x,t)
            print "Test Perofrmance:", performance(W,xTest,tTest)
        iter += 1
    return W, performanceM, performanceMTest


def softmax(o):
    """
    softmax function
    """
    y = np.exp(o)
    y /= y.sum(1).reshape((-1, 1))
    return y

def loss_fn(y, t):
    """
    loss function
    """
    return -np.sum(t*np.log(y)) # sum of the negative log-probabilities


def f(x,t,W):
    """
    combined function of forward part
    """
    o = np.dot(x,W.T)
    y = softmax(o)
    return loss_fn(y, t)

def df(x,t,W):
    """
    combined function of back propagation part
    """
    o = np.dot(x,W.T)
    y = softmax(o)
    lossGradient = y - t
    combined_grad = np.dot(lossGradient.T, x)
    return combined_grad

def performance(trainedW,x,t):
    """
    evaluate the performance of the learning
    """
    o = np.dot(x,trainedW.T)
    y = softmax(o)
    corrNum = 0
    for i in range(y.shape[0]):
        ind = y[i].argmax(axis=0)
        flag = t[i].argmax(axis=0)
        #print ind,flag
        if ind == flag:
            corrNum += 1
    return corrNum / y.shape[0]

# load the data
M = loadmat("mnist_all.mat")

### Training Set
# construct x
x = M["train0"][0].reshape((1,784))
for i in range(10):
    for j in range(M["train"+str(i)].shape[0]):
        print i,j
        if i == 0 and j == 0:
            continue
        else:
            x = vstack( (x , M["train"+str(i)][j].reshape((1,784))))
x1 = x/255.0
x1 = np.concatenate((x1,np.ones((x1.shape[0],1))),axis=1)

# construct W
W1 = np.random.normal(0, 0.0001, (10,785))

# construct t
t1 = np.zeros((60000,10))
count = 0
for i in range(10):
    for j in range(M["train"+str(i)].shape[0]):
        t1[count,i] = 1
        print(count)
        count += 1
        
### Test set
# construct x
xTest = M["test0"][0].reshape((1,784))
for i in range(10):
    for j in range(M["test"+str(i)].shape[0]):
        print i,j
        if i == 0 and j == 0:
            continue
        else:
            xTest = vstack( (xTest , M["test"+str(i)][j].reshape((1,784))))
xTest1 = xTest/255.0
xTest1 = np.concatenate((xTest1,np.ones((xTest1.shape[0],1))),axis=1)

# construct t
tTest1 = np.zeros((xTest1.shape[0],10))
count = 0
for i in range(10):
    for j in range(M["test"+str(i)].shape[0]):
        tTest1[count,i] = 1
        print(count)
        count += 1
 
    
    
    
#########
# Part 4
#########
       
# learn using gradient descent without momentum       
trainedW, performanceM, performanceMTest = grad_descent(f, df, x1, t1,xTest1, tTest1, W1, 5e-6)
# print the result
print("The accuracy rate for training set is:", performance(trainedW,x1,t1))
print("The accuracy rate for test set is:", performance(trainedW,xTest1,tTest1))

# plot the result 
fig = plt.figure()
plt.plot(range(1500),performanceM.tolist(),'r')
plt.plot(range(1500),performanceMTest.tolist(),'b')
plt.show()



#########
# Part 5
#########

# learn using gradient descent without momentum     
trainedW2, performanceW2, performanceMTest2 = grad_descent_Momentum \
(f, df, x1,t1,xTest1, tTest1, W1, 5e-6, 0.9)
# print the result W1, 5e-6, 0.9)
# print the result
print "The accuracy rate for training set is:", performance(trainedW2,x1,t1)
print "The accuracy rate for test set is:", performance(trainedW2,xTest1,tTest1)

# plot the result 
fig = plt.figure()
plt.plot(range(1500),performanceM2.tolist(),'r')
plt.plot(range(1500),performanceMTest2.tolist(),'b')
plt.show()






