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


def grad_descent_Momentum_Plot(f, df, x, t, init_W, alpha, momentum, max_iter):
    i,j = 5, 385
    m,n = 5, 400
    mo_traj = []
    EPS = 1e-5   #EPS = 10**(-5)
    prev_W = init_W-10*EPS
    W = init_W.copy()
    iter  = 0
    p = np.zeros((W.shape[0],W.shape[1]))
    while norm(W - prev_W) >  EPS and iter < max_iter:
        mo_traj.append((W[i,j],W[m,n]))
        prev_W = W.copy()
        p[i,j] = momentum * p[i,j] - alpha * df(x, t, W)[i,j]
        p[m,n] = momentum * p[m,n] - alpha * df(x, t, W)[m,n]
        W[i,j] += p[i,j]
        W[m,n] += p[m,n]
  
        print "Iter", iter
        print "Cost: ", f(x,t,W)
        print "Gradient: ", sum(df(x, t, W))
        iter += 1
    return mo_traj


def grad_descent_Plot(f, df, x, t, init_W, alpha, max_iter):
    i,j = 5, 385
    m,n = 5, 400
    gd_traj = []
    EPS = 1e-5   #EPS = 10**(-5)
    prev_W = init_W-10*EPS
    W = init_W.copy()
    iter  = 0
    while norm(W - prev_W) >  EPS and iter < max_iter:
        gd_traj.append((W[i,j],W[m,n]))
        prev_W = W.copy()
        gradient = df(x, t, W)
        W[i,j] -= alpha* gradient[i,j]
        W[m,n] -= alpha* gradient[m,n]

        #print "Iter", iter
        #print f(x,t,W)
        #print "Gradient: ", sum(df(x, t, W))
        iter += 1
    return gd_traj


def softmax(o):
    y = np.exp(o)
    y /= y.sum(1).reshape((-1, 1))
    return y


def f(x,t,W):
    # np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
    o = np.dot(x,W.T)
    y = softmax(o)
    return loss_fn(y, t)

def df(x,t,W):
    # np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
    o = np.dot(x,W.T)
    y = softmax(o)
    lossGradient = y - t
    combined_grad = np.dot(lossGradient.T, x)
    
    return combined_grad

alpha = 0.0001
Wplot = np.random.normal(0, 0.0001, (10,785))
Wplot[5,385] = 1
Wplot[5,400] = -0.5
gd_traj = grad_descent_Plot(f, df, x1, t1, Wplot, alpha,15)
mo_traj = grad_descent_Momentum_Plot(f, df, x1, t1, Wplot, alpha, 0.5,15)


w1s = np.arange(-3, 3.5, 0.5)
w2s = np.arange(-3, 3.5, 0.5)
w1z, w2z = np.meshgrid(w1s, w2s)
C = np.zeros([w1s.size, w2s.size])
for i, w1 in enumerate(w1s):
    for j, w2 in enumerate(w2s):
        Wplot[4,385] = w1
        Wplot[5,400] = w2
        C[j,i] = f(x1,t1,Wplot)


CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
plt.legend(loc='top left')
plt.title('Contour plot')
plt.show()
