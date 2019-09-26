from __future__ import division
import numpy as np
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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
from pylab import *

def grad_descent(f, df, X, t, XTest, tTest, XVali, tVali, init_W, alpha,r):
    """
    the gradient descent function without momentum
    """
    performanceM = array([])
    performanceMTest = array([])
    performanceMVali = array([])
    EPS = 1e-5   #EPS = 10**(-5)
    prev_W = init_W-10*EPS
    W = init_W.copy()
    max_iter = 6000
    iter  = 0
    while norm(W - prev_W) >  EPS and iter < max_iter:
        prev_W = W.copy()
        W -= alpha*df(X, t, W, r)
        performanceM = np.append(performanceM,performance(W,X,t))
        performanceMVali = np.append(performanceMVali,performance(W,XVali,tVali))
        performanceMTest = np.append(performanceMTest,performance(W,XTest,tTest))
        if iter % 100 == 0:
            print "Iter", iter
            print f(X,t,W,r)
            print "Gradient: ", sum(df(X, t, W, r))
            print "Train Performance:", performanceM[iter]
            print "Validation Perofrmance:", performanceMVali[iter]
            print "Test Perofrmance:", performanceMTest[iter]        
        iter += 1
    return W , performanceM, performanceMTest, performanceMVali


def sigmoid(o):
    """
    sigmoid function
    """
    y = 1/(1+np.exp(-o))
    return y

def loss_fn(y, t):
    """
    loss function
    """
    L = sum(-t*np.log(y) - (1-t)*np.log(1-y))
    return L


def f(x,t,W,r):
    """
    combined function of forward part
    """
    o = np.dot(x,W.T)
    y = sigmoid(o)
    R = r * sum(W**2)
    return loss_fn(y, t) + R

def df(x,t,W,r):
    """
    combined function of back propagation part
    """
    o = np.dot(x,W.T)
    y = sigmoid(o)
    lossGradient = y-t
    combined_grad = np.dot(lossGradient.T, x)
    return combined_grad + 2*r*W

def performance(trainedW,x,t):
    """
    evaluate the performance of the learning
    """
    o = np.dot(x,trainedW.T)
    y = sigmoid(o)
    corrNum = 0
    for i in range(y.shape[0]):
        if y[i,0] > 0.5:
            y[i,0] = 1
        else:
            y[i,0] = 0
    for i in range(y.shape[0]):
        if y[i,0] == t[i,0]:
            corrNum += 1
    return corrNum / y.shape[0]

def constX(Set,allWords):
    N = len(Set)
    K = len(allWords)
    X = np.zeros((N,K))
    for i in range(len(Set)):
        for j in range(len(allWords)):
            if allWords[j] in Set[i]:
                X[i,j] = 1
    return X

def NBWord(word,allWordsT,P_AllWordsReal,P_real, P_AllWordsFake,P_fake):
    '''
    This function is used to calculate normalized P(Xi|Y)P(Y), which can
    represent 'presence/absense most strongly predicts that the news is real'
    and 'presence/abasense most strongly predicts that the news is fake.'
    '''
    for i in range(len(allWordsT)):
        if allWordsT[i] == word:
            Pxi_r = P_AllWordsReal[i]
            Pxi_f = P_AllWordsFake[i]
            break
    Pr = Pxi_r * P_real
    Pf = Pxi_f * P_fake
    
    return Pr/(Pr+Pf), Pf/(Pr+Pf)
                

# Read fake and real news from .txt files
currPath = os.getcwd()
realPath = currPath + '/clean_real.txt'
fakePath = currPath + '/clean_fake.txt'

# Split the text into sentences
f1 = open(realPath,'r')
realLines = f1.readlines()
f2 = open(fakePath,'r')
fakeLines = f2.readlines()

# Split each sentence into words
fakeWords = []
for i in range(len(fakeLines)):
    fakeWords.append([])
    fakeWords[i] = fakeLines[i].split()   
realWords = []
for i in range(len(realLines)):
    realWords.append([])
    realWords[i] = realLines[i].split()
    
# Divide the database into Training Set, Validation Set and Test Set
np.random.seed(2)
np.random.shuffle(realWords)
np.random.shuffle(fakeWords)

index70 = int(np.floor(len(realWords)*0.7))
index85 = int(np.floor(len(realWords)*0.85))
realTrain = realWords[0:index70]
realVali = realWords[index70:index85]
realTest = realWords[index85:]

index70 = int(np.floor(len(fakeWords)*0.7))
index85 = int(np.floor(len(fakeWords)*0.85))
fakeTrain = fakeWords[0:index70]
fakeVali = fakeWords[index70:index85]
fakeTest = fakeWords[index85:]


allWords = []
for s in (realWords + fakeWords):
    for w in s:
        if w not in allWords:
            allWords.append(w)
            

# Construct X matrix from training set
X1 = constX(realTrain,allWords)
X2 = constX(fakeTrain,allWords)
X = np.concatenate((X1,X2),axis=0)
X = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)

# Construct t matrix from training set
N = len(realTrain)+len(fakeTrain)
t = np.zeros((N,1))
for i in range(len(realTrain)):
    t[i,0] = 1
    
for i in range(len(fakeTrain)):
    t[len(realTrain)+i,0] = 0

# construct W for Training Set
W = np.random.normal(0, 0.0001, (1,5833))

# construct X for Validation Set
XVali1 = constX(realVali,allWords)
XVali2 = constX(fakeVali,allWords)
XVali = np.concatenate((XVali1,XVali2),axis=0)
XVali = np.concatenate((XVali,np.ones((XVali.shape[0],1))),axis=1)

# construct t for Validation Set
N = len(realVali)+len(fakeVali)
tVali = np.zeros((N,1))
for i in range(len(realVali)):
    tVali[i,0] = 1
    
for i in range(len(fakeVali)):
    tVali[len(realVali)+i,0] = 0

# construct X for Test Set
XTest1 = constX(realTest,allWords)
XTest2 = constX(fakeTest,allWords)
XTest = np.concatenate((XTest1,XTest2),axis=0)
XTest = np.concatenate((XTest,np.ones((XTest.shape[0],1))),axis=1)

# construct t for Validation Set
N = len(realTest)+len(fakeTest)
tTest = np.zeros((N,1))
for i in range(len(realTest)):
    tTest[i,0] = 1
for i in range(len(fakeTest)):
    tTest[len(realTest)+i,0] = 0



# Training Using Gradient Descent
trainedW,performanceM, performanceMTest,performanceMVali =\
 grad_descent(f, df, X, t, XTest, tTest, XVali, tVali, W, 0.0001,0.0001)
# print the result for Training Set
print("The accuracy rate for training set is:", performance(trainedW,X,t))
# print the result for Validation Set
print("The accuracy rate for validation set is:", performance(trainedW,XVali,tVali))
# print the result for Validation Set
print("The accuracy rate for test set is:", performance(trainedW,XTest,tTest))


# plot the result 
fig = plt.figure()
plt.plot(range(6000),performanceM.tolist(),'r',label = 'Training Set')
plt.plot(range(6000),performanceMTest.tolist(),'b',label = 'Test Set')
plt.plot(range(6000),performanceMVali.tolist(),'g',label = 'Validation Set')
plt.show()

##################
#### Part 6(a)
##################
print('Without Stop Word:')
print('10 words corresponding to top 10 posistive weights:')
WW = trainedW.copy()
WW = WW[0,0:5832]
for i in range(10):
    indx = np.argmax(WW)
    print WW[indx]
    WW[indx] = float('-inf')
    print(allWords[indx])
print('\n')

print('Without Stop Word:')
print('10 words corresponding to top 10 negative weights:')   
WW = trainedW.copy()
WW = WW[0,0:5832]
for i in range(10):
    indx = np.argmin(WW)
    print WW[indx]
    WW[indx] = float('inf')
    print(allWords[indx])
print('\n')
    
##################
#### Part 6(b)
##################
print('With Stop Word:')
print('10 words corresponding to top 10 posistive weights:')
WW = trainedW.copy()
WW = WW[0,0:5832]
i = 0 
while i < 10:
    indx = np.argmax(WW)
    temp = WW[indx]
    WW[indx] = float('-inf')
    if allWords[indx] not in ENGLISH_STOP_WORDS:
        print(allWords[indx])
        print temp
        i += 1
print('\n')

print('With Stop Word:')
print('10 words corresponding to top 10 negative weights:')
WW = trainedW.copy()
WW = WW[0,0:5832]
i = 0 
while i < 10:
    indx = np.argmin(WW)
    temp = WW[indx]
    WW[indx] = float('inf')
    if allWords[indx] not in ENGLISH_STOP_WORDS:
        print(allWords[indx])
        print temp

        i += 1
        
    

