#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:38:25 2018

@author: michaelkwok
"""

from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imread
from torch.utils.data import Dataset, DataLoader

from scipy.io import loadmat



#%matplotlib inline  

#M = loadmat("mnist_all.mat")


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255.



def get_set(Set):
    '''
    This function generates the x matrix from the pictures in training set 
    '''
    x = np.zeros((0,1024))
    t = np.zeros((0,6))
    for i in range(6):
        xAdd = np.zeros((len(Set[i]),1024))
        tAdd = np.zeros((len(Set[i]),6))
        for j in range(len(Set[i])):
            filename = Set[i][j]
            img = imread(os.getcwd() + "/cropped/"+filename)
            img = rgb2gray(img)
            imgReshaped = img.reshape((1,1024))
            xAdd[j,:] = imgReshaped
            tAdd[j,i] = 1
        x = np.vstack( (x, xAdd))
        t = np.vstack((t,tAdd))
    return x,t


torch.manual_seed(1) 
# Classify the pictures for different people and store the file names in 
# nameList. The index of the list corresponds to different poeple.
names = ["bracco","gilpin","harmon","baldwin","hader","carell"]
dirs = os.listdir(os.getcwd() + "/cropped/") 
nameList = [[],[],[],[],[],[]]
for i in range(6):
    act = names[i]
    for filename in dirs:
        if act in filename:
            nameList[i].append(filename)

# Randomly select the sample pictures for training set, calidation set and 
# test set. 
trainSet = [[],[],[],[],[],[]]
valiSet = [[],[],[],[],[],[]]
testSet = [[],[],[],[],[],[]]
numSample = 0
for i in range(6):
    np.random.seed(1)
    np.random.shuffle(nameList[i])
    size = len(nameList[i])
    trainSet[i] = nameList[i][30:size]
    valiSet[i] = nameList[i][20:30]
    testSet[i] = nameList[i][0:20]
    numSample += size
# construct x and t matrix from training set
x_train,t_train = get_set(trainSet)
x_test,t_test = get_set(testSet)
x_vali,t_vali = get_set(valiSet)

# set dimension
dim_x = 1024
dim_h = 40
dim_out = 6

# define the type of variable
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

# load the data
trainDataset = np.concatenate((x_train,t_train),axis=1)
dataloader = DataLoader(trainDataset, batch_size=200, shuffle=True, num_workers=0)

################################################################################
#Subsample the training set for faster training

# train_idx = np.random.permutation(range(x_train.shape[0]))[:1000]
# x = Variable(torch.from_numpy(x_train[train_idx]), requires_grad=False).type(dtype_float)
# t_classes = Variable(torch.from_numpy(np.argmax(t_train[train_idx], 1)), requires_grad=False).type(dtype_long)
#################################################################################


# set model
model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
    torch.nn.Softmax(),
)

# set loss function
loss_fn = torch.nn.CrossEntropyLoss()

# set learning rate and optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

# array to store performance
trainPerformance = np.array([])
valiPerformance = np.array([])
testPerformance = np.array([])

for i in range(2500):
    if i % 500 == 0:
        print "iertration:", i
    for j,sample in enumerate(dataloader):
        x1 = Variable(sample[:,0:1024], requires_grad=False).type(dtype_float)
        t_classes1 = sample[:,1024:1030].numpy()
        t_classes1 = Variable(torch.from_numpy(np.argmax(t_classes1, 1)), requires_grad=False).type(dtype_long)

        
        t_pred1 = model(x1)
        loss = loss_fn(t_pred1, t_classes1)
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                           # make a step
                           
        # save the performance for training, validation and test set
        x = Variable(torch.from_numpy(x_train), requires_grad=False).type(dtype_float)
        t_pred = model(x).data.numpy()
        trainPerformance = np.append(trainPerformance, np.mean(np.argmax(t_pred, 1) == np.argmax(t_train, 1)))
         
        x = Variable(torch.from_numpy(x_vali), requires_grad=False).type(dtype_float)
        t_pred = model(x).data.numpy()
        valiPerformance = np.append(valiPerformance, np.mean(np.argmax(t_pred, 1) == np.argmax(t_vali, 1)))
        
        x = Variable(torch.from_numpy(x_test), requires_grad=False).type(dtype_float)
        t_pred = model(x).data.numpy()
        testPerformance = np.append(testPerformance, np.mean(np.argmax(t_pred, 1) == np.argmax(t_test, 1)))      
        
# plot the learning curve    
t = np.arange(0,trainPerformance.shape[0])
#plt.plot(t, trainPerformance, 'r', t, valiPerformance, 'b', t, testPerformance, 'g')
plt.plot(t,trainPerformance,'r', label = "Train Performance")
plt.plot(t,valiPerformance,'b', label = "Validation Performance")
plt.plot(t,testPerformance,'g', label = "Test Performance")
plt.legend(loc = "lower right")
plt.xlabel("Iterations") 
plt.ylabel("Accuracy")
plt.show()
                   
                       
# Performance on Training Set             
x = Variable(torch.from_numpy(x_train), requires_grad=False).type(dtype_float)
t_pred = model(x).data.numpy()
print "Accuracy on the Training Set  is:", np.mean(np.argmax(t_pred, 1) == np.argmax(t_train, 1))

# Performance on Validation Set                
x = Variable(torch.from_numpy(x_vali), requires_grad=False).type(dtype_float)
t_pred = model(x).data.numpy()
print "Accuracy on the Validation Set  is:", np.mean(np.argmax(t_pred, 1) == np.argmax(t_vali, 1))

# Performance on Test Set                    
x = Variable(torch.from_numpy(x_test), requires_grad=False).type(dtype_float)
t_pred = model(x).data.numpy()
print "Accuracy on the Test Set  is:", np.mean(np.argmax(t_pred, 1) == np.argmax(t_test, 1))

plt.imshow(model[0].weight.data.numpy()[0,:].reshape((32,32)),cmap=plt.cm.coolwarm)
plt.show()