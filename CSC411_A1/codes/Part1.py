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

# Load the file and store in M
M = loadmat("mnist_all.mat")

# Plot the images
for i in range(10):
    plt.figure(i)
    f, axarr = plt.subplots(5,2)
    for j in range(5):
        np.random.seed(0)
        ind = np.random.rand(10) * M["train"+str(i)].shape[0]
        axarr[j,0].imshow(M["train"+str(i)][int(ind[j])].reshape((28,28)), cmap=cm.gray)
        axarr[j,1].imshow(M["train"+str(i)][int(ind[j+1])].reshape((28,28)), cmap=cm.gray)
        axarr[j,0].axes.get_xaxis().set_visible(False)
        axarr[j,0].axes.get_yaxis().set_visible(False)
        axarr[j,1].axes.get_xaxis().set_visible(False)
        axarr[j,1].axes.get_yaxis().set_visible(False)
    plt.show()
        

