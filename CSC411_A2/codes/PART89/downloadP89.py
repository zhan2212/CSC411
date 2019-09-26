from scipy.misc import imsave
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image

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



def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result
    
dirpath = os.getcwd()

try:
    os.makedirs(dirpath+"/cropped/")
    os.makedirs(dirpath+"/uncropped/")
except:
    print("")

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

act = list(set([a.split("\t")[0] for a in open(dirpath + "/facescrub_actors.txt").readlines()]))
#act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open(dirpath + "/facescrub_actors.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], dirpath + "/uncropped/"+filename), {}, 5)
            if not os.path.isfile(dirpath + "/uncropped/"+filename):
                continue
            print filename
            i += 1
            
            try:
                uncropped = imread(dirpath + "/uncropped/"+filename)
                curr_hash = line.split()[-1]
                if curr_hash[:-1] != hashlib.sha256(uncropped).hexdigest():
                    continue
                uncropped = rgb2gray(uncropped)
                bbox = line.split()[5]
                x1 = int(bbox.split(',')[0])
                y1 = int(bbox.split(',')[1])
                x2 = int(bbox.split(',')[2])
                y2 = int(bbox.split(',')[3])
                cropped = uncropped[y1:y2, x1:x2]
                img = imresize(cropped,(32,32))
                imsave(dirpath + "/cropped/"+filename,img,cmap = cm.gray)
                print("x")
                
            except:
                print("y")
                continue
                
            


actress = list(set([a.split("\t")[0] for a in open(dirpath + "/facescrub_actresses.txt").readlines()]))

for a in actress:
    name = a.split()[1].lower()
    i = 0
    for line in open(dirpath + "/facescrub_actresses.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], dirpath+"/uncropped/"+filename), {}, 5)
            if not os.path.isfile(dirpath + "/uncropped/"+filename):
                continue   
            
            print filename
            i += 1
            
            try:
                uncropped = imread(dirpath + "/uncropped/"+filename)
                uncropped = rgb2gray(uncropped)
                bbox = line.split()[5]
                x1 = int(bbox.split(',')[0])
                y1 = int(bbox.split(',')[1])
                x2 = int(bbox.split(',')[2])
                y2 = int(bbox.split(',')[3])
                cropped = uncropped[y1:y2, x1:x2]
                img = imresize(cropped,(32,32))
                
                imsave(dirpath + "/cropped/"+filename,img,cmap = cm.gray)
                print("x")
            except Exception as e:
                print("y")
                continue