import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import os

# a list of class names
#from caffe_classes import class_names

# We modify the torchvision implementation so that the features
# after the final pooling layer is easily accessible by calling
#       net.features(...)
# If you would like to use other layer features, you will need to
# make similar modifications.
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        #classifier_weight_i = [1, 4, 6]
        #for i in classifier_weight_i:
        #    self.classifier[i].weight = an_builtin.classifier[i].weight
        #   self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 50),   
            nn.ReLU(),
            nn.Linear(50, 6),
            nn.Softmax() 
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
     
    
def get_set(Set):
    '''
    This function generates the x matrix from the pictures in training set 
    '''
    x = np.zeros((0,3,227,227))
    t = np.zeros((0,3,227,227))
    for i in range(6):
        xAdd = np.zeros((len(Set[i]),3,227,227))
        tAdd = np.zeros((len(Set[i]),3,227,227))
        for j in range(len(Set[i])):
            filename = Set[i][j]
            
            im = imread(os.getcwd() + "/croppedA2/"+filename)[:,:,:3]
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)           
            
            xAdd[j,:,:,:] = im
            tAdd[j,0,0,0] = i
        x = np.vstack((x, xAdd))
        t = np.vstack((t,tAdd))
    return x,t


torch.manual_seed(1) 
# Classify the pictures for different people and store the file names in 
# nameList. The index of the list corresponds to different poeple.
names = ["bracco","gilpin","harmon","baldwin","hader","carell"]
dirs = os.listdir(os.getcwd() + "/croppedA2/") 
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

for i in range(6):
    np.random.seed(1)
    np.random.shuffle(nameList[i])
    trainSet[i] = nameList[i][30:80]
    valiSet[i] = nameList[i][20:30]
    testSet[i] = nameList[i][0:20]
    
# construct x and t matrix from training set
x_train,t_train = get_set(trainSet)
x_test,t_test = get_set(testSet)
x_vali,t_vali = get_set(valiSet)



########
# set model
model = MyAlexNet()

# fix the weights for features
for param in model.features.parameters():
        param.requires_grad = False 

# define the type of variable
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

# load the data
trainDataset = np.concatenate((x_train,t_train),axis=1)
dataloader = DataLoader(trainDataset, batch_size=50, shuffle=True, num_workers=2)

# set learning rate and optimizer
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.classifier.parameters(), learning_rate)

# set loss function
loss_fn = torch.nn.CrossEntropyLoss()

# array to store performance
trainPerformance = np.array([])
valiPerformance = np.array([])
testPerformance = np.array([])


for i in range(40):
    print "iertration:", i
    for j,sample in enumerate(dataloader):
        sample = sample.numpy()
        x = sample[:,0:3,:,:]
        t = sample[:,3:6,:,:]
        t = t[:,0,0,0]
        
        x1 = Variable(torch.from_numpy(x), requires_grad=False).type(dtype_float)
        t_classes1 = Variable(torch.from_numpy(t), requires_grad=False).type(dtype_long)

        
        t_pred1 = model(x1)
        loss = loss_fn(t_pred1, t_classes1)
        print "Loss:", loss
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                           # make a step
                           
        # save the performance for training, validation and test set
        x = Variable(torch.from_numpy(x_train), requires_grad=False).type(dtype_float)
        t_pred = model(x).data.numpy()
        trainPerformance = np.append(trainPerformance, np.mean(np.argmax(t_pred, 1) == t_train[:,0,0,0]))
         
        x = Variable(torch.from_numpy(x_vali), requires_grad=False).type(dtype_float)
        t_pred = model(x).data.numpy()
        valiPerformance = np.append(valiPerformance, np.mean(np.argmax(t_pred, 1) == t_vali[:,0,0,0]))
        
        x = Variable(torch.from_numpy(x_test), requires_grad=False).type(dtype_float)
        t_pred = model(x).data.numpy()
        testPerformance = np.append(testPerformance, np.mean(np.argmax(t_pred, 1) == t_test[:,0,0,0]))      
        
# plot the learning curve    
t = np.arange(0,trainPerformance.shape[0])
plt.plot(t, trainPerformance, 'r', t, valiPerformance, 'b', t, testPerformance, 'g')
plt.show()

   
# Performance on Training Set             
x = Variable(torch.from_numpy(x_train), requires_grad=False).type(dtype_float)
t_pred = model(x).data.numpy()
print "Accuracy on the Training Set  is:", np.mean(np.argmax(t_pred, 1) == t_train[:,0,0,0])

# Performance on Validation Set                
x = Variable(torch.from_numpy(x_vali), requires_grad=False).type(dtype_float)
t_pred = model(x).data.numpy()
print "Accuracy on the Validation Set  is:", np.mean(np.argmax(t_pred, 1) == t_vali[:,0,0,0])

# Performance on Test Set                    
x = Variable(torch.from_numpy(x_test), requires_grad=False).type(dtype_float)
t_pred = model(x).data.numpy()
print "Accuracy on the Test Set  is:", np.mean(np.argmax(t_pred, 1) == t_test[:,0,0,0])
  
    
    
    