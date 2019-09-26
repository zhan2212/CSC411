# Part 10 code

import os 
import argparse
import pickle 

import torch
import torchvision.models as models
import torchvision
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable

import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize

from data_loader import FaceDatasetTL, get_loader


# a list of class names
# from caffe_classes import class_names


act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
act_names = [n.split()[-1].lower() for n in act]


########################################   model for transfer leanring  ###############################


class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

        # extractor_weight_i = []
        # for i in features_weight_i:
        #     if i <= self.layer_idx:
        #         extractor_weight_i.append(i)
        # for i in extractor_weight_i:
        #     self.extractor[i].weight = an_builtin.features[i].weight
        #     self.extractor[i].bias = an_builtin.features[i].bias
            
        # classifier_weight_i = [1, 4, 6]
        # for i in classifier_weight_i:
        #     self.classifier[i].weight = an_builtin.classifier[i].weight
        #     self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, input_size, hidden_size, output_size, num_classes=1000, layer_idx=8):
        super(MyAlexNet, self).__init__()

        self.input_size = 256 * 6 * 6
        self.hidden_size = hidden_size
        self.hidden_size2 = 50
        self.output_size = output_size
        self.layer_idx = layer_idx

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),    # 256 * 13 * 13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),           # 256 * 6 * 6
        )

        # test layer output dim [torch.FloatTensor of size 32x256x3x3]
        self.face_classifier = nn.Sequential(
          nn.Linear(self.input_size, self.hidden_size),   # no input size 
          nn.ReLU(),
          # nn.Linear(self.hidden_size, self.hidden_size2),   
          # nn.ReLU(),
          nn.Linear(self.hidden_size, self.output_size),
          nn.Softmax() 
        )
        
        # self.build_extractor()  # by default is conv4 (layer_idx 8)
        self.load_weights()


    def forward(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        
        x = self.features(x)
        x = x.view(x.size(0), self.input_size)
        x = self.face_classifier(x)
        return x
        # return self.features(x)

    # def build_extractor(self):
    #     # by default layer being extracted is in features 
    #     self.extractor = nn.Sequential(*list(self.features.children())[:self.layer_idx+1])

    def extract_activations(self, x):
        # return self.extractor(x)
        return self.features(x)


########################################   utility functions ###############################

def accuracy(output, y):
    idx1 = torch.max(output, 1)[1]
    # idx2 = torch.max(y, 1)[1]
    acc = torch.sum(torch.eq(idx1, y)).float() / idx1.size()[0]
    return acc 

def validation(model, criterion, data_loader_valid):
    """ evaluate on validation data 
    """
    x, y = next(iter(data_loader_valid))
    x  = Variable(x.float())
    y = Variable(y).long()
    output = model(x)
    loss = criterion(output, y)
    acc = accuracy(output, y)
    print( 'Validatoin loss: %.4f, accuracy: %.4f' % (loss.data[0], acc.data[0]))
    return output, loss, acc 


def test(model, criterion, data_loader_test):
    """ evaluate on test data 
    """
    x, y = next(iter(data_loader_test))
    x  = Variable(x.float())
    y = Variable(y).long()
    output = model(x)
    loss = criterion(output, y)
    acc = accuracy(output, y)
    print( 'Test loss: %.4f, accuracy: %.4f' % (loss.data[0], acc.data[0]))
    return output, loss, acc 


def plot_performance(loss_history, acc_history, path="../images/face_performance_TL.png", display=True):
    plt.figure(figsize=(10, 5))
    # loss curve 
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title("Loss Curve")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    # accuracy curve 
    plt.subplot(1, 2, 2)
    plt.plot(acc_history)
    plt.title("Accuracy Curve")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    # display plot
    plt.grid(True)
    plt.savefig(path)
    if display:
        plt.show()


def train_model(model):
    pass 


def main(config):
    data_loader_train = get_loader(path = config.data_path, 
                             batch_size = config.batch_size, 
                             num_workers = config.num_workers,
                             mode="train", 
                             flatten=False)
    data_loader_valid = get_loader(path = config.data_path, 
                             batch_size = config.batch_size, 
                             num_workers = config.num_workers,
                             mode="validation", 
                             flatten=False)
    data_loader_test = get_loader(path = config.data_path, 
                             batch_size = config.batch_size, 
                             num_workers = config.num_workers,
                             mode="test", 
                             flatten=False)
    data_loader = {"train": data_loader_train,
                   "validation": data_loader_valid,
                   "test": data_loader_test
                   }

    # construct model 
    model = MyAlexNet(config.input_dim, config.hidden_dim, config.output_dim)
    # freeze the feature extractor 
    for param in model.features.parameters():
        param.requires_grad = False 
    # for param in model.extractor.parameters():
    #     param.requires_grad = False 
    # print(model)

    # training the model 
    loss_history = []
    acc_history = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.face_classifier.parameters(), config.learning_rate)

    # for p in model.parameters():
    #     print(p.requires_grad)

    # for k in model.state_dict():
    #     print(k)


    # training loop 
    total_step = len(data_loader_train)
    for epoch in range(config.num_epochs):
        for i, (x, y) in enumerate(data_loader_train):
            x  = Variable(x.float())
            y = Variable(y).long()
            output = model(x)
        #     print(output)
        #     break
        # break
            loss = criterion(output, y)
            loss_history.append(loss.data[0])
            loss.backward()
            optimizer.step()
            acc = accuracy(output, y)
            acc_history.append(acc.data[0])

            if (i+1) % config.log_step == 0:
                print( 'Epoch [%d/%d], Step[%d/%d], loss: %.4f, accuracy: %.4f' 
                      % (epoch+1, config.num_epochs, i+1, total_step, loss.data[0], acc.data[0]))

        # save the model per epoch, only save parameters 
        if (epoch+1) % config.save_freq == 0:
            model_path = os.path.join(config.model_dir, 'modelTL-%d.pkl' %(epoch+1))
            torch.save(model.state_dict(), model_path)

        # on validation set (not used here)
        # x, y = next(iter(self.data_loader_valid))
        # x  = Variable(x.float())
        # y = Variable(y).long()
        # output = model(x)
        # loss = criterion(output, y)
        # acc = accuracy(output, y)
        # print( 'Validation: Epoch [%d/%d], loss: %.4f, accuracy: %.4f' 
        #       % (epoch+1, config.num_epochs, loss.data[0], acc.data[0]))

    # print(output)
    test(model, criterion, data_loader_test)
    plot_performance(loss_history, acc_history) 




########################################   main function   ###############################


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    # hyperparameters
    parser.add_argument("--data_path", type=str, default="../data/datasets.pkl")
    parser.add_argument("--model_dir", type=str, default="modelsTL")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00001)  # 0.001 works well 
    parser.add_argument("--log_step", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=5)


    parser.add_argument("--input_dim", type=int, default=227*227) # 32x32 works well
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--output_dim", type=int, default=6)

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_path", type=str, default="./models/modelTL-200.pkl")


    config = parser.parse_args()
    print(config)
    main(config)