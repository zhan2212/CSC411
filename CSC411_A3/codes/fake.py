from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
import os
import urllib
# Constants
# -----------------------------------------------------------------------------


# Functions used in multiple parts
# -----------------------------------------------------------------------------
def getNews(seed = 0):
    
    fakeFile = open("clean_fake.txt",'r')
    fakeNews = fakeFile.read().splitlines()
    fakeFile.close()
    realFile = open("clean_real.txt",'r')
    realNews = realFile.read().splitlines()
    realFile.close()
    
    np.random.shuffle(fakeNews)
    np.random.shuffle(realNews)
    
    return fakeNews, realNews

def dataSetSplit(fakeNews, realNews,seed = 0):
    
    trainingSet ={'real': realNews[:int(0.7*len(realNews))], 'fake':fakeNews[:int(0.7*len(fakeNews))]} 
    validationSet = {'real':realNews[int(0.7*len(realNews)):int(0.85*len(realNews))], 'fake':fakeNews[int(0.7*len(fakeNews)):int(0.85*len(fakeNews))]}
    testSet = {'real': realNews[int(0.85*len(realNews)):] , 'fake': fakeNews[int(0.85*len(fakeNews)):]}

    return trainingSet, validationSet, testSet

def count(trainingSet, validationSet, testSet):
    trainingCount = {'real':{}, 'fake':{}}
    validationCount = {'real':{}, 'fake':{}}
    testCount = {'real':{}, 'fake':{}}
    
    for rof in ['real','fake']:
        for news in trainingSet[rof]:
            for word in set(news.split(' ')):
                if not word in trainingCount[rof].keys():
                    trainingCount[rof][word] = 1
                else:
                    trainingCount[rof][word] += 1
        
        for news in validationSet[rof]:
            for word in set(news.split(' ')):
                if not word in validationCount[rof].keys():
                    validationCount[rof][word] = 1
                else:
                    validationCount[rof][word] += 1
        
        for news in testSet[rof]:
            for word in set(news.split(' ')):
                if not word in testCount[rof].keys():
                    testCount[rof][word] = 1
                else:
                    testCount[rof][word] += 1

    trainingRealCount = len(trainingSet['real'])
    trainingFakeCount = len(trainingSet['fake'])
    validationRealCount = len(validationSet['real'])
    validationFakeCount = len(validationSet['fake'])
    testRealCount = len(testSet['real'])
    testFakeCount = len(testSet['fake'])
    
    return trainingRealCount,trainingFakeCount,validationRealCount,validationFakeCount,testRealCount,testFakeCount, trainingCount, validationCount, testCount

def probability(trainingRealCount,trainingFakeCount, trainingCount, m, pHat):
    trainingProbability = {'real': {}, 'fake': {}}
    for word in trainingCount['real'].keys():
        trainingProbability['real'][word] = (trainingCount['real'][word]+m*pHat)/float(trainingRealCount+m)
    for word in trainingCount['fake'].keys():
        trainingProbability['fake'][word] = (trainingCount['fake'][word]+m*pHat)/float(trainingFakeCount+m)
    
    realProbability = float(trainingRealCount)/(trainingRealCount+trainingFakeCount)
    fakeProbability = float(trainingFakeCount)/(trainingRealCount+trainingFakeCount)
    
    return trainingProbability, realProbability, fakeProbability

def getAccuracy(trainingProbability,realProbability, fakeProbability, targetSet, m, pHat):
    accuracy = 0
    for rof in ['real', 'fake']:
        for numNews in range(len(targetSet[rof])):
            logProb = 0   
            for word in trainingProbability['real'].keys():
                if word in targetSet[rof][numNews].split(' '):
                    logProb += log(trainingProbability['real'][word])
                else:
                    logProb += log(1-trainingProbability['real'][word])
            for word in targetSet[rof][numNews].split(' '):
                if not word in trainingProbability['real'].keys():
                    logProb += log(pHat)
            logProb += log(realProbability)
            realProb = logProb

            logProb = 0 
            for word in trainingProbability['fake'].keys():
                if word in targetSet[rof][numNews].split(' '):
                    logProb += log(trainingProbability['fake'][word])
                else:
                    logProb += log(1-trainingProbability['fake'][word])
            for word in targetSet[rof][numNews].split(' '):
                if not word in trainingProbability['fake'].keys():
                    logProb += log(pHat)
            logProb += log(fakeProbability)
            fakeProb = logProb
            
            if rof == 'real':
                if realProb > fakeProb:
                    accuracy += 1
            else:
                if realProb < fakeProb:
                    accuracy += 1
                    
    return accuracy / (float(len(targetSet['fake']))+float(len(targetSet['real'])))

# Part 2
# -----------------------------------------------------------------------------
def part2(seed = 0):
    np.random.seed(0)

    fakeNews, realNews = getNews()
    trainingSet, validationSet, testSet = dataSetSplit(fakeNews, realNews)
    trainingRealCount,trainingFakeCount,validationRealCount, \
    validationFakeCount,testRealCount,testFakeCount, trainingCount, \
    validationCount, testCount = count(trainingSet, validationSet, testSet)
    
    
    for i in range(5):
        m = 0.5
        pHat = 0.0005 +i*0.0005
        trainingProbability, realProbability, fakeProbability = \
        probability(trainingRealCount,trainingFakeCount, trainingCount, m,pHat)
    
        print '----'
        print pHat
        print "VALIDATION: ",getAccuracy(trainingProbability,realProbability, fakeProbability, validationSet, m, pHat)
        # print "TEST: ",getAccuracy(trainingProbability,realProbability, fakeProbability, testSet, m, pHat)
        print "Training: ", getAccuracy(trainingProbability,realProbability, fakeProbability, trainingSet, m, pHat)
    
