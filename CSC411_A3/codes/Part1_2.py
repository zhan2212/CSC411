from __future__ import division
import numpy as np
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def NBCount(realTrain,fakeTrain,m,p):
    '''
    This function is to perform the count step of the Naive Bayes algorithm, 
    calculate P(Xi|Y) and P(Y) and return them as result
    '''
    countReal = len(realTrain)
    countFake = len(fakeTrain)
    N = countReal + countFake # total number of news
    
    allWords = []
    for s in (realTrain + fakeTrain):
        for w in s:
            if w not in allWords:
                allWords.append(w) # extract all the words that appear in TrainSet          
 
    P_AllWordsReal = np.zeros(len(allWords)) # Store P(Xi|Y = Real)
    P_AllWordsFake = np.zeros(len(allWords)) # Store P(Xi|Y = Fake)
 
    for i in range(len(allWords)):
        countXiReal = 0
        for s in realTrain:
            if allWords[i] in s:
                countXiReal += 1
        P_AllWordsReal[i] = float(countXiReal + m*p) /float(countReal+ m)
    
    for i in range(len(allWords)):
        countXiFake = 0
        for s in fakeTrain:
            if allWords[i] in s:
                countXiFake += 1
        P_AllWordsFake[i] = float(countXiFake + m*p)/float(countFake + m)
        
    return (allWords, P_AllWordsReal, countReal/N, P_AllWordsFake, countFake/N)
        
def NBPred(news,allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake):
    '''
    This function is to make the predection step of the Navie Bayes algorithm,
    calculate P(Y|X) and return the argmax_Y P(Y|X).
    '''    
    P_XReal, P_XFake = 0,0
    
    for i in range(len(allWords)):
        if allWords[i] in news:
            P_XReal += float(np.log(P_AllWordsReal[i]))
            P_XFake += float(np.log(P_AllWordsFake[i]))
        else:
            P_XReal += float(np.log((1.0 - P_AllWordsReal[i])))
            P_XFake += float(np.log((1.0 - P_AllWordsFake[i])))
    
    Pr = np.exp(P_XReal)*P_real
    Pf = np.exp(P_XFake)*P_fake
    
    P_realX = float(Pr)/float(Pr+Pf)
    P_fakeX = 1-P_realX
    
    if P_realX > P_fakeX:
        return 'Real'
    elif P_realX < P_fakeX:
        return 'Fake'
    else:
        return '???'
    
def Performance(realSet,fakeSet, allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake):
    '''
    This function is to calculate 
    '''
    count = 0
    for s in realSet:
        if NBPred(s,allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake) == 'Real':
            count += 1
    for s in fakeSet:
        if NBPred(s,allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake) == 'Fake':
            count += 1
    N = len(realSet)+len(fakeSet)
    return count/N
  
##################
#### Part 1
##################
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
np.random.seed(1)
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

# count the database
allWords, P_AllWordsReal, P_real, P_AllWordsFake, P_fake = NBCount(realTrain,fakeTrain,0.1,0.0001)

print('3 useful key words for real news')
RealP2 = P_AllWordsReal.copy()
for i in range(3):
     indx = np.argmax(RealP2)
     RealP2[indx] = float('-inf')
     print(allWords[indx])
print('\n')

print('3 useful key words for fake news')
FakeP2 = P_AllWordsFake.copy()
for i in range(3):
     indx = np.argmax(FakeP2)
     FakeP2[indx] = float('-inf')
     print(allWords[indx])
print('\n')



##################
#### Part 2
##################
# m and p paramaters
m = 0.1
p = 0.0001

# count the database
allWords, P_AllWordsReal, P_real, P_AllWordsFake, P_fake = NBCount(realTrain,fakeTrain,m,p)

# Make predection and evaluate the performance
p = Performance(realTrain,fakeTrain, allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake)
print 'The accuracy of the Train Set:', p
p = Performance(realVali,fakeVali, allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake)
print 'The accuracy of the Validation Set:', p
p = Performance(realTest,fakeTest, allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake)
print 'The accuracy of the Test Set:', p



        