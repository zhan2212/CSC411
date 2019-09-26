from __future__ import division
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import os  
from sklearn.tree import export_graphviz
import numpy as np
import matplotlib.pyplot as plt
def constX(Set,allWords):
    N = len(Set)
    K = len(allWords)
    X = np.zeros((N,K))
    for i in range(len(Set)):
        for j in range(len(allWords)):
            if allWords[j] in Set[i]:
                X[i,j] = 1
    return X

currPath = os.getcwd()
realPath = currPath + '/clean_real.txt'
fakePath = currPath + '/clean_fake.txt'


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


allWords = []
for s in (realWords + fakeWords):
    for w in s:
        if w not in allWords:
            allWords.append(w)
            

# Construct X matrix from training set
X1 = constX(realTrain,allWords)
X2 = constX(fakeTrain,allWords)
X = np.concatenate((X1,X2),axis=0)

# Construct t matrix from training set
N = len(realTrain)+len(fakeTrain)
t = np.zeros((N,1))
for i in range(len(realTrain)):
    t[i,0] = 1
    
for i in range(len(fakeTrain)):
    t[len(realTrain)+i,0] = 0

# construct W for Training Set
np.random.seed(0)
W = np.random.normal(0, 0.0001, (1,5833))

# construct X for Validation Set
XVali1 = constX(realVali,allWords)
XVali2 = constX(fakeVali,allWords)
XVali = np.concatenate((XVali1,XVali2),axis=0)

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

# construct t for Validation Set
N = len(realTest)+len(fakeTest)
tTest = np.zeros((N,1))
for i in range(len(realTest)):
    tTest[i,0] = 1
for i in range(len(fakeTest)):
    tTest[len(realTest)+i,0] = 0

data_feature_name = allWords
data_classe_name = ['real','fake']
path1 = currPath + '/tree/tree.dot'
path2 = currPath + '/tree/tree.png'

train = np.array([])
vali = np.array([])
test = np.array([])
depth = np.array([])

# Build the tree and test using 10 different depths
for i in range(20): #,25,30,35,40,45,50,55,60,65]:
    d = 5 + i*5
    clf = DecisionTreeClassifier(random_state=0, max_depth = d,criterion = 'entropy')
    clf.fit(X,t)
    depth= np.append(depth,d)
    train = np.append(train,clf.score(X,t))
    vali = np.append(vali,clf.score(XVali,tVali))
    test = np.append(test,clf.score(XTest,tTest))
    #print('max_depth:',d)
    #print(clf.score(X,t))
    #print(clf.score(XVali,tVali))
    #rint(clf.score(XTest,tTest))
    
    
# plot the result 
fig = plt.figure()
plt.plot(depth.tolist(),train.tolist(),'r',label = 'Training Set')
plt.plot(depth.tolist(),test.tolist(),'b',label = 'Test Set')
plt.plot(depth.tolist(),vali.tolist(),'g',label = 'Validation Set')
plt.legend(loc='top left')
plt.show()
 

    graph = export_graphviz(clf, feature_names = data_feature_name, class_names = data_classe_name,
                         out_file = None, filled = True,
                         rounded = True, special_characters=True)
    
    #os.system('dot -Tpng' + path1 + '-o' + path2)
    
##################
#### Part 8(a)
##################
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
        P_AllWordsReal[i] = (countXiReal + m*p)/(countReal+ m)
    
    for i in range(len(allWords)):
        countXiFake = 0
        for s in fakeTrain:
            if allWords[i] in s:
                countXiFake += 1
        P_AllWordsFake[i] = (countXiFake + m*p)/(countFake + m)
        
    return (allWords, P_AllWordsReal, countReal/N, P_AllWordsFake, countFake/N)    

def H(realTrain,fakeTrain):
    '''
    This function computes the entropy of the label, that is H(T)
    '''
    Pr = len(realTrain)/(len(realTrain)+len(fakeTrain))
    Pf = 1.0 - Pr
    return - Pr*np.log2(Pr) - Pf*np.log2(Pf)

def NBWord(word,allWordsT,P_AllWordsReal,P_real, P_AllWordsFake,P_fake):
    '''
    return: P(T=real|xi=1), P(T=fake|xi=1), P(T=real|xi=0), P(T=fake|xi=0)   
    '''
    for i in range(len(allWordsT)):
        if allWordsT[i] == word:
            Pxi_r = P_AllWordsReal[i]
            Pxi_f = P_AllWordsFake[i]
            break
    Pr = Pxi_r * P_real
    Pf = Pxi_f * P_fake
    Prnot = (1.0 - Pxi_r) * P_real
    Pfnot = (1.0 - Pxi_f) * P_fake
    
    return Pr/(Pr+Pf), Pf/(Pr+Pf), Prnot/(Prnot+Pfnot), Pfnot/(Prnot+Pfnot)

def condH(xi,realTrain,fakeTrain,Pr_xi1,Pf_xi1,Pr_xi0,Pf_xi0):
    '''
    H(T|xi)
    '''
    Pxi = 0
    for news in (realTrain+fakeTrain):
        if xi in news:
            Pxi += 1
    Pxi = Pxi/ (len(realTrain)+len(fakeTrain))
    result = Pxi*(- Pr_xi1*np.log2(Pr_xi1) - Pf_xi1*np.log2(Pf_xi1))
    result += (1.0-Pxi)*(- Pr_xi0*np.log2(Pr_xi0) - Pf_xi0*np.log2(Pf_xi0))
    
    return result

# m and p paramaters
m = 0.1
p = 0.0001
# count the database
xi =allWords[clf.tree_.feature[0]]
allWords, P_AllWordsReal, P_real, P_AllWordsFake, P_fake = NBCount(realTrain,fakeTrain,m,p)

Pr_xi1,Pf_xi1,Pr_xi0,Pf_xi0 = NBWord(xi,allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake)
I = H(realTrain,fakeTrain) - condH(xi,realTrain,fakeTrain,Pr_xi1,Pf_xi1,Pr_xi0,Pf_xi0)
print xi,I

##################
#### Part 8(b)
##################
for i in range(1,10):
    xi =allWords[clf.tree_.feature[i]]
    Pr_xi1,Pf_xi1,Pr_xi0,Pf_xi0 = NBWord(xi,allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake)
    I = H(realTrain,fakeTrain) - condH(xi,realTrain,fakeTrain,Pr_xi1,Pf_xi1,Pr_xi0,Pf_xi0)
    print i,xi,I      
    
    
    
    
    
    


    
    
