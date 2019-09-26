from __future__ import division
import numpy as np
import os
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])

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
        
def NBWord(word,allWordsT,P_AllWordsReal,P_real, P_AllWordsFake,P_fake):
    '''
    This function is used to calculate normalized P(Xi|Y)P(Y), which can
    represent 'presence/absense most strongly predicts that the news is real'
    and 'presence/abasense most strongly predicts that the news is fake.'
    '''
    for i in range(len(allWordsT)):
        if allWordsT[i] == word:
            PrP = P_AllWordsReal[i]
            PfP = P_AllWordsFake[i]
            PrA = 1.0 - PrP
            PfA = 1.0 - PfP
            break
    PrP *= P_real
    PfP *= P_fake
    PrA *= P_real
    PfA *= P_fake
    
    return PrP/(PrP+PfP), PfP/(PrP+PfP),PrA/(PrA+PfA), PfA/(PrA+PfA)
    
    
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

# m and p paramaters
m = 0.1
p = 0.0001

# count the database
allWords, P_AllWordsReal, P_real, P_AllWordsFake, P_fake = NBCount(realTrain,fakeTrain,m,p)
 
# store the result of NBWord() for later comparison
RealP = np.array([])
FakeP = np.array([])
RealA = np.array([])
FakeA = np.array([])
for i in range(len(allWords)):
    if i%100 == 0:
        print(i)
    PrP,PfP,PrA,PfA = NBWord(allWords[i],allWords,P_AllWordsReal,P_real, P_AllWordsFake,P_fake)
    RealP = np.append(RealP,PrP)
    FakeP = np.append(FakeP,PfP)
    RealA = np.append(RealA,PrA)
    FakeA = np.append(FakeA,PfA)
        
##################
#### Part 3(a)
##################
print('Without Stop Word:')
print('10 words whose presence most strongly predicts that the news is real:')
# largest 10 for Real
RealP2 = RealP.copy()
for i in range(10):
     indx = np.argmax(RealP2)
     RealP2[indx] = float('-inf')
     print(allWords[indx])
     print(P_AllWordsReal[indx])

print('\n')


print('Without Stop Word:')    
print('10 words whose absense most strongly predicts that the news is real:')
RealA2 = RealA.copy()
for i in range(10):
     indx = np.argmax(RealA2)
     RealA2[indx] = float('-inf')
     print(allWords[indx])
     print(P_AllWordsReal[indx])

print('\n')


# largest 10 for Fake
print('Without Stop Word:') 
print('10 words whose presence most strongly predicts that the news is fake:')
FakeP2 = FakeP.copy()
for i in range(10):
     indx = np.argmax(FakeP2)
     FakeP2[indx] = float('-inf')
     print(allWords[indx])
print('\n')

print('Without Stop Word:') 
print('10 words whose absense most strongly predicts that the news is fake:')
FakeA2 = FakeA.copy()
for i in range(10):
     indx = np.argmax(FakeA2)
     FakeA2[indx] = float('-inf')
     print(allWords[indx])
print('\n')


##################
#### Part 3(b)
##################
print('With Stop Word:')
print('10 words whose presence most strongly predicts that the news is real:')
RealP2 = RealP.copy()
i = 0
while i < 10:
    indx = np.argmax(RealP2)
    RealP2[indx] = float('-inf')
    if not allWords[indx] in ENGLISH_STOP_WORDS:
        print(allWords[indx])
        i+=1
print('\n')

print('With Stop Word:')
print('10 words whose absense most strongly predicts that the news is real:')
RealA2 = RealA.copy()
i = 0
while i < 10:
    indx = np.argmax(RealA2)
    RealA2[indx] = float('-inf')
    if not allWords[indx] in ENGLISH_STOP_WORDS:
        print(allWords[indx])
        i+=1
print('\n')

print('With Stop Word:')
print('10 words whose presence most strongly predicts that the news is fake:')
FakeP2 = FakeP.copy()
i = 0
while i < 10:
    indx = np.argmax(FakeP2)
    FakeP2[indx] = float('-inf')
    if not allWords[indx] in ENGLISH_STOP_WORDS:
        print(allWords[indx])
        i+=1
print('\n')

print('With Stop Word:')
print('10 words whose absense most strongly predicts that the news is fake:')
FakeA2 = FakeA.copy()
i = 0
while i < 10:
    indx = np.argmax(FakeA2)
    FakeA2[indx] = float('-inf')
    if not allWords[indx] in ENGLISH_STOP_WORDS:
        print(allWords[indx])
        i+=1
print('\n')
