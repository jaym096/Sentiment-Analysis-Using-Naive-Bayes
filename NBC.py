import numpy as np

def maxLofclass(train_y):
    """calculates and returns Maximum likelihood of a class"""
    count_0 = 0
    count_1 = 0
    for i in range(len(train_y)):
        if(train_y[i]==1):
            count_1 += 1
        if(train_y[i]==0):
            count_0 += 1
    total = len(train_y)
    maxl_0 = count_0/total
    maxl_1 = count_1/total
    maxL = [float(maxl_0), float(maxl_1)]
    return maxL

def getTokenCount(train_x,train_y):
    """count all the words in a given class"""
    count_0 = 0
    count_1 = 0
    for i in range(len(train_y)):
        #get token count for class 1
        if(train_y[i]==1):
            count_1 += (len(train_x[i]))
        
        #get token count for class 0
        if(train_y[i]==0):
            count_0 += len(train_x[i])
    
    return count_0, count_1

def getVocab(train_x):
    """generates a vocabulary for the data set"""
    vo = []
    for i in range(len(train_x)):
        statement = train_x[i]
        for word in statement:
            if word not in vo:
                vo.append(word)
    return vo

def generateClassVocab(train_x,train_y):
    """generate dictionary of words with their counts for each class"""
    vocab_0 = {}
    vocab_1 = {}
    for i in range(len(train_y)):
        #if class is 1
        if(train_y[i]==1):
            for j in range(len(train_x[i])):
                word = train_x[i][j]
                if(word not in vocab_1):
                    vocab_1[word] = 1
                else:
                    counter = vocab_1.get(word)
                    counter += 1
                    vocab_1[word] = counter
        #if class is 0
        if(train_y[i]==0):
            for j in range(len(train_x[i])):
                word = train_x[i][j]
                if(word not in vocab_0):
                    vocab_0[word] = 1
                else:
                    counter = vocab_0.get(word)
                    counter += 1
                    vocab_0[word] = counter
    return vocab_0, vocab_1

def wordGivenCount(voabulary, train_x, train_y):
    wgc_0 = {}
    wgc_1 = {}
    vocab_0, vocab_1 = generateClassVocab(train_x,train_y)
    for word in voabulary:
        #for class 0
        if word in vocab_0:
            wgc_0[word] = vocab_0[word]
        if word not in vocab_0:
            wgc_0[word] = 0
            
        #for class 1
        if word in vocab_1:
            wgc_1[word] = vocab_1[word]
        if word not in vocab_1:
            wgc_1[word] = 0
    return wgc_0, wgc_1
    
    
def MAP_estimate_of_tokens(vocabulary, train_x, train_y, m):
    """creates a dictionary for maxL of tokens"""
    map_0 = {}
    map_1 = {}
    
    #get word given count
    wgc_0, wgc_1 = wordGivenCount(vocabulary, train_x, train_y)
    
    #Token counts
    n_0,n_1 = getTokenCount(train_x,train_y)
    
    #no.of words in vocabulary
    V = len(vocabulary)

    for word in vocabulary:
        #for class 0
        mp1 = (wgc_0[word] + m)/(n_0 + (m * V))
        map_0[word] = mp1
        
        #for class 1
        mp2 = (wgc_1[word] + m)/(n_1 + (m * V))
        map_1[word] = mp2
    return map_0, map_1

def calAccuracy(test_y,pred_list):
    '''calculates and returns accuracy'''
    count = 0
    for i in range(len(test_y)):
        if(pred_list[i] == test_y[i]):
            count += 1
    accuracy = count/len(test_y)
    return accuracy
    
def calScore(statement,maxlC,maxl,vocabulary):
    """calculates and returns the score for a class"""
    score = np.log(maxlC)
    for word in statement:
        #check if word present in training set
        if word not in vocabulary:
            pass
        else:
            if(maxl[word]==0.0):
                score += -999
            else:
                score += np.log(maxl[word])
    return score

def predictMAP(train_x, train_y, test_x, test_y, m):
    '''prediction using Maximum Likelihood'''
    #based on the value of m, we can use this function for MLE(m=0) as well

    #get the maxL of class
    maxlC = maxLofclass(train_y)
    
    #generate a vocabulary
    vocabulary = getVocab(train_x)
    
    #get MAP estimate of the tokens
    map_0, map_1 = MAP_estimate_of_tokens(vocabulary, train_x, train_y, m)
    
    #calculate score for each class
    prediction = []
    for i in range(len(test_y)):
        p_class0 = calScore(test_x[i],maxlC[0],map_0,vocabulary)
        p_class1 = calScore(test_x[i],maxlC[1],map_1,vocabulary)
        if(p_class0 > p_class1):
            prediction.append(0)
        if(p_class1 > p_class0):
            prediction.append(1)
        if(p_class1 == p_class0):
            prediction.append(1)
    #calculating Accuracy
    accu = calAccuracy(test_y,prediction)
    return accu