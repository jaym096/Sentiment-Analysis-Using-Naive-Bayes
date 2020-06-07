import re
import random

def seperate(raw_data):
    data = []
    t_class = []
    for i in range(len(raw_data)):
        sentence, t = raw_data[i].split("\t")
        data.append(sentence.lower())
        t_class.append(int(t))
    return data, t_class
    
def dataPreprocessing(data):
    #for preprocessing and refining the data 
    temp_data = []
    for i in range(len(data)):
        sent = " ".join(re.findall("[a-zA-Z]+", data[i])).lower()
        splitter = sent.split(" ")
        temp_data.append(splitter)
    return temp_data

def createTuple(data, t_class):
    temp_data = []
    for i in range(len(t_class)):
        tup = (data[i],t_class[i])
        temp_data.append(tup)
    return temp_data

def divideClass(data):
    """Divide data set by classes"""
    data_0 = []
    data_1 = []
    for i in range(len(data)):
        if(data[i][1] == 1):
            data_1.append(data[i])
        if(data[i][1] == 0):
            data_0.append(data[i])
    return data_0, data_1

def function2_kfolds(tup_data_0, tup_data_1, no_folds):
    """"divide class data set into 10 folds"""
    f_data_0 = []
    f_data_1 = []
    temp_data0 = []
    temp_data1 = []
    size = len(tup_data_0)/no_folds
    for i in range(len(tup_data_0)):
        if((i%size==0) and (i>0)):
            f_data_0.append(temp_data0)
            f_data_1.append(temp_data1)
            temp_data0 = []
            temp_data1 = []
            temp_data0.append(tup_data_0[i])
            temp_data1.append(tup_data_1[i])
        else:
            temp_data0.append(tup_data_0[i])
            temp_data1.append(tup_data_1[i])
    f_data_0.append(temp_data0)
    f_data_1.append(temp_data1)
    return f_data_0, f_data_1
    
def generateFolds(ff_data_0, ff_data_1):
    """Permute and merge"""
    data = []
    for i in range(len(ff_data_0)):
        #n = (len(ff_data_0)-1) - i
        #merge the first element of one class with the last of others
        temp = ff_data_0[i] + ff_data_1[i]
        random.shuffle(temp)
        #temp = random.(temp, len(temp))
        data.append(temp)
    #shuffle the folds
    r_data = random.sample(data, len(data))
    #r_data = data
    return r_data

def getStratifiedKfold(raw_data):
    no_folds = 10
    #seperate statements and class
    temp_data1, t_class = seperate(raw_data)
    #clean the data
    temp_data2 = dataPreprocessing(temp_data1)
    #create a list of tuple which has document and it's class
    tup_data = createTuple(temp_data2,t_class)
    #divide the data into k folds
    tup_data_0, tup_data_1 = divideClass(tup_data)
    ff_data_0, ff_data_1 = function2_kfolds(tup_data_0, tup_data_1, no_folds)
    #generate stratified kfold dataset
    refined_data = generateFolds(ff_data_0, ff_data_1)
    
    return refined_data

#if __name__ == "__main__":
#    raw_data = getData("imdb_labelled.txt")
#    Kfold_data = getStratifiedKfold(raw_data)
#    #data = dataPreprocessing(raw_data)