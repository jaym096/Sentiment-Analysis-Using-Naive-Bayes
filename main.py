#system packages
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
#user defined modules
import generateKfolds as gkf
import train_test
import NBC

def getData(filename):
    #read the file and get a list of all lines
    with open(filename, "r") as f:
        raw_data = f.read().split("\n")
    raw_data = raw_data[:-1]
    return raw_data

def calAccuracy(accuracy_m0, accuracy_m1):
    temp_acc_m0 = []
    temp_acc_m1 = []
    for i in range(len(accuracy_m0)):
        t_ai_0 = []
        t_ai_1 = []
        for j in range(len(accuracy_m0)):
            t_ai_0.append(accuracy_m0[j][i])
            t_ai_1.append(accuracy_m1[j][i])
        temp_acc_m0.append(t_ai_0)
        temp_acc_m1.append(t_ai_1)
    for i in range(len(temp_acc_m0)):
        avg0 = np.average(temp_acc_m0[i])
        avg1 = np.average(temp_acc_m1[i])
        temp_acc_m0[i] = avg0
        temp_acc_m1[i] = avg1
    return temp_acc_m0, temp_acc_m1

def Experiment_1(skfold_data):
    print("Running experiment 1...")
    accuracy_m0 = []
    accuracy_m1 = []
    
    #smoothing factor
    m = [0, 1]
    #for each smoothing factor
    for sm in m:
        for i in range(len(skfold_data)):
            #this loop considers ith fold for test dataset
            #get train(900) and test(100)
            t_train, t_test = train_test.kfold_train_test(skfold_data,i)
            
            #generate subsample factors
            sampling_factor = np.arange(0.1,1.1,0.1)
            check_acc = []
            size_of_train = []
            for n in sampling_factor:
                #loop for subsamples
                n = round(n, 2)
                #n = 0.2
                sample_size_for_train = int(len(t_train) * n)
                size_of_train.append(sample_size_for_train)
                #randomly select datapoints
                #sample_train = random.sample(t_train,sample_size_for_train)
                sample_train=t_train[0:sample_size_for_train] 
                train_x,train_y,test_x,test_y = train_test.train_test_split(sample_train,t_test)
                accu = NBC.predictMAP(train_x,train_y,test_x,test_y,sm)
                #append all the accuracies of subsamples of kth fold
                check_acc.append(accu)
            if(sm == 0):
                accuracy_m0.append(check_acc)
            if(sm == 1):
                accuracy_m1.append(check_acc)
    avgAccu0, avgAccu1 = calAccuracy(accuracy_m0, accuracy_m1)
    print("Average accuracies when m=0: ")
    print(avgAccu0)
    print("Average accuracies when m=1: ")
    print(avgAccu1)
    
    #calculate standard deviation
    sd_0 = []
    sd_1 = []
    for i in range(len(accuracy_m0)):
        x1 = np.std(accuracy_m0[i])
        sd_0.append(x1)
        x2 = np.std(accuracy_m1[i])
        sd_1.append(x2)
    
    print("standard deviation for m=0: ", sd_0)
    print("standard deviation for m=1: ", sd_1)
    
    """Refered some online material to know about how to plot error bar graphs"""
    """https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html"""
    """https://pythonforundergradengineers.com/python-matplotlib-error-bars.html"""
    plt.errorbar(size_of_train, avgAccu0,sd_0,label='m=0')
    plt.errorbar(size_of_train,avgAccu1,sd_1, label='m=1')
    plt.legend(loc='lower right')
    plt.xlabel('train set size')
    plt.ylabel('Average Accuracies')
    plt.show()

def Experiment_2(skfold_data):
    print("Running experiment 2...")
    Accuracy = []
    m1 = list(np.arange(0,1,0.1))
    m2 = list(np.arange(1,11,1))
    m = m1 + m2
    for sm in m:
        sm = round(sm,2)
        check_acc = []
        for i in range(len(skfold_data)):
            t_train, t_test = train_test.kfold_train_test(skfold_data, i)
            train_x,train_y,test_x,test_y = train_test.train_test_split(t_train,t_test)
            accu = NBC.predictMAP(train_x,train_y,test_x,test_y,sm)
            check_acc.append(accu)
        Accuracy.append(check_acc)
    
    #calculating average accuracy
    avgAccuracy2 = []
    for i in range(len(Accuracy)):
        x = np.average(Accuracy[i])
        avgAccuracy2.append(x)
    print("list of accuracies:")
    print(Accuracy)
    print("list of average accuracies")
    print(avgAccuracy2)
    
    #calculating standard deviation
    std = []
    for i in range(len(Accuracy)):
        x = np.std(Accuracy[i])
        std.append(x)
    print("Standard Deviation: ")
    print(std)
    
    #plotting
    plt.errorbar(m, avgAccuracy2,std)
    plt.xlabel('smoothing factor')
    plt.ylabel('Average Accuracies')
    plt.show()
    
if __name__ == "__main__":
    raw_data = getData(sys.argv[1])
    skfold_data = gkf.getStratifiedKfold(raw_data)
    x = int(sys.argv[2])
    if(x == 1):
        Experiment_1(skfold_data)
    elif(x == 2):
        Experiment_2(skfold_data)
    