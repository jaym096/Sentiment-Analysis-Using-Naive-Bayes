def kfold_train_test(skfold_data, k):
    temp_test = skfold_data[k]
    temp_train = []
    for i in range(len(skfold_data)):
        if(i == k):
            continue
        temp_train += skfold_data[i]
    return temp_train, temp_test

def train_test_split(train,test):
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []
    for i in range(len(train)):
        xtrain.append(train[i][0])
        ytrain.append(train[i][1])
    for i in range(len(test)):
        xtest.append(test[i][0])
        ytest.append(test[i][1])
    return xtrain, ytrain, xtest, ytest