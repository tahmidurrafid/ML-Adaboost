import math
from os import replace
import pandas
import numpy
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

class DataSet:
    def preprocess(self, preProcessor):
        self.X_train, self.X_test, self.Y_train, self.Y_test = preProcessor()

        self.featureCount = len(self.X_test[0])

        yes = numpy.count_nonzero(self.Y_train == 1)
        no = numpy.count_nonzero(self.Y_train == -1)
        # print(self.Y_train[0:100])
        p = yes/(yes+no)
        print(p)
        self.entropy = -(p*math.log(p , 2) + (1-p)*math.log(1-p, 2) )
        self.setPriority()

    def calculateIG(self, col):
        arr = self.X_train[:,col].copy()
        uniques = numpy.unique(arr)
        max_bucket = 10
        ig = self.entropy
        if(len(uniques) > max_bucket):
            arr = numpy.array(arr).reshape(-1, 1)
            est = preprocessing.KBinsDiscretizer(n_bins=[max_bucket], 
                    encode='ordinal').fit(arr)
            arr = numpy.array(est.transform(arr)).reshape(1, -1)[0]
            uniques = numpy.unique(arr)        
        matrix = numpy.zeros(shape=(len(uniques),2))
        for i in range(0, len(uniques)):
            for j in range(0, len(arr)):
                if(arr[j] == uniques[i]):
                    if(self.Y_train[j] == 1):
                        matrix[i, 0] += 1
                    else:
                        matrix[i, 1] += 1
        for i in range(0, len(matrix)):            
            sum = matrix[i][0] + matrix[i][1]
            p0 = matrix[i][0]/sum
            p1 = 1 - p0
            if(p0 == 0 or p1 == 0):
                continue
            entropy = -(p0*math.log(p0, 2) + p1*math.log(p1, 2))
            ig -= (sum/len(self.X_train) )*entropy
        return ig

    def setPriority(self):
        priority = []
        for i in range(0, len(self.X_train[0,:])):
            priority.append((self.calculateIG(i), i))
        priority[1:] = sorted(priority[1:])
        priority[1:] = reversed(priority[1:])
        self.featureOrder = []
        for i in range(0, len(priority)):
            self.featureOrder.append(priority[i][1])

    def resample(self, w):
        numpy.random.seed(0)
        indices = [i for i in range(0, len(self.X_train))]
        indices = numpy.random.choice(indices, len(indices), p=w) 
        X = [self.X_train[indices[i]] for i in range(0, len(self.X_train))]
        Y = [self.Y_train[indices[i]] for i in range(0, len(self.X_train))]
        print(type(self.X_train))
        self.X_train = numpy.array(X)
        self.Y_train = numpy.array(Y)
        

class LogisticRegression:
    def __init__(self, dataSet : DataSet, featureCount = 10, minError = 0, maxIteration = 5):
        self.dataSet = dataSet
        if(featureCount == 0):
            featureCount = dataSet.featureCount
        self.featureCount = featureCount
        self.minError = minError
        self.maxIteration = maxIteration

    def calculateZ(self, row):
        val = 0
        featureOrder = self.dataSet.featureOrder
        for i in range(0, self.featureCount):
            val += self.w[featureOrder[i]]*row[featureOrder[i]]
        return val

    def getOutput(self, row):
        z = self.calculateZ(row)
        return 1 if numpy.tanh(z) >= 0 else -1

    def calculateTrainingError(self):
        correct = 0
        wrong = 0
        for i in range(0, len(self.dataSet.X_train)):
            z = self.calculateZ(self.dataSet.X_train[i])
            # print(z)
            out = 1 if numpy.tanh(z) >= 0 else -1
            if(out == self.dataSet.Y_train[i]):
                correct += 1
            else:
                wrong += 1
        return (wrong/(correct + wrong))

    def run(self):
        X = self.dataSet.X_train
        Y = self.dataSet.Y_train
        self.w = [0 for i in range(0, len(X[0])) ]
        alpha = .0005
        for iter in range(0, self.maxIteration):
            print(self.calculateTrainingError())            
            for i in range(0, len(X)):
                z = self.calculateZ(X[i])  
                for k in range(0, self.featureCount):
                    j = self.dataSet.featureOrder[k]
                    gx = numpy.tanh(z)
                    self.w[j] += alpha*(Y[i] - gx)*(1 - gx*gx)*X[i][j]
            if self.calculateTrainingError() < self.minError:
                break
        return self.w

class AdaBoost:
    def __init__(self, dataSet:DataSet, k):
        self.dataSet = dataSet
        self.k = k
        self.z = [0 for i in range(0, k)]
        self.h = []

    def run(self):
        N = len(self.dataSet.X_train)
        w = [1/N for i in range(0, N)]
        for k in range(0, self.k):
            sampledSet = deepcopy(self.dataSet)
            sampledSet.resample(w)
            self.h.append(LogisticRegression(sampledSet))
            self.h[k].run()
            error = 0
            for j in range(0, N):
                if(self.h[k].getOutput(self.dataSet.X_train[j]) != self.dataSet.Y_train[j]):
                    error += w[j]
            if error > .5:
                continue
            for j in range(0, N):
                if(self.h[k].getOutput(self.dataSet.X_train[j]) == self.dataSet.Y_train[j]):
                    w[j] = w[j]*error/(1-error)
            s = sum(w)
            for i in range(0, len(w)):
                w[i] = w[i]/s
            self.z[k] = math.log((1-error)/error)
        s = sum(self.z)
        for i in range(0, len(self.z)):
            self.z[i] = self.z[i]/s
        
    def accuracy(self):
        right = 0
        wrong = 0
        print(self.z)
        for i in range(0, len(self.dataSet.X_test)):
            for j in range(0, self.k):
                output = 0
                output += self.h[j].getOutput(self.dataSet.X_test[i])*self.z[j]
                output = 1 if output >= 0 else -1
                if(output == self.dataSet.Y_test[i]):
                    right += 1
                else:
                    wrong += 1
        print("======measure=======")
        print(right/(right+wrong))

def telcoPreprocessor():
    csvData = pandas.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    csvData.replace(r'^ $', numpy.NaN, regex=True,inplace=True)
    csvData.dropna(inplace=True)

    X = csvData.iloc[:, 1:-1].values
    print(type(X))
    Y = csvData.iloc[:, -1].values
    labelencoder = preprocessing.LabelEncoder()
    Y = labelencoder.fit_transform(Y)
    Y[Y == 0] = -1
    columnNum = [1, 4, 17, 18]
    columnCat = [i for i in range(0,19) if i not in columnNum]

    ct = ColumnTransformer([("cols", OneHotEncoder(drop='if_binary'), columnCat)], remainder = 'passthrough')
    X = ct.fit_transform(X)

    sc_X = preprocessing.StandardScaler()
    X = sc_X.fit_transform(X)

    X = numpy.insert(X, 0, values=1, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train, X_test, Y_train, Y_test

def adultPreprocessor():
    csvData = pandas.read_csv("adult.data")
    csvTest = pandas.read_csv("adult.test.csv", skiprows=[0])

    csvData.replace(r'^ $', numpy.NaN, regex=True,inplace=True)
    csvData.replace('?', numpy.NaN, inplace=True)
    csvData.dropna(inplace=True)

    csvTest.replace(r'^ $', numpy.NaN, regex=True,inplace=True)
    csvTest.replace('?', numpy.NaN, inplace=True)
    csvTest.dropna(inplace=True)

    X = csvData.iloc[:, 0:-1].values
    Y = csvData.iloc[:, -1].values
    testLen = len(X)

    X = numpy.concatenate((X , csvTest.iloc[:, 0:-1].values), axis=0)

    Y = numpy.concatenate((Y , csvTest.iloc[:, -1].values), axis=0)
    for i in range(0, len(Y)):        
        Y[i] = Y[i].replace('.', '')

    labelencoder = preprocessing.LabelEncoder()
    Y = labelencoder.fit_transform(Y)
    Y[Y == 0] = -1
    columnNum = [0, 2, 4, 10, 11, 12]
    columnCat = [i for i in range(0,len(X[0])) if i not in columnNum]

    ct = ColumnTransformer([("cols", OneHotEncoder(drop='if_binary'), columnCat)], remainder = 'passthrough')
    X = ct.fit_transform(X).toarray()

    sc_X = preprocessing.StandardScaler(with_mean=False)
    X = sc_X.fit_transform(X)
    X = numpy.insert(X, 0, values=1, axis=1)

    X_train = X[:testLen]
    X_test = X[testLen:]
    Y_train = Y[:testLen]
    Y_test = Y[testLen:]
    return X_train, X_test, Y_train, Y_test


dataset = DataSet()
dataset.preprocess(adultPreprocessor)
# dataset.preprocess(telcoPreprocessor)

adaboost = AdaBoost(dataset, 5)
adaboost.run()
adaboost.accuracy()
# adaboost.print()


# learner = LogisticRegression(dataset, 20)
# learner.run()
