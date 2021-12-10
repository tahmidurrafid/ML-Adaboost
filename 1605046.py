from copy import deepcopy
import math
import pandas
import numpy
from pandas.core.arrays import categorical
from pandas.io.formats.format import CategoricalFormatter
from scipy.sparse import data
from scipy.sparse.construct import random
from scipy.sparse.sputils import matrix
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class DataSet:
    def preprocess(self):
        csvData = pandas.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        csvData.replace(r'^ $', numpy.NaN, regex=True,inplace=True)
        csvData.dropna(inplace=True)

        # csvData = pandas.read_csv("test.csv")
        X = csvData.iloc[:, 1:-1].values
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
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.featureCount = len(self.X_test[0])

        yes = numpy.count_nonzero(Y_train == 1)
        no = numpy.count_nonzero(Y_train == -1)
        p = yes/(yes+no)
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
        print(self.featureOrder)

class LogisticRegression:
    def __init__(self, dataSet : DataSet, featureCount = 0, minError = 0, maxIteration = 5):
        self.dataSet = dataSet
        if(featureCount == 0):
            featureCount = dataSet.featureCount
        self.featureCount = featureCount
        self.minError = minError
        self.maxIteration = maxIteration

    def calculateZ(self, row, cor = 0):
        val = 0
        featureOrder = self.dataSet.featureOrder
        for i in range(0, self.featureCount):
            val += self.w[featureOrder[i]]*row[featureOrder[i]]
        return val

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
        print(wrong/(correct + wrong))
        return (wrong/(correct + wrong))

    def run(self):
        X = self.dataSet.X_train
        Y = self.dataSet.Y_train
        self.w = [0 for i in range(0, len(X[0])) ]
        alpha = .0005
        for iter in range(0, self.maxIteration):
            for i in range(0, len(X)):
                z = self.calculateZ(X[i])  
                for k in range(0, self.featureCount):
                    j = self.dataSet.featureOrder[k]
                    gx = numpy.tanh(z)
                    self.w[j] += alpha*(Y[i] - gx)*(1 - gx*gx)*X[i][j]
            if self.calculateTrainingError() < self.minError:
                break
        return self.w


dataset = DataSet()
dataset.preprocess()

learner = LogisticRegression(dataset, 30)
learner.run()
