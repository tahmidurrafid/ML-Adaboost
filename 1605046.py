import pandas
import numpy
from pandas.core.arrays import categorical
from pandas.io.formats.format import CategoricalFormatter
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

csvData = pandas.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# csvData = pandas.read_csv("test.csv")
X = csvData.iloc[:, :-1].values
Y = csvData.iloc[:, -1].values
labelencoder = preprocessing.LabelEncoder()
Y = labelencoder.fit_transform(Y)
Y[Y == 0] = -1
ct = ColumnTransformer([("cols", OneHotEncoder(drop='if_binary'), [1, 2])], remainder = 'passthrough')
X = ct.fit_transform(X)
sc_X = preprocessing.StandardScaler()
X = sc_X.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X)
print(Y)

