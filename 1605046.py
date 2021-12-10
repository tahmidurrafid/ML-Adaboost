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

print(X)
# print(Y)

