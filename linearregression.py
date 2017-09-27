import pandas as pd
import quandl
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_CHANGE', 'Adj. Volume']]
forecast_col = 'Adj. Close'

df.fillna(-999999, inplace=True)
forcase_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forcase_out)
df.dropna(inplace=True)
#print(df.head())

x = np.array(df.drop(['label'],1))
y = np.array(df['label'])
#print(len(x), len(y))

x_train , x_test , y_train , y_test = cross_validation.train_test_split(x,y, test_size=0.2)
clf = LinearRegression()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test , y_test)

print(accuracy)