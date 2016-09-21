#!/usr/bin/python

import quandl, math, datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle

df = quandl.get("WIKI/GOOGL")
print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_Change'] = (df['Adj. High'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# Here we take 0.01 or 1% of the length of all the rows within the dataframe. Each row in the df is representative of the stock price per day. 
# So if the stock has been trading for 365 days, there will be 365 rows in the df. 
# 1% of 365 is 3.65 days which is then rounded up by the math.ceil() to 4 days. 
# The 4 days will be the forecast_out variable which is the variable that used to shift the forecast_col price column in the df up by 4.
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
print("length of X is ", len(X))
#print(df.head())

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

print("length of X_lately is ", len(X_lately))
print("length of X is ", len(X))

df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# The following 4 lines are needed, as without them, the NaN will sneak into your scaled data set and 
# you will get ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
# http://stackoverflow.com/questions/34779961/scikit-learn-error-in-fitting-model-input-contains-nan-infinity-or-a-value
x_train[np.isnan(x_train)] = np.median(x_train[~np.isnan(x_train)])
y_train[np.isnan(y_train)] = np.median(y_train[~np.isnan(y_train)])

x_test[np.isnan(x_test)] = np.median(x_test[~np.isnan(x_test)])
y_test[np.isnan(y_test)] = np.median(y_test[~np.isnan(y_test)])

clf = LinearRegression(n_jobs=5)
# clf = svm.SVR()
clf.fit(x_train, y_train)                   # this is to train the model with training dataset
accuracy = clf.score(x_test, y_test)        # this is to test the model with test dataset
print(accuracy)

# we need to save the classifier so we don't have to run it over and over each time we use it
with open('linearRegression.pickle', 'wb') as f:
    pickle.dump(clf, f)

# to use the saved classifier
pickle_in = open('linearRegression.pickle', 'rb');
clf = pickle.load(pickle_in)

# do the prediction
forecast_set = clf.predict(X_lately)
print(forecast_set)

# now we put the predicted values into the df, we need to find the last date and do thing accordingly
df['forecast'] = np.nan

pd.DatetimeIndex(df.iloc[-1])
last_date = df.iloc[-1]     # this will fetch the index of the dataframe df, which is a datetime
print(last_date)
#last_date_unix = (pd.to_datetime(last_date)).timestamp()
last_date_unix = last_date.timestamp()
one_day = 86400
next_date_unix = last_date_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_date_unix)
    next_date_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    
# the original vedio has a graph plot lines here, I didn't do it as it's not easy here




