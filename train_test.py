
"""
Preprocessing of Train and Test data
includes feature engineering and label encoding
"""

import pandas as pd
from sklearn import preprocessing


# read training and test datasets
train = pd.read_csv('../data/train_set.csv', parse_dates=[2,])
test = pd.read_csv('../data/test_set.csv', parse_dates=[3,])


# generate new features from the quote_date
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
train['dayofyear'] = train.quote_date.dt.dayofyear
train['dayofweek'] = train.quote_date.dt.dayofweek
train['day'] = train.quote_date.dt.day
train['week'] = train.quote_date.dt.week

test['year'] = test.quote_date.dt.year
test['month'] = test.quote_date.dt.month
test['dayofyear'] = test.quote_date.dt.dayofyear
test['dayofweek'] = test.quote_date.dt.dayofweek
test['day'] = test.quote_date.dt.day
test['week'] = test.quote_date.dt.week


# the cost of the tubes
labels = pd.DataFrame(train.cost, columns=['cost'])


# drop columns that are not needed anymore
test = test.drop(['id', 'quote_date'], axis = 1)
train = train.drop(['quote_date', 'cost'], axis = 1)


# label encoding of categorical variables
lbl = preprocessing.LabelEncoder()
for i in [1,4]:
  lbl.fit(list(train.ix[:,i]) + list(test.ix[:,i]))
  train.ix[:,i] = 1 + lbl.transform(train.ix[:,i])
  test.ix[:,i] = 1 + lbl.transform(test.ix[:,i])


# save train, test, and label data  
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
labels.to_csv('labels.csv', index=False)
