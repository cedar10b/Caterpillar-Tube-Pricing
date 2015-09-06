
"""
Feature ranking script.

The feature ranking is done in 3 steps:
First,  the f-score is computed using Scikit-learn's f_regression
Second, for all pairs of features with correlation > 0.95,
        keep the feature of the pair with higher f-score (drop the other)
Third,  apply either a Random Forest Regressor or Gradient Boosting
        Regressor to rank the features

The number of features that will be used is determined with CV.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest


# read data (have been preprocessed)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
tube = pd.read_csv('tube.csv')
materials = pd.read_csv('materials.csv')
specs = pd.read_csv('specs.csv')
labels = pd.read_csv('labels.csv')


# merge tube, materials, and specs dataframes with train and test data 
train = pd.merge(train, tube, on='tube_assembly_id', how='left')
train = pd.merge(train, materials, on='tube_assembly_id', how='left')
train = pd.merge(train, specs, on='tube_assembly_id', how='left')
test = pd.merge(test, tube, on='tube_assembly_id', how='left')
test = pd.merge(test, materials, on='tube_assembly_id', how='left')
test = pd.merge(test, specs, on='tube_assembly_id', how='left')


# delete tube, materials, specs dataframes (not needed anymore)
del tube, materials, specs


#label encoding of tube_assembly_id
lbl = preprocessing.LabelEncoder()
for i in ['tube_assembly_id']:
  lbl.fit(list(train[i]) + list(test[i]))
  train[i] = 1 + lbl.transform(train[i])
  test[i] = 1 + lbl.transform(test[i])


# copy column names before converting pandas dataframe to numpy array
columns = train.columns.values


# copy train and test datasets before converting them to numpy array
# they will be needed later for feature selection with Random Forest 
train_copy = train.copy()
test_copy = test.copy()
print train_copy.shape, test_copy.shape


# convert data to numpy array
train = np.array(train)
test = np.array(test)


# change data type to float
train = train.astype(float)
test = test.astype(float)


#train on labels^(1/power)
power = 20.
label_pow = np.power(labels.cost.values, 1./power)


# standardize data
scaler = StandardScaler().fit(np.concatenate((train,test), axis=0))
train = scaler.transform(train); test = scaler.transform(test)


# compute F-score for all features and keep those with fscore > fscore_threshold
fscore_threshold = 0.0
selector = SelectKBest(f_regression, k='all')
selector.fit(train, label_pow)
train = selector.transform(train)
test = selector.transform(test)
scores=selector.scores_
for i in range(scores.shape[0]):
  if np.isnan(scores[i]): 
    scores[i] = 0.0
#zip(np.arange(1,scores.shape[0]+1)[::-1], np.argsort(scores), 
#    scores[np.argsort(scores)], columns[np.argsort(scores)])


# keep only f-scores > fscore_threshold and corresponding column_names
low_fscore_cols = list(columns[scores < fscore_threshold])
columns = columns[scores >= fscore_threshold]
scores = scores[scores >= fscore_threshold]


# define new dataframe f and store F-score rankings of features  
f = pd.DataFrame(np.argsort(scores)[::-1], columns=['attr_index'])
f['fscore'] = scores[np.argsort(scores)][::-1]
f['attr'] = columns[np.argsort(scores)][::-1]


#find all pairs of correlated features with Pearson corr > 0.95
attr1 = []; attr2 = []; corr_lst = []
for i in np.argsort(scores)[::-1]:
  for j in np.argsort(scores)[::-1]:
    corr = np.corrcoef(train[:,i], train[:,j])
    if (i > j) and corr[0,1] >= 0.95:
      attr1.append(columns[i])
      attr2.append(columns[j])
      corr_lst.append(corr[0,1])


# save in mcorr dataframe the names of highly correlated attributes, 
# their pearson correlation, and the corresponding f-scores
mcorr = pd.DataFrame(zip(attr1, attr2, corr_lst), columns=['attr1', 'attr2', 'corr_lst'])
mcorr.sort('corr_lst', ascending=False, inplace=True)
f1temp = [f.fscore[f.attr[f.attr==i].index.values[0]] for i in mcorr.attr1.values]
f2temp = [f.fscore[f.attr[f.attr==i].index.values[0]] for i in mcorr.attr2.values]
mcorr['f1'] = f1temp
mcorr['f2'] = f2temp

    
# for every pair of highly correlated attributes, 
# keep the one with the highest f-score
drop_cols = []
for i in range(mcorr.shape[0]):
  if mcorr.f1[i] >= mcorr.f2[i]:
    drop_cols.append(mcorr.attr2[i])
  elif mcorr.f1[i] < mcorr.f2[i]:
    drop_cols.append(mcorr.attr1[i])
drop_cols = list(set(drop_cols))


# use original pandas dataframes and drop features that have low f-score 
# or are highly correlated with other features
train = train_copy.drop(low_fscore_cols + drop_cols, axis=1)
test = test_copy.drop(low_fscore_cols + drop_cols, axis=1)
columns = train.columns.values


# print shape of new train and test data
print train.shape, test.shape


# convert data to numpy array
train = np.array(train)
test = np.array(test)


# change data type to float
train = train.astype(float)
test = test.astype(float)


# standardize data
scaler = StandardScaler().fit(np.concatenate((train,test), axis=0))
train = scaler.transform(train); test = scaler.transform(test)


# feature selection with Random Forest Regressor
print 'RF feature selection begins... please wait...'
clf = RandomForestRegressor(n_estimators=2000, random_state=7)
clf.fit(train, label_pow)
feature_importances = clf.feature_importances_
best_features = np.argsort(feature_importances)[::-1]


# save features and their feature_importances
pd.DataFrame({'best_features': columns[best_features], \
              'score': sorted(feature_importances)[::-1]}). \
             to_csv('best_features_RF.csv', index=False)
