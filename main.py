
"""
Read preprocessed data and merge all the dataframes into one 
Use the XGBoost library to build a predictive model
Apply a Bagging metaestimator to improve model
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


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


# generate a list with tube_assembly_ids of train set
tubeID_list = list(train['tube_assembly_id'].values)


#label encoding of tube_assembly_id
lbl = preprocessing.LabelEncoder()
for i in ['tube_assembly_id']:
  lbl.fit(list(train[i]) + list(test[i]))
  train[i] = 1 + lbl.transform(train[i])
  test[i] = 1 + lbl.transform(test[i])


# feature selection
# features have been ranked either with Random Forest or GB Regressor
# keep only the most informative features (exact number determined with CV)
N_attr_keep = 72
best_features = pd.read_csv('best_features_RF.csv')
train = train.ix[:, best_features.best_features[0:N_attr_keep].values]
test = test.ix[:, best_features.best_features[0:N_attr_keep].values]
print 'shape of train data = ', train.shape
print 'shape of test data = ', test.shape


# copy column names before converting pandas dataframe to numpy array
columns = train.columns.values


# convert data to numpy array
train = np.array(train)
test = np.array(test)
print train.shape, test.shape


# change data type to float
train = train.astype(float)
test = test.astype(float)


#train on labels^(1/power)
power = 20.
label_pow = np.power(labels.cost.values, 1./power)


# standardize data
scaler = StandardScaler().fit(np.concatenate((train,test), axis=0))
train = scaler.transform(train); test = scaler.transform(test)


#compute bagging models for training and testing
BAG_ITER = 20
num_round = 9000
models = []       
for i in range(BAG_ITER):
  print 'training model # ', i+1
  param = {'max_depth':12, 'eta':0.01, 'min_child_weight': 18.0,
         'objective':'reg:linear', 'subsample': 0.7, 'nthread':4,
         'colsample_bytree': 0.65, 'seed': i}
  np.random.seed(10*i)
  train_ind = np.random.choice(train.shape[0], train.shape[0])  
  xgbtrain = xgb.DMatrix(train[train_ind, :], label=label_pow[train_ind])
  xgbtest = xgb.DMatrix(test)
  clf = xgb.train(param, xgbtrain, num_round)
  models.append(clf)


# compute predictions for every bagged model  
predictions = np.zeros((test.shape[0], len(models)))
for i,m in enumerate(models):
  predictions[:,i] = m.predict(xgbtest)


# generate submission file
preds = np.mean(np.power(predictions, power), axis=1)
submission = pd.DataFrame([], columns=['id', 'cost'])
submission['id'] = np.arange(1,preds.shape[0]+1)
submission['cost'] = preds
submission.to_csv('predictions.csv', index=False)
