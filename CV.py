
"""
Read preprocessed data and merge all the dataframes into one 
Use 5-fold CV for feature selection and XGBoost parameter tuning
(make sure tubes with the same tube_assembly_id belong to the same fold)
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


# distribute tubes into N groups (will be used for Cross Validation)
# tubes with same ID should be in the same group 
def index_split(tubeID_list, Nsplits):
  """
  Args: a list of all tube_assembly_ids and the number of CV folds
  Returns: a list of lists with the indices for each CV fold
  """
  SPLITS = Nsplits
  lst = []
  for i in range(SPLITS):
    lst.append([])
  lst[0].append(0)
  listnum = 0  
  for i in range(1, len(tubeID_list)):
    if tubeID_list[i] == tubeID_list[i-1]:
      lst[listnum].append(i)
    else:
      listnum += 1
      if listnum == SPLITS:
        listnum = 0
      lst[listnum].append(i)
  return lst


# find indices for CV train and test datasets
def CV_split(all_ind, Nsplits, i):
  """
  Args: a list of lists with the indices for each CV fold
        the total number of CV folds
        the current number of CV iteration (from 1 to Nsplits)
  Returns: the indices of the CV train and test tubes for
           the current CV iteration
  """
  test_ind = all_ind[i]
  train_ind = []
  other_ind = range(Nsplits); other_ind.remove(i)
  for j in other_ind:  
    train_ind = train_ind + all_ind[j]
  return sorted(train_ind), test_ind  


# CV training and testing
Nsplits = 5   # number of CV folds 
train_preds = -np.ones(train.shape[0])
all_ind = index_split(tubeID_list, Nsplits)
error = -np.ones(Nsplits)
for i in range(Nsplits):
  print 'iteration # ', i
  
  # generate CV train and test data
  train_ind, test_ind = CV_split(all_ind, Nsplits, i)  
  xgbtrain = xgb.DMatrix(train[train_ind, :], label=label_pow[train_ind])
  xgbtest = xgb.DMatrix(train[test_ind])

  # XGBoost parameters
  param = {'max_depth':12, 'eta':0.01, 'min_child_weight': 18.0, 
           'objective':'reg:linear', 'subsample': 0.7, 'nthread':4,
           'colsample_bytree': 0.65, 'seed':1}
  num_round = 9000
  
  # training, testing, and computation of RMS logarithmic error
  clf = xgb.train(param, xgbtrain, num_round)
  preds = clf.predict(xgbtest)
  error[i] = np.sqrt(np.mean( (np.log1p(labels.cost.values)[[test_ind]]-
                               np.log1p(np.power(preds, power)))**2 ))
  train_preds[test_ind] = np.power(preds, power)

# print error and save CV predictions
print 'Mean error = ', np.mean(error)
np.savetxt('train_preds.csv', train_preds)
