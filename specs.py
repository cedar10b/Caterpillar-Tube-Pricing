
"""
Preprocessing of Specs data
includes data cleaning, feature engineering, label encoding, 
one-hot encoding, and grouping of rare labels
"""

import pandas as pd
import numpy as np


# read specs data
specs = pd.read_csv('../data/Specs.csv')


# fix row 37 (for TA-00038) which has both spec1, spec2 = SP-0007
# substitute spec2 with most frequent value for spec2 of 'similar' tubes 
# 'similar' means tubes with all other specs the same
ind = (specs.spec1 == specs.spec1[37]) & (specs.spec3 == specs.spec3[37]) & \
      (specs.spec4 == specs.spec4[37]) & (specs.spec5 == specs.spec5[37])
specs.ix[37, 'spec2'] = specs.ix[ind, :].spec2.value_counts().index.values[0]

      
# find unique specs codes
all_specs = []
for i in specs.columns.values[1:]:
  for j in specs[i].unique():
    all_specs.append(j)
all_specs = set(all_specs)
all_specs.remove(np.nan)
specs_list = sorted(list(all_specs))


# one-hot encoding for specs codes
transformed_specs = pd.DataFrame(np.zeros((specs.shape[0], 
                                 len(specs_list))), columns=all_specs)
for i in range(specs.shape[0]):
  for j in range(1, specs.shape[1]):
    if specs.ix[i,j] != specs.ix[i,j]: break
    temp_spec = specs.ix[i, j]
    transformed_specs.ix[i, temp_spec] = 1

    
# generate new feature: total number of specs in a tube  
Nspecs = np.sum(transformed_specs.values, axis=1)


# find specs that appear at least min_count times (e.g. 75) in data and
# generate new feature: SP-rare = sum of rare specs with count < min_count  
min_count = 75.
counts = zip(transformed_specs.columns.values, np.sum(transformed_specs))
transformed_specs['SP-rare'] = 0.0
rare_specs = []
freq_specs = []
for i in range(len(counts)):
  if counts[i][1] >= min_count:
    freq_specs.append(transformed_specs.columns.values[i])
  else:
    rare_specs.append(transformed_specs.columns.values[i])
    transformed_specs.ix[:, 'SP-rare'] += transformed_specs.ix[:, counts[i][0]]


# drop rare specs (keep only SP-rare to represent them)
transformed_specs.drop(rare_specs, axis=1, inplace=True)


# add new feature:'total_specs_score': a label for all specs in a tube
# each spec is encoded using a uniformly spaced label encoding from 1 to 1.1
# for tubes with multiple specs, the total_specs_score is the sum 
# of all individual specs labels in that tube
# the label range of 1-1.1 was chosen to avoid collisions: 
# each tube has up to 10 specs so with this encoding 9*1.1 < 10*1
# the most frequent specs (e.g. SP-0080) get the smallest labels (1.0)
total_specs_score = np.zeros(specs.shape[0])
ind=sorted(np.arange(len(transformed_specs.sum())), 
           key=lambda x: transformed_specs.sum()[x])
counts = transformed_specs.sum()[ind][::-1]
Nlabels = counts.shape[0]
spec_label = pd.Series(np.linspace(1.0,1.1,Nlabels), index=counts.index)
for i in range(specs.shape[0]):
  specs_in_tube = specs.ix[i, ~specs.iloc[i].isnull()][1:].values
  for j in range(specs_in_tube.shape[0]):
    spec = specs_in_tube[j]
    if spec in rare_specs: spec = 'SP-rare'
    total_specs_score[i] += spec_label[spec]
transformed_specs['total_specs_score'] = total_specs_score


# add a column for Nspecs (total number of specs in a tube) as new feature
transformed_specs['Nspecs'] = Nspecs                          


# add tube_assembly_id feature to the new specs dataframe
transformed_specs['tube_assembly_id'] = specs['tube_assembly_id']


# save new specs data
transformed_specs.to_csv('specs.csv', index=False)