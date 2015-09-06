
"""
Preprocessing of Bill_of_materials and Comp_[type] data
includes data cleaning, feature engineering, label encoding, 
one-hot encoding, and grouping of rare labels
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing


# prepare Bill_of_Materials data    
materials = pd.read_csv('../data/Bill_of_Materials.csv')


# fix rows 21141, 21142 (contain '9999' in the component_id_1 field)
# substitute with most frequent value
i = 'component_id_1'
most_freq_item = materials[i].groupby(materials[i]).count().idxmax()
materials.ix[[21141, 21142], i] = most_freq_item


#------ fix component_id_3 and component_id_4 -----------------------
# there are NaN values for component_id_3 and _4 for which
# the corresponding quantity_3 and _4 fields are equal to 1
# for each one of these rows do the following: 
# find which other rows have same component_id_1 and _2 (similar_rows)
# among these rows find the most frequent component_id_3 and _4
# substitute missing values with most frequent components

# define some variables for convenience
id1 = materials.component_id_1; id2 = materials.component_id_2
id3 = materials.component_id_3; id4 = materials.component_id_4
q1 = materials.quantity_1; q2 = materials.quantity_2
q3 = materials.quantity_3; q4 = materials.quantity_4

# find indices of problematic rows that need to be fixed for component_id_3
ind = q3[id3[id3.isnull().values].index.values][~q3.isnull()].index.values

# find "similar" rows for every problematic row and
# substitute missing values with most frequent components of "similar" rows
for i in ind:
  comp1 = id1[i]; comp2 = id2[i]
  col1_ind = id1[id1 == comp1].index.values
  col2_ind = (id2[col1_ind] == comp2)[id2[col1_ind] == comp2].index.values
  freq_val = id3[col2_ind].groupby(id3[col2_ind]).count().idxmax()
  materials.ix[i,'component_id_3'] = freq_val

# find indices of problematic rows that need to be fixed for component_id_4
ind = q4[id4[id4.isnull().values].index.values][~q4.isnull()].index.values

# find "similar" rows for every problematic row and
# substitute missing values with most frequent components of "similar" rows
for i in ind:
  comp1 = id1[i]; comp2 = id2[i]; comp3 = id3[i]
  col1_ind = id1[id1 == comp1].index.values
  col2_ind = (id2[col1_ind] == comp2)[id2[col1_ind] == comp2].index.values
  col3_ind = (id3[col2_ind] == comp3)[id3[col2_ind] == comp3].index.values  
  freq_val = id4[col3_ind].groupby(id4[col3_ind]).count().idxmax()
  materials.ix[i,'component_id_4'] = freq_val
#-------------------------------------------------------------------- 


# find unique components in materials dataframe (ignore quantity)
categ_col = [1,3,5,7,9,11,13,15]
all_attr = pd.Series(materials.ix[:, categ_col].values.ravel()).unique()
all_attr = list(all_attr); all_attr.remove(np.nan)
all_attr = sorted(all_attr)


# Number of component_ids to keep
Nid_keep = len(all_attr)


#find how many times each item is used and keep top Nid_keep items
id_counts = pd.DataFrame(np.zeros((len(all_attr))),index=all_attr, 
                         columns=['times_used'])
for i in materials.columns[categ_col]:
  ids = list(materials[i].groupby(materials[i]).count().index.values)
  idcounts = materials[i].groupby(materials[i]).count().values
  id_counts.ix[ids, 'times_used'] += idcounts
id_counts.sort(['times_used'], ascending=False, inplace=True)
keep_id = list(id_counts.ix[0:Nid_keep].index.values)


#generate transformed_materials dataframe which has as new features 
#the most frequent component_ids and as indices the tube_assembly_ids
transformed_materials = pd.DataFrame(np.zeros((materials.shape[0], len(keep_id))), 
                           index=materials.tube_assembly_id.values,
                           columns = keep_id)
for i in range(materials.shape[0]):
  TAid = materials.tube_assembly_id.values[i]
  for j in categ_col:
    if materials.ix[i,j] == materials.ix[i,j]:
      if materials.ix[i,j] in keep_id:
        transformed_materials.ix[TAid, materials.ix[i,j]] = materials.ix[i,j+1]
    else:
      break

#=============================================================

# read and clean Comp_[type] data and generate new features
# new features include:
# number of each type component in a tube (e.g. number of adaptors, sleeves etc)
# total number of components in a tube
# properties of adaptors, nuts etc (e.g weight, orientation etc)
# total_comp_score: a label for all comp. of same type in a tube

#=============================================================


# read Comp_[type] data as a list of dataframes
comp_type_names = ['adaptor', 'boss', 'elbow', 'hfl', 'nut', 'other', 
                   'sleeve', 'straight', 'tee', 'threaded', 'cfloat']              
comp_type_data = []
for i in range(len(comp_type_names)-1):
  comp_type_data.append(pd.read_csv('../data/comp_' +
                        comp_type_names[i] + '.csv'))
comp_type_data.append(pd.read_csv('../data/comp_float.csv'))

# --------------------   data cleaning   -------------------------------------
# fix missing values (9999.0 or '9999') for adaptor type data
# for categ. attributes substitute with most frequent value
# for numerican attributes substitute with median value
k = 0 #adaptor
for i in ['end_form_id_2', 'connection_type_id_2']: #categorical features
  ind2 = comp_type_data[k][i][comp_type_data[k][i] == 9999.0].index.values
  ind = comp_type_data[k][i][comp_type_data[k][i] == '9999'].index.values
  ind = np.concatenate((ind, ind2))
  freq_val = comp_type_data[k][i].groupby(comp_type_data[k][i]).count().idxmax()
  if ind.shape[0] > 0:
    comp_type_data[k].ix[ind, i] = freq_val
for i in ['thread_size_2', 'thread_pitch_2']: #numerical features
  ind = comp_type_data[k][i][comp_type_data[k][i] == 9999.0].index.values
  freq_val = comp_type_data[k][i].median()
  if ind.shape[0] > 0:
    comp_type_data[k].ix[ind, i] = freq_val
    
# fix missing values (9999.0 or '9999') for boss type data
# for categ. attributes substitute with most frequent value
# for numerican attributes substitute with median value    
k = 1 #boss
for i in ['connection_type_id']: #categorical features
  ind2 = comp_type_data[k][i][comp_type_data[k][i] == 9999.0].index.values
  ind = comp_type_data[k][i][comp_type_data[k][i] == '9999'].index.values
  ind = np.concatenate((ind, ind2))
  freq_val = comp_type_data[k][i].groupby(comp_type_data[k][i]).count().idxmax()
  if ind.shape[0] > 0:
    comp_type_data[k].ix[ind, i] = freq_val
for i in ['height_over_tube']: #numerical features
  ind = comp_type_data[k][i][comp_type_data[k][i] == 9999.0].index.values
  freq_val = comp_type_data[k][i].median()
  if ind.shape[0] > 0:
    comp_type_data[k].ix[ind, i] = freq_val
    
# fix missing values (9999.0 or '9999') for elbow type data
# for categ. attributes substitute with most frequent value
# for numerican attributes substitute with median value    
k = 2 #elbow
for i in ['drop_length']: #numerical features
  ind = comp_type_data[k][i][comp_type_data[k][i] == 9999.0].index.values
  freq_val = comp_type_data[k][i].median()
  if ind.shape[0] > 0:
    comp_type_data[k].ix[ind, i] = freq_val

# fix missing values (9999.0 or '9999') for sleeve type data
# for categ. attributes substitute with most frequent value
# for numerican attributes substitute with median value
k = 6 #sleeve
for i in ['length']: #numerical features
  ind = comp_type_data[k][i][comp_type_data[k][i] == 9999.0].index.values
  freq_val = comp_type_data[k][i].median()
  if ind.shape[0] > 0:
    comp_type_data[k].ix[ind, i] = freq_val

# fix missing values (9999.0 or '9999') for threaded type data
# substitute with nans
k = 9 #threaded
for i in ['nominal_size_1', 'nominal_size_2', 'nominal_size_3']:
  ind = comp_type_data[k][i][comp_type_data[k][i] == 9999.0].index.values
  if ind.shape[0] > 0:
    comp_type_data[k].ix[ind, i] = np.nan
for i in ['nominal_size_1']:
  ind = comp_type_data[k][i][comp_type_data[k][i] == 'See Drawing'].index.values 
  ind2 = comp_type_data[k][i][comp_type_data[k][i] == '9999'].index.values  
  ind = np.concatenate((ind, ind2))  
  if ind.shape[0] > 0:
    comp_type_data[k].ix[ind, i] = np.nan
#convert string type data to numerical type data   
comp_type_data[k]['nominal_size_1'] = comp_type_data[k]['nominal_size_1'].astype(float)
#drop nominal_size_4 column (it only has NaNs)
comp_type_data[k].drop('nominal_size_4', axis=1, inplace=True)

# fix 'M6', 'M8', 'M10', 'M12' values for thread_size of nut type data
# M=metric, the values are in mm, convert to inches
comp_type_data[4].ix[comp_type_data[4].thread_size=='M6','thread_size'] = 0.236
comp_type_data[4].ix[comp_type_data[4].thread_size=='M8','thread_size'] = 0.315
comp_type_data[4].ix[comp_type_data[4].thread_size=='M10','thread_size'] = 0.394
comp_type_data[4].ix[comp_type_data[4].thread_size=='M12','thread_size'] = 0.472

# fix thread_pitch of nut type data (some values in mm, others in threads per inch)
#e.g. 1 mm = 0.0393 inches = 25.4 threads/inch  
comp_type_data[4].ix[comp_type_data[4].thread_pitch==1,'thread_pitch'] = 25.4
comp_type_data[4].ix[comp_type_data[4].thread_pitch==1.25,'thread_pitch'] = 20.32
comp_type_data[4].ix[comp_type_data[4].thread_pitch==1.5,'thread_pitch'] = 16.93
comp_type_data[4].ix[comp_type_data[4].thread_pitch==1.75,'thread_pitch'] = 14.51
# ---------------------------------------------------------------------------------


# generate table with all the component_ids and the corresponding types
compID_type = pd.DataFrame(index=range(len(all_attr)), columns=['attr', 'type'])
compID_type['attr'] = all_attr
for i in range(len(comp_type_data)):
  for j in range(comp_type_data[i].shape[0]):
    ind = compID_type[compID_type.attr == comp_type_data[i]. \
          component_id[j]].index.values[0]
    compID_type.ix[ind, 'type'] = comp_type_names[i]


# find components that appear at least min_count times in data
# generate new features: 
# C-rare_adaptor = sum of rare adaptor comp. with count < min_count
# C-rare_boss = sum of rare boss comp. with count < min_count
# and so on for elbow, hfl, etc
# first define a min_count for each type of component
# this is determined such that attributes with higher count than 
# ~min_count can appear in top ~60 most important attributes
# importance is determined either with Random Forest or Gradient Boosting
min_count_list = [357, 75, 33, 12, 62, 57, 53, 63, 4, 41, 7]
rare_comp = [];  freq_comp = []
for k in range(len(comp_type_names)):
  ind = comp_type_data[k].component_id.values
  temp_materials = transformed_materials.ix[:, ind]
  min_count = min_count_list[k]
  counts = zip(temp_materials.columns.values, np.sum(temp_materials))
  # [counts[x] for x in sorted(np.arange(len(counts)), key=lambda x: counts[x][1])]
  temp_materials['C-rare'] = 0.0
  for i in range(len(counts)):
    if counts[i][1] >= min_count:
      freq_comp.append(temp_materials.columns.values[i])
    else:
      rare_comp.append(temp_materials.columns.values[i])
      temp_materials.ix[:, 'C-rare'] += temp_materials.ix[:, counts[i][0]]
  transformed_materials['C-rare' + '_' + comp_type_names[k]] = temp_materials['C-rare']

  
# add new feature:'total_comp_score': a label for all comp. of same type in a tube
# each comp is encoded using a uniformly spaced label encoding from 1 to 1.09
# total_comp_score = weighted sum of all labels (with weights the number of comp.)
# range 1-1.09 was chosen to avoid collisions: 
# each tube has up to 11 comp of same type so with this encoding 10*1.09 < 11*1
# the most frequent comp of each type get the smallest labels (1.0)
weights=dict(zip(comp_type_names, [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
total_comp_score = np.zeros(materials.shape[0])
Nlabels = np.zeros(len(comp_type_names)) 
ind = [[] for x in range(len(comp_type_names))]
comp_label = ['' for x in range(len(comp_type_names))]
col_sums = transformed_materials.sum()
sort_ind = sorted(np.arange(col_sums.shape[0]),
                  key=lambda x: col_sums[x])
counts = col_sums[sort_ind][::-1]
for i in counts.index.values:
  if i in rare_comp: counts.drop(i, inplace=True)
for i in range(len(counts)):
  comp_temp = counts.index.values[i]
  if comp_temp in compID_type.attr.values:
    comp_temp_type = compID_type.type[compID_type.attr == comp_temp].values[0]
    ind[comp_type_names.index(comp_temp_type)].append(comp_temp)
for i in range(len(comp_type_names)):
  Nlabels[i] = 1 + len(ind[i])
  ind[i] += ['C-rare' + '_' + comp_type_names[i]]
  comp_label[i] = pd.Series(np.linspace(1.0,1.09,Nlabels[i]), index=ind[i])
for i in range(materials.shape[0]):
  comp_in_tube = materials.ix[i, ~materials.iloc[i].isnull()][1:].values
  for comp_temp in comp_in_tube[0::2]:
    Ncomp_temp = comp_in_tube[np.where(comp_in_tube == comp_temp)[0]+1][0]
    comp_type_temp = compID_type.type[compID_type.attr == comp_temp].values[0]
    if comp_temp in rare_comp:
      comp_temp = 'C-rare' + '_' + compID_type.type[compID_type.attr == comp_temp].values[0]
    total_comp_score[i] += Ncomp_temp*weights[comp_type_temp]* \
                           comp_label[comp_type_names.index(comp_type_temp)][comp_temp]
transformed_materials['total_comp_score'] = total_comp_score


# add new feature: N_comp_total = total number of all components in a tube
transformed_materials['N_comp_total'] = 0    
for i in range(transformed_materials.shape[0]):
  if materials.ix[i,2] > 0: 
    transformed_materials.ix[i, 'N_comp_total'] = materials.ix[i,range(2,17,2)].sum()
  else:
    transformed_materials.ix[i, 'N_comp_total'] = 0
    
    
# add new features: number of comp. of each type in a tube (Nadaptor, Nboss, etc)
for i in comp_type_names:
  transformed_materials['N'+i] = 0
for i in range(materials.shape[0]):
  for j in range(1, 16, 2):
    if not pd.isnull(materials.ix[i,j]):
      for k in range(len(comp_type_data)):
        if materials.ix[i,j] in comp_type_data[k]['component_id'].values:
          transformed_materials.ix[i, 'N' + comp_type_names[k]] += materials.ix[i, j+1]
          break
    else:
      break


# drop rare components (keep only C-rare_[type] to represent them)
transformed_materials.drop(rare_comp, axis=1, inplace=True)


# fill gaps for comp_type_data
# for numerical attributes substitute with median value
# for categorical attributes substitute with most frequent value 
for i in range(len(comp_type_names)):
  for j in comp_type_data[i].columns.values:
    if (comp_type_data[i][j]).dtype in ['float64', 'int64']:
      if comp_type_data[i][j].isnull().any():
        comp_type_data[i][j].fillna(comp_type_data[i][j].median(), inplace=True)
        print 'filling gaps for ', comp_type_names[i], '  ', j, \
              ' with ', comp_type_data[i][j].median()    
    elif (comp_type_data[i][j]).dtype in ['object']:
      if comp_type_data[i][j].isnull().any():
        freq_val = comp_type_data[i].ix[:,j].value_counts().idxmax()
        comp_type_data[i][j].fillna(freq_val, inplace=True)
        print 'filling gaps for ', comp_type_names[i], '  ', j, \
              ' with ', freq_val


# label encoding of categorical variables for comp_type_data
# labels should start from 1 and end at 1.09 (uniformly spaced)
# this is in order to avoid collisions
# e.g. 11*min_label != 10*max_label (tubes have up to 11 comp. of same type)
lbl = preprocessing.LabelEncoder()  
for i in range(len(comp_type_names)):
  for j in comp_type_data[i].columns.values[1:]:
    if (comp_type_data[i][j]).dtype in ['object']:
      new_labels = 1 + lbl.fit_transform(comp_type_data[i].ix[:,j])
      comp_type_data[i].ix[:,j] = new_labels
      if np.unique(new_labels).shape[0] > 1:
        newer_labels = np.linspace(1,1.09,np.unique(new_labels).shape[0])
        label_dict = dict(item for item in 
                     zip(sorted(np.unique(new_labels)), sorted(newer_labels)))
        newer_labels = np.array([label_dict[k] for k in new_labels])
        comp_type_data[i].ix[:,j] = newer_labels


# generate a dataframe called types which has features all of the
# features that appear in comp_type_data and indices all the TA_id
# if TA_id has 2 nut comp. and 1 elbow comp., then the corresponding
# values from nut, elbow tables will be multiplied by 2, 1 respectively
# for TA_ids that don't have certain components, corresponding fields = 0
types_columns = ['tube_assembly_id'] + \
    [x for i in range(len(comp_type_names)) \
    for x in (comp_type_data[i].columns.values[1:]+'_'+comp_type_names[i])]
types = pd.DataFrame(np.zeros((materials.shape[0], len(types_columns))), 
                     columns = types_columns)
types['tube_assembly_id'] = materials.tube_assembly_id.values
for i in range(materials.shape[0]):
  for j in range(1, 16, 2):
    if not pd.isnull(materials.ix[i,j]):
      comp1 = materials.ix[i,j]
      type1 = compID_type.ix[compID_type.attr == comp1, 'type'].values[0]
      type1_ind = comp_type_names.index(type1)
      comp1_num = materials.ix[i,j+1]
      col1 = [x + '_' + type1 for x in comp_type_data[type1_ind].columns.values[1:]]
      types.ix[i, col1] += comp1_num*comp_type_data[type1_ind]. \
      ix[comp_type_data[type1_ind].component_id == comp1,:].values[0][1:]
    else:
      break


# add tube_assembly_id as feature to transformed_materials
transformed_materials['tube_assembly_id'] = transformed_materials.index.values


# merge transformed_materials with types
transformed_materials = pd.merge(transformed_materials, types, 
                                 on='tube_assembly_id', how='left')


# save materials data
transformed_materials.to_csv('materials.csv', index=False)
