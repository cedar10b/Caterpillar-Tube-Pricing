
"""
Preprocessing of Tube and Tube_End_Form data
includes data cleaning, label encoding, and grouping of rare labels
"""

import pandas as pd
from sklearn import preprocessing


# read tube data
tube = pd.read_csv('../data/Tube.csv')


#corrections for length provided by Kaggle admin in forum
corrections = {'TA-00152': 19, 'TA-00154': 75, 'TA-00156': 24, 
               'TA-01098': 10, 'TA-01631': 48, 'TA-03520': 46,
               'TA-04114': 135, 'TA-17390': 40, 'TA-18227': 74,
               'TA-18229': 51}
for key, value in corrections.iteritems():
  tube.ix[tube['tube_assembly_id'] == key, 'length'] = value


# substitute NaNs in material_id column with most frequent value
most_freq_mat_id = tube.material_id.groupby \
                   (tube.material_id).count().idxmax()
tube['material_id'] = tube.material_id.fillna(most_freq_mat_id)


# substitute missing values in bend_radius column with most frequent value
most_freq_bend_rad = tube.bend_radius.groupby \
                     (tube.bend_radius).count().idxmax()
tube.ix[tube.bend_radius == 9999.00, 'bend_radius'] = most_freq_bend_rad
             

# read tube_end_form data
tube_end_form = pd.read_csv('../data/Tube_End_Form.csv')


# make end_form_id column the index of the dataframe
tube_end_form.index = tube_end_form['end_form_id']
tube_end_form = tube_end_form.drop(['end_form_id'], axis=1)


# generate 2 new features that show whether 
# end_a and end_x features have forming
tube = pd.merge(tube, tube_end_form, left_on='end_a', 
                  right_index=True, how='left', sort=False)
names = tube.columns.values; names[-1] = 'forming_a'                  
tube.columns = names                
tube = pd.merge(tube, tube_end_form, left_on='end_x', 
                  right_index=True, how='left', sort=False) 
names = tube.columns.values; names[-1] = 'forming_x'                  
tube.columns = names


# set forming_a and forming_x equal to 'No' 
# for tubes that have end_a, end_x equal to NONE 
tube.ix[tube['end_a'] == 'NONE', 'forming_a'] = 'No'
tube.ix[tube['end_x'] == 'NONE', 'forming_x'] = 'No'
#end_form_id from tube_end_form has 9999 values (forming = No)
#end_x has also 9999 values which make forming_x = No. that's OK.


# eliminate low-count material_ids:
# material_ids with count < min_count should be 
# grouped together with the same label
min_count = 100.
counts = tube.material_id.value_counts()
for i in range(len(counts)):
  if counts[i] < min_count:
    tube.ix[tube.material_id == counts.index.values[i], \
            'material_id'] = 'SP-rare'


# label encoding for categorical features
lbl = preprocessing.LabelEncoder()
for i in ['material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 
          'end_x_2x', 'end_a', 'end_x', 'forming_a', 'forming_x']:
  tube[i] = 1 + lbl.fit_transform(tube[i].values)


# save tube data
tube.to_csv('tube.csv', index=False)

