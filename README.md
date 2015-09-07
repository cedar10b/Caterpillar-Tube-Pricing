# **Kaggle Caterpillar Tube Pricing Competition**

### **Summary**

The goal of this project is to **predict the price of industrial tube assemblies** that Caterpillar is using in its construction and mining equipment. Currently, Caterpillar relies on a variety of suppliers to manufacture these tube assemblies, each having their own unique pricing model. The competition provides detailed information about the physical characteristics of the tubes and their components, as well as annual usage data. The goal of the competition is to develop a machine learning model that minimizes the **Root Mean Squared Logarithmic Error (RMSLE)**. My best submission achieved a score of about **0.2125** on the private leaderboard and was ranked in the **top 3% among more than 1300 teams**.

### **Data pre-processing**

The dataset is comprised of 21 relational tables. The ER diagram below (posted by EaswerC in the Kaggle forum) shows how these tables are associated to each other. Each tube assembly may have one or more components, and there are 11 different types of components each one having its own features. Combining these tables together into a single dataframe and generating new features from the existing ones is one of the challenges of the competition.

 ![](https://kaggle2.blob.core.windows.net/forum-message-attachments/83029/2665/CAT_ER.png?sv=2012-02-12&se=2015-09-09T20%3A22%3A23Z&sr=b&sp=r&sig=cz1jf4%2Fv93MSxLEANFN9pYKdhyg%2FaV9O8%2BPk%2BPpIAOo%3D)

There is a large number of features that either have missing values or need cleaning. For numerical features, missing or corrupted values were replaced with the **median value**, and for categorical features, missing or corrupted values were replaced with the **most frequent value**.

There are also features that are associated only with a specific type of component (e.g. adaptors) and therefore can only describe tube assemblies that have this type of component. For tube assemblies that don't have this type of component, I filled in the gaps with zeros for numerical features and with an additional label for categorical features. 

Categorical features were transformed into numerical format either by replacing each label with a number (**label encoding**) or with **one-hot encoding**. Specifically for specs and component_ids, I used one-hot encoding, and for all others, label encoding. Both label and one-hot encoding were combined with hashing, i.e. for label encoding labels with low counts were all represented with the same numerical value, and for one-hot encoding rare features were all combined together as one feature.

Last, I transformed the response variable (i.e. the price of the tube assemblies) by taking the 1/20 power. Other values for **power transformation** as well as **log transformation** were also explored but the 1/20 power gave the lowest Cross-Validation (CV) error.

### **Feature engineering**

I created the following features:

From quote_date: **year**, **month**, **week**, **day**, **day of year**, **day of week**

From specs: **total number of specs in a tube**, and **total_specs_score**. The total_specs_score is the sum of all individual specs labels (encoded as numbers) in a tube. The labels are carefully chosen to avoid collisions (e.g. tubes with N specs will always have higher total_specs_score than tubes with N-1 specs).  

From Bill_of_materials: **total number of all components in a tube**, **total number of components for each component type**, **and total_comp_score** (defined similarly to total_specs_score but here the sum is weighted with the number of times that each component is used in the tube). 

From Comp_[type]: I used all the features for each one of the 11 different component types. However, the value of each feature was modified to take into account that one tube may have multiple components of the same type. For example if one tube has 2 adaptors with component_id='C-1230' and 4 more adaptors with component_id='C-1695', the weight feature of the adaptor type will be a **weighted sum** of the 'C-1230' and 'C-1695' components with the weights of the sum equal to 2 and 4 respectively.

### **Feature selection**

There are totally 301 features in the dataset after the pre-processing of the data. The feature selection is performed in 4 steps:

1. Using the transformed cost, I applied the f_regression function of the Scikit-learn feature_selection module on the data. The f_regression function computes first the cross correlation between a feature and the response variable and then computes the **F score** and the corresponding p-value. At the end of this step, each feature is associated with an F score and a p-value.

2. For every pair of features that have **pearson correlation** larger than 0.95, I eliminated the feature of the pair that had lower F score (computed in step 1). At the end of this step, about half of the features were eliminated.

3. I applied either a **Random Forest Regressor** (with 2000 trees) or a **Gradient Boosting Regressor** (with n_estimators = 4000 and learning_rate = 0.02) to rank the remaining features. Some of my models used the RF method and some the GB method.

4. The exact number of features used in the model was determined with CV. My best single model used the top **72 features** (ranked with Random Forest Regressor).

One interesting observation is that models that used the GB method for feature ranking achieved the lowest CV error with only about 40 features while models that used the RF method needed at least 60-80 features for best CV score. It should be probably expected that using the same method both for feature selection and modeling is a better option than using different methods.

### **Machine learning Model**

My best single model used the **XGBoost** library and a **bagging** metaestimator and achieved a score of 0.2124 in the private leaderboard. The parameters of the model are: num_round = 9000, eta = 0.01, max_depth = 12, min_child_weight = 18, subsample = 0.7, colsample_bytree = 0.65. The bagging metaestimator was applied by running the XGBoost model 20 times using a different sample (with replacement) of the training  data each time and different seed. Finally the average of the 20 bagged models was taken. Other machine learning methods such as Random Forests, Extra Trees Regressor, Support Vector Regressor, KNN, Adaptive Boosting, and Neural Nets were also explored but none of them had as good performance as the XGBoost method.

The parameters of the XGBoost model were tuned with a **5-fold Cross-Validation**. It should be mentioned that in order to implement CV properly, tube assemblies with the same ID should be all assigned to the same CV fold. Otherwise, the CV error is underestimated and it's not consistent with the public and private leaderboards.  

The final model was an **ensemble of 5 single XGBoost models** weighted with weights ranging from about 0.05 to 0.45 These models may have differences in the pre-processing of the data, the feature engineering part, the feature selection method, the exact number of features that were used by the model, and the parameters of the XGBoost model. Although the ensemble model achieved a better score in the public leaderboard than any of the single models, the best single model had a better score than the ensemble model in the private leaderboard. The ensemble model may have overfitted the public leaderboard to some extent. In addition, one of the reasons that the ensemble model did not perform significantly better than the single models is because all the single models were developed using the same algorithm and there was not enough diversity in these models. A better ensembling method is required to significantly improve the performance of the best single model.



### **Conclusions**

Due to the complicated form of the input data, the data pre-processing and engineering of new features were very important parts of this competition. The XGBoost method had clearly the best performance among all the standard machine learning methods. The new features that I created were very important and helped the best single XGBoost model to achieve a very good score. However, the ensemble model did not have enough diversity to significantly improve the score of the best single model, and perhaps it overfitted the public leaderboard to some extent. Nonetheless, the ensemble model was ranked on the private leaderboard in the top 3% among more than 1300 teams. Once again, **creativity** in feature engineering and model ensembling turned out to be the most important factor in the competition!

