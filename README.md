# **Kaggle Caterpillar Tube Pricing Competition**

### **Summary**

The goal of this project is to predict the price of industrial tube assemblies that Caterpillar is using in its construction and mining equipment. Currently, Caterpillar relies on a variety of suppliers to manufacture these tube assemblies, each having their own unique pricing model. The competition provides detailed information about the physical characteristics of the tubes and their components, as well as annual usage data. The goal of the competition is to develop a machine learning model that minimizes the **Root Mean Squared Logarithmic Error (RMSLE)**. My best submission achieved a score of about **0.2125** on the private leaderboard and was ranked in the **top 3% among more than 1300 teams**.

### **Data pre-processing**

The dataset is comprised of 21 relational tables. The ER diagram below (posted by EaswerC in the Kaggle forum) shows how these tables are associated to each other. Each tube assembly may have one or more components, and there are 11 different types of components each one having its own features. Combining these tables together into a single dataframe and generating new features from the existing ones is one of the challenges of the competition.

 ![](https://kaggle2.blob.core.windows.net/forum-message-attachments/83029/2665/CAT_ER.png?sv=2012-02-12&se=2015-09-09T20%3A22%3A23Z&sr=b&sp=r&sig=cz1jf4%2Fv93MSxLEANFN9pYKdhyg%2FaV9O8%2BPk%2BPpIAOo%3D)

There is a large number of features that either have missing values or need cleaning. For numerical features, missing or corrupted values were replaced with the **median value**, and for categorical features, missing or corrupted values were replaced with the **most frequent value**.

There are also features that are associated only with a specific type of component (e.g. adaptors) and therefore can only describe tube assemblies that have this type of component. For tube assemblies that don't have this type of component, I filled in the gaps with zeros for numerical features and with an additional label for categorical features. 

Categorical features were transformed into numerical format either by replacing each label with a number (**label encoding**) or with **one-hot encoding**. Specifically for specs and component_ids, I used one-hot encoding, and for all others, label encoding. Both label and one-hot encoding were combined with hashing, i.e. for label encoding labels with low counts were all represented with the same numerical value, and for one-hot encoding rare features were all combined together as one feature.

### **Feature engineering**

I created the following features:

From the quote_date: **year**, **month**, **week**, **day**, **day of year**, **day of week**

From the specs: **total number of specs in a tube**, and **total_specs_score**. The total_specs_score is the sum of all individual specs labels (encoded as numbers) in a tube. The labels are carefully chosen to avoid collisions (e.g. tubes with N specs will always have higher total_specs_score than tubes with N-1 specs).  

From Bill_of_materials: **total number of all components in a tube**, **total number of components for each component type**, **and total_comp_score** (defined similarly to total_specs_score but here the sum is weighted with the number of times that each component is used in the tube). 

From Comp_[type]: I used all the features for each one of the 11 different component types. However, the value of each feature was modified to take into account that one tube may have multiple components of the same type. For example if one tube has 2 adaptors with component_id='C-1230' and 4 more adaptors with component_id='C-1695', the weight feature of the adaptor type will be a **weighted sum** of the 'C-1230' and 'C-1695' components with the weights of the sum equal to 2 and 4 respectively.

