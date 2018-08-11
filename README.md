# risk_project retail credit model
This model is using logistic regression to model retail default probability.

The main file is mmf2000.py. WOEIV.py is used to obtain the information value and Cluster.py is used to group highly correlated variables.

1, The first step is to clean the dataset, columns with more than 50% empty cells have been deleted in excel files.

2, The target variable has been identified, which is defaults within next 12 months.

3, Default number has been oversampled to increase the percentage of defaults.

4, Then calculate the information value of each variables and delete the insignificant ones. 

5, Using clustering techniques to identify the highly correlated variables and select one variables from each cluster

6, Using stepwise and cross validation technique to train the model.

7, Finally, test the model on the out of time data set.
