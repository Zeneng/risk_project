#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:48:21 2018

@author: IanFan
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


#prescreening by cleaning them in the excel file
#delete column with too many blanks
df_train = pd.read_excel('/Users/IanFan/Desktop/MMF2000/reduced_data.xlsx',index_col = 0)

#fill up missing values with the columns means
df_filled=df_train.fillna(df_train.mean())

#identify the target variables any default in 12 months
df_filled["Target"]=df_filled.loc[:,"t1":"t12"].sum(axis=1)
df_filled["Target"]=pd.Series(np.where(df_filled.Target.values > 0, 1, 0),
          df_filled.index)


#oversampling to incease the percentage of defaults
sum(df_filled.Target)/len(df_filled.Target)

trained_features=df_filled.drop(['TIME_KEY','t1','t2','t3' ,'t4' ,'t5', 't6', 't7' ,'t8' ,'t9' ,'t10' ,'t11' ,'t12','Target'], axis=1)
trained_target=df_filled.loc[:,'Target']

#transform into dummies
trained_features = pd.get_dummies(trained_features)

#upsampling
sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(trained_features, trained_target)


#identify the useful features based on classfication tree method
clf = ExtraTreesClassifier()
clf = clf.fit(x_res, y_res)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
x_new = model.transform(x_res)
x_new.shape 

index=model.get_support()

#further identify the useful features based on cross validation method
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(x_new, y_res)



#pre selecting the features based on the columns variance

#build logistic model


#build scorecard model

