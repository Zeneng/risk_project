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

features=trained_features.columns.tolist()
trained_features=pd.DataFrame(x_res,columns=features)
target=trained_target.name
trained_target=pd.DataFrame(y_res,columns=[target])


# Calculate information value
from WOEIV import WOE
woe = WOE()
IV= np.zeros(x_res.shape[1])
for i in range(0,x_res.shape[1]):
    IV[i]=woe.woe_single_x(x_res[i],y_res,1)[1]
    
x_res_IV=x_res[:,IV>0.8]
trained_features=trained_features.loc[:, IV>0.80]




##identify the useful features based on classfication tree method
#clf = ExtraTreesClassifier()
#clf = clf.fit(x_res, y_res)
#clf.feature_importances_  
#
#model = SelectFromModel(clf, prefit=True)
#x_new = model.transform(x_res)
#x_new.shape 
#
#index=model.get_support()

#Clustering
from Cluster import PFA
pfa = PFA(n_features=20)
pfa.fit(x_res_IV)
column_indices = pfa.indices_


trained_features=trained_features.iloc[:, column_indices]

df = pd.concat([trained_features, trained_target], axis=1, join_axes=[trained_features.index])




#build stepwise logistic model
import statsmodels.formula.api as smf
import statsmodels.api as sm

def forward_selected(data, response):
    """logistic Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = -14997,-14997
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = -smf.glm(formula, data,family=sm.families.Binomial()).fit().deviance
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.glm(formula, data,family=sm.families.Binomial()).fit()
    return model

#random cross validation to find the best fitted model

L=np.zeros(9)
for i in range(0, 9):
     #Seperate the trained file into in sample and out of sample

     outdf = df.sample(frac=0.1, replace=False)

     indf = df.drop(outdf.index)
     
     model = forward_selected(indf, 'Target')

     L[i]=sum(abs(outdf.iloc[:,-1].values-model.model.predict(model.params,outdf)))
     
     print(i)
     
     print(model.model.formula)
     
     print(model.params)
     

#build scorecard model

