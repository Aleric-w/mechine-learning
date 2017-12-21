# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 17:42:58 2017

@author: wangpeng02
"""

import numpy as np
import pandas as pd
import visuals as vs

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def median(x):
    if (len(x) % 2 == 0):
        median_v = ( np.sort(x)[len(x)/2] + np.sort(x)[(len(x)-1)/2] )/2.0
    else:
        median_v = np.sort(x)[(len(x)-1)/2]
    return median_v

##  data read and convert

data_bj=pd.read_csv('bj_housing.csv')
price_bj=data_bj['Value']
data_bj=data_bj.drop('Value',axis=1)

    
print "STATISTICS of price of beijing:"
print "   min:    {:,.2f}".format(np.min(price_bj))
print "   max:    {:,.2f}".format(np.max(price_bj))
print "   mean:   {:,.2f}".format(np.average(price_bj))
#print "   median: {:,.2f}".format(median_bj)

X_train,X_test,y_train,y_test=train_test_split(data_bj,price_bj,test_size=0.2,random_state=10)

def fit_model(X,y):
#    clf=DecisionTreeRegressor()
    clf=DecisionTreeRegressor(random_state=10)

    params={"max_depth":np.arange(3,6),"min_samples_leaf":np.arange(2,6)}
    val_bj=KFold(n_splits=10,random_state=None,shuffle=True)
#    val_bj=KFold(n_splits=10,random_state=None,shuffle=False)
    
    def scroe_1(y1,y2):
        return r2_score(y1,y2)
   
    grid=GridSearchCV(clf,param_grid=params,scoring=make_scorer(scroe_1),cv=val_bj)
    grid=grid.fit(X,y)
    #print grid.cv_results_.keys()
    print grid.cv_results_ ['params']
    return grid.best_estimator_
 
print "\n"
optimal_depth=fit_model(X_train,y_train)
print "train optimal depth:{}\nsamples_leaf:{}\n".format(optimal_depth.max_depth,optimal_depth.min_samples_leaf)
    
r2 = r2_score(y_test,optimal_depth.predict(X_test))    
print "test final score {:,.2f}".format(r2)
