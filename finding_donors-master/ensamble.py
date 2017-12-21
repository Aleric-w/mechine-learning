# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:56:56 2017

@author: wangpeng02
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:53:00 2017

@author: wangpeng02
"""

import warnings
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import assert_all_finite
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.exceptions import UndefinedMetricWarning

from time import time
from IPython.display import display # 允许为DataFrame使用display()
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import visuals as vs
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

## 预处理
data = pd.read_csv("census.csv")

n_records = len(data)

income_raw = data['income']
features_raw = data.drop('income', axis = 1)

skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

features = pd.get_dummies(features_raw)

# TODO：将'income_raw'编码成数字值
income = np.array([1 if i =='>50K' else 0 for i in income_raw ])

print(income)
print("n_records:{}".format(n_records))
print("income:{}".format(len(income)))

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
#print "{} total features after one-hot encoding.".format(len(encoded))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0,
                                                    stratify = income)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                    stratify = y_train)

print "Training set has {} samples.".format(X_train.shape[0])
print "Validation set has {} samples.".format(X_val.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


def train_predict(learner, sample_size, X_train, y_train, X_val, y_val): 
    results = {}
    X_sample= X_train[:sample_size]
    y_sample= y_train[:sample_size]

    start = time() # 获得程序开始时间
    learner = learner.fit(X_train,y_train)    
    end = time() # 获得程序结束时间
    results['train_time'] = end - start
    
    # TODO: 得到在验证集上的预测值
    #       然后得到对前300个训练数据的预测结果
    start = time() # 获得程序开始时间
    predictions_val = learner.predict(X_val)
    
    #   X_sample  ,y_sample ,predictions_train
    predictions_train = learner.predict(X_sample)
    end = time() # 获得程序结束时间
    results['pred_time'] = end - start        
    results['acc_train'] = accuracy_score(y_sample,predictions_train)
    results['acc_val'] = accuracy_score(y_val,predictions_val)   
    # TODO：计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_sample, predictions_train, average='macro', beta=0.5)       
    # TODO：计算验证集上的F-score
    results['f_val'] = fbeta_score(y_val, predictions_val, average='macro', beta=0.5)
    
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
    
    print ("debug position 4: {}".format(results))
    return results

'''
# TODO：初始化三个模型
clf_A = RandomForestClassifier(random_state=10)
clf_B = AdaBoostClassifier(random_state=10)

# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = (X_train.shape[0])/100
samples_10 = (X_train.shape[0])/10
samples_100 = X_train.shape[0]

# 收集学习器的结果
results = {}
for clf in [clf_A,clf_B]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_100]):
        print ("---------------------------------------------")   
        print ("debug position   samples:" ,clf_name,samples)
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)
# 对选择的三个模型得到的评价结果进行可视化
#vs.evaluate(results, accuracy, fscore)

'''
clf = AdaBoostClassifier(random_state=10)
parameters = {"algorithm":['SAMME', 'SAMME.R'],"base_estimator":[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2)]}

def score_func(y_true,y_predict):
    return fbeta_score(y_true, y_predict, average='macro', beta=0.5)
    
# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf,param_grid=parameters,scoring=make_scorer(score_func))
grid_obj=grid_obj.fit(X_train,y_train)
# TODO：用训练数据拟合网格搜索对象并找到最佳参数

# 得到estimator
best_clf = grid_obj.best_estimator_
print(best_clf)

# 使用没有调优的模型做预测
predictions = (clf.fit(X_train, y_train)).predict(X_val)
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
