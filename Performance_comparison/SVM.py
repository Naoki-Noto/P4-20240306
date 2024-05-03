#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn import metrics


dataset = "mordred"
data = pd.read_csv('data/{}.csv'.format(dataset))
y = pd.DataFrame(data['Yield'],columns=['Yield'])
print(y)
X = data.drop(columns=['Yield', 'Ligand_name', 'Ligand_No', 'Substrate_name', 'Substrate_No'])
print(X)


r2_train = []
r2_test = []
mae_train = []
mae_test = []
rmse_train = []
rmse_test = []
for i in range(0, 10):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    a_X_train = (X_train - X_train.mean()) / X_train.std()
    a_X_test = (X_test - X_train.mean()) / X_train.std()
    a_X_train = a_X_train.dropna(how='any', axis=1)
    a_X_test = a_X_test[a_X_train.columns]
    param = {'C':[1e+04, 1e+05, 1e+06, 1e+07, 1e+08, 1e+09],
             'gamma':[1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04],
             'epsilon':[0.1, 1, 10, 50, 100]}
    reg = GridSearchCV(SVR(kernel='rbf'), param_grid=param, cv=5, n_jobs=16)
    reg.fit(a_X_train,y_train['Yield'])
    best = reg.best_estimator_
    print(reg.best_estimator_)
    y_pred1 = best.predict(a_X_train)
    y_pred2 = best.predict(a_X_test)
    r2_train.append(metrics.r2_score(y_train, y_pred1))
    r2_test.append(metrics.r2_score(y_test, y_pred2))
    print(metrics.r2_score(y_test, y_pred2))
    mae_train.append(metrics.mean_absolute_error(y_train, y_pred1))
    mae_test.append(metrics.mean_absolute_error(y_test, y_pred2))
    rmse_train.append(metrics.root_mean_squared_error(y_train, y_pred1))
    rmse_test.append(metrics.root_mean_squared_error(y_test, y_pred2))
    
    
r2_train = pd.DataFrame(data=r2_train, columns=['r2_train'])
r2_test = pd.DataFrame(data=r2_test, columns=['r2_test'])
mae_train = pd.DataFrame(data=mae_train, columns=['mae_train'])
mae_test = pd.DataFrame(data=mae_test, columns=['mae_test'])
rmse_train = pd.DataFrame(data=rmse_train, columns=['rmse_train'])
rmse_test = pd.DataFrame(data=rmse_test, columns=['rmse_test'])
result = pd.concat([r2_train, r2_test, mae_train, mae_test, rmse_train, rmse_test], axis=1, join='inner')
result.to_csv('result/SVM/result_{}.csv'.format(dataset))
