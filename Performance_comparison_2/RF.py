#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

dataset = "mordred"

r2_train = []
r2_test = []
mae_train = []
mae_test = []
rmse_train = []
rmse_test = []
for i in range(10):
    np.random.seed(i)
    print(i)
    data = pd.read_csv('data/{}.csv'.format(dataset))
    data['group'] = data.index // 10
    shuffled_groups = data['group'].unique()
    np.random.shuffle(shuffled_groups)
    train_groups = shuffled_groups[:21]
    test_groups = shuffled_groups[21:]
    train_data = data[data['group'].isin(train_groups)].reset_index(drop=True)
    test_data = data[data['group'].isin(test_groups)].reset_index(drop=True)
    y_train = pd.DataFrame(train_data['Yield'],columns=['Yield'])
    X_train = train_data.drop(columns=['Yield', 'Ligand_name', 'Ligand_No', 'Substrate_name', 'Substrate_No', 'group'])
    y_test = pd.DataFrame(test_data['Yield'],columns=['Yield'])
    X_test = test_data.drop(columns=['Yield', 'Ligand_name', 'Ligand_No', 'Substrate_name', 'Substrate_No', 'group'])
    param = {"n_estimators": [100, 500, 1000, 3000, 5000], "max_depth": [4, 5, 6, 7, 8]}
    reg = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param, cv=5, n_jobs=16)
    reg.fit(X_train, y_train['Yield'])
    best = reg.best_estimator_
    print(reg.best_estimator_)
    y_pred1 = best.predict(X_train)
    y_pred2 = best.predict(X_test)
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
result.to_csv('result/RF/result_{}.csv'.format(dataset))
