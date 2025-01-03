#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


dataset = "mordred"
data = pd.read_csv('data/{}.csv'.format(dataset))
y = pd.DataFrame(data['Yield'],columns=['Yield'])
print(y)
X = data.drop(columns=['Yield', 'Ligand_name', 'Ligand_No', 'Substrate_name', 'Substrate_No'])
print(X)


r2_test = []
mae_test = []
rmse_test = []
for i in range(0, 10):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    y_pred_mean = np.mean(y_test)
    y_pred = np.full_like(y_test, y_pred_mean)
    r2_test.append(metrics.r2_score(y_test, y_pred))
    mae_test.append(metrics.mean_absolute_error(y_test, y_pred))
    rmse_test.append(metrics.root_mean_squared_error(y_test, y_pred))
    

r2_test = pd.DataFrame(data=r2_test, columns=['r2_test'])
mae_test = pd.DataFrame(data=mae_test, columns=['mae_test'])
rmse_test = pd.DataFrame(data=rmse_test, columns=['rmse_test'])
result = pd.concat([r2_test, mae_test, rmse_test], axis=1, join='inner')
result.to_csv('result/Mean/result.csv')
