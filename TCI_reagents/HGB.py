#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

data = pd.read_csv('data/mordred2.csv')
data_train, data_test = train_test_split(data, train_size=300, shuffle=False)
y_train = pd.DataFrame(data_train['Yield'],columns=['Yield'])
print(y_train)
X_train = data_train.drop(columns=['Yield', 'Ligand_name', 'Ligand_smiles', 'Substrate_name'])
print(X_train)

Lig_smiles = data_test['Ligand_smiles']
Sub_name = data_test['Substrate_name']
print(Lig_smiles)
print(Sub_name)
X_test = data_test.drop(columns=['Yield', 'Ligand_name', 'Ligand_smiles', 'Substrate_name'])
print(X_test)

param = {"max_leaf_nodes": [3, 5, 7, 15], "max_depth": [4, 6, 8], "l2_regularization": [0, 0.1, 1]}
reg = GridSearchCV(HistGradientBoostingRegressor(random_state=0, max_bins=51, min_samples_leaf=5),
                   param_grid=param, cv=5, n_jobs=16)
reg.fit(X_train,y_train['Yield'])
best = reg.best_estimator_
print(reg.best_params_)
print(reg.best_score_)

y_pred = best.predict(X_test)
Yield_pred = pd.DataFrame(y_pred, columns=['Yield_pred'])
print(Yield_pred)
Lig_smiles = np.array(Lig_smiles)
Sub_name = np.array(Sub_name)
Lig_smiles = pd.DataFrame(Lig_smiles, columns=['Lig_smiles'])
Sub_name = pd.DataFrame(Sub_name, columns=['Sub_name'])

pred_HGB = pd.concat([Lig_smiles, Sub_name, Yield_pred], axis=1, join='inner')
print(pred_HGB)
pred_HGB.to_csv('result/pred_HGB.csv', index=False)
