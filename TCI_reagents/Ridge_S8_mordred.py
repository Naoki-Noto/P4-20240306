#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
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

a_X_train = (X_train - X_train.mean()) / X_train.std()
a_X_test = (X_test - X_train.mean()) / X_train.std()
a_X_train = a_X_train.dropna(how='any', axis=1)
a_X_test = a_X_test[a_X_train.columns]
model = RidgeCV(alphas=np.linspace(0.1, 30, num=150), cv=5)
model.fit(a_X_train, y_train['Yield'])

y_pred = model.predict(a_X_test)
Yield_pred = pd.DataFrame(y_pred, columns=['Yield_pred'])
print(Yield_pred)
Lig_smiles = np.array(Lig_smiles)
Sub_name = np.array(Sub_name)
Lig_smiles = pd.DataFrame(Lig_smiles, columns=['Lig_smiles'])
Sub_name = pd.DataFrame(Sub_name, columns=['Sub_name'])

pred_HGB = pd.concat([Lig_smiles, Sub_name, Yield_pred], axis=1, join='inner')
print(pred_HGB)
pred_HGB.to_csv('result/pred_ridge_S8_mordred.csv', index=False)
