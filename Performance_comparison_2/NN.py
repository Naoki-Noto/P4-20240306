#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:17:38 2021

@author: notonaoki
"""

import numpy as np
import pandas as pd
from keras import models
from keras import layers
from sklearn import metrics
from scikeras.wrappers import KerasRegressor
from keras import regularizers

def build_model(ar=0.1):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', activity_regularizer=regularizers.L2(ar),
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu', activity_regularizer=regularizers.L2(ar)))
    model.add(layers.Dense(64, activation='relu', activity_regularizer=regularizers.L2(ar)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


dataset = "RDKit"

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
    a_X_train = (X_train - X_train.mean()) / X_train.std()
    a_X_test = (X_test - X_train.mean()) / X_train.std()
    a_X_train = a_X_train.dropna(how='any', axis=1)
    a_X_test = a_X_test[a_X_train.columns]

    train_data = np.array(a_X_train, dtype='float32')
    train_target = np.array(y_train, dtype='float32')
    test_data = np.array(a_X_test, dtype='float32')
    test_target = np.array(y_test, dtype='float32')

    regressor = KerasRegressor(model=build_model, batch_size=-1, random_state=0, verbose=0, epochs=100)
    regressor.fit(train_data, train_target)

    y_pred1 = regressor.predict(train_data)
    y_pred2 = regressor.predict(test_data)
    r2_train.append(metrics.r2_score(train_target, y_pred1))
    r2_test.append(metrics.r2_score(test_target, y_pred2))
    print(metrics.r2_score(y_test, y_pred2))
    mae_train.append(metrics.mean_absolute_error(train_target, y_pred1))
    mae_test.append(metrics.mean_absolute_error(test_target, y_pred2))
    rmse_train.append(metrics.root_mean_squared_error(y_train, y_pred1))
    rmse_test.append(metrics.root_mean_squared_error(y_test, y_pred2))

r2_train = pd.DataFrame(data=r2_train, columns=['r2_train'])
r2_test = pd.DataFrame(data=r2_test, columns=['r2_test'])
mae_train = pd.DataFrame(data=mae_train, columns=['mae_train'])
mae_test = pd.DataFrame(data=mae_test, columns=['mae_test'])
rmse_train = pd.DataFrame(data=rmse_train, columns=['rmse_train'])
rmse_test = pd.DataFrame(data=rmse_test, columns=['rmse_test'])
result = pd.concat([r2_train, r2_test, mae_train, mae_test, rmse_train, rmse_test], axis=1, join='inner')
result.to_csv('result/NN/result_{}.csv'.format(dataset))
