#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:17:38 2021

@author: notonaoki
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from sklearn import metrics
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras import regularizers

def build_model(ar=0.2):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', activity_regularizer=regularizers.L2(ar),
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu', activity_regularizer=regularizers.L2(ar)))
    model.add(layers.Dense(64, activation='relu', activity_regularizer=regularizers.L2(ar)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


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
for i in range(0, 10):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    a_X_train = (X_train - X_train.mean()) / X_train.std()
    a_X_test = (X_test - X_train.mean()) / X_train.std()
    a_X_train = a_X_train.dropna(how='any', axis=1)
    a_X_test = a_X_test[a_X_train.columns]

    train_data = np.array(a_X_train, dtype='float32')
    train_target = np.array(y_train, dtype='float32')
    test_data = np.array(a_X_test, dtype='float32')
    test_target = np.array(y_test, dtype='float32')

    regressor = KerasRegressor(model=build_model, batch_size=-1, random_state=0, verbose=0)
    param_grid = {"epochs": [500, 1000]}
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5)
    grid_search.fit(train_data, train_target)
    print("Best estimator:", grid_search.best_params_)

    best = grid_search.best_estimator_
    y_pred1 = best.predict(train_data)
    y_pred2 = best.predict(test_data)
    r2_train.append(metrics.r2_score(train_target, y_pred1))
    r2_test.append(metrics.r2_score(test_target, y_pred2))
    print(metrics.r2_score(y_test, y_pred2))
    mae_train.append(metrics.mean_absolute_error(train_target, y_pred1))
    mae_test.append(metrics.mean_absolute_error(test_target, y_pred2))

r2_train = pd.DataFrame(data=r2_train, columns=['r2_train'])
r2_test = pd.DataFrame(data=r2_test, columns=['r2_test'])
mae_train = pd.DataFrame(data=mae_train, columns=['mae_train'])
mae_test = pd.DataFrame(data=mae_test, columns=['mae_test'])
result = pd.concat([r2_train, r2_test, mae_train, mae_test], axis=1, join='inner')
result.to_csv('result/NN/result_{}.csv'.format(dataset))
