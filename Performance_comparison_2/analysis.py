#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd

print('Ridge')
for dataset in ['RDKit']:
    print(dataset)
    data = pd.read_csv(f'result/Ridge/result_{dataset}.csv')
    print(data['r2_train'].mean())
    print(data['r2_train'].std(ddof=0))
    print(data['rmse_train'].mean())
    print(data['rmse_train'].std(ddof=0))
    
print('SVM')
for dataset in ['RDKit']:
    print(dataset)
    data = pd.read_csv(f'result/SVM/result_{dataset}.csv')
    print(data['r2_train'].mean())
    print(data['r2_train'].std(ddof=0))
    print(data['rmse_train'].mean())
    print(data['rmse_train'].std(ddof=0))
    
print('HGB')
for dataset in ['RDKit']:
    print(dataset)
    data = pd.read_csv(f'result/HGB/result_{dataset}.csv')
    print(data['r2_train'].mean())
    print(data['r2_train'].std(ddof=0))
    print(data['rmse_train'].mean())
    print(data['rmse_train'].std(ddof=0))
    
print('NN')
for dataset in ['RDKit']:
    print(dataset)
    data = pd.read_csv(f'result/NN/result_{dataset}.csv')
    print(data['r2_train'].mean())
    print(data['r2_train'].std(ddof=0))
    print(data['rmse_train'].mean())
    print(data['rmse_train'].std(ddof=0))

"""
print('RF')
for dataset in ['RDKit', 'MK', 'MF2', 'mordred']:
    print(dataset)
    data = pd.read_csv(f'result/RF/result_{dataset}.csv')
    print(data['r2_train'].mean())
    print(data['r2_train'].std(ddof=0))
    print(data['rmse_train'].mean())
    print(data['rmse_train'].std(ddof=0))"""
    