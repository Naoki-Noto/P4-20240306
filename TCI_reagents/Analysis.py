#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""

import pandas as pd

for dataset in ['HGB_S8', 'HGB_S10', 'SVM_S8', 'Ridge_S8', 'RF_S8', 'Ridge_S8_mordred']:
    data = pd.read_csv(f'result/pred_{dataset}.csv')
    if dataset in ['HGB_S8', 'SVM_S8', 'Ridge_S8', 'RF_S8']:
        pred_alpha = data.query('Sub_name == "alpha-tetralone"')
        pred_alpha = pred_alpha.sort_values('Yield_pred', ascending=False)
        pred_alpha.to_csv(f'result/pred_asc_{dataset}.csv')
    
    else:
        pred_cyclo = data.query('Sub_name == "dicyclohexyl_ketone"')
        pred_cyclo = pred_cyclo.sort_values('Yield_pred', ascending=False)
        pred_cyclo.to_csv(f'result/pred_asc_{dataset}.csv')
        