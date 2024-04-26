#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd

data = pd.read_csv('result/pred_HGB.csv')
pred_alpha = data.query('Sub_name == "alpha-tetralone"')

pred_alpha = pred_alpha.sort_values('Yield_pred', ascending=False)
pred_alpha.to_csv('result/pred_alpha-tetralone_HGB.csv')
