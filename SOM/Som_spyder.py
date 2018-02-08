# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:05:21 2018

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

from minisom import MiniSom 
som = MiniSom(x = 10, y = 10,input_len = 15, sigma = 1.0, learning_rate = 0.5)

som.random_weights_init(X)

#som.train_batch()