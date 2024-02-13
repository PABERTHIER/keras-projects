# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:16:48 2021

@author: Pierre-Antoine
"""
from sklearn import linear_model

lr = linear_model.SGDRegressor(alpha=0.1)
lr.fit(xt, yt)
predictedq = lr.predict(xq)