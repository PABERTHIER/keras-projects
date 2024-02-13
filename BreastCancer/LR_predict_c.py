# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:32:39 2021

@author: Pierre-Antoine
"""

import numpy as np

def LR_predict(xt, regressor, norm_t):
    x = norm_t.transform(xt);
    x = np.c_[np.ones(x.shape[0]), x];
    y = np.dot(x, regressor);
    return y
