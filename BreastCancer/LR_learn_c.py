# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:32:42 2021

@author: Pierre-Antoine
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

def LR_learn(x, y, eta, iter_max, eps):
        norm_model = StandardScaler();
        norm_t = norm_model.fit(x);
        x = norm_t.transform(x);
        x = np.c_[np.ones(x.shape[0]), x];
        w = np.zeros(x.shape[1]);
        t = 0;
        old_error = 10 ** 3;
        diff_error = 10 ** 3;
        
        while (t < iter_max and diff_error > eps):
            h = np.dot(x, w);
            r = y - h;
            f = np.power(r, 2);
            error = 0.5 * np.mean(f);
            grad_f = - 2 * np.multiply(r, x.transpose());
            grad_error = 0.5 * np.mean(grad_f, axis=1);
            w = w - eta * grad_error;
            diff_error = np.abs(old_error * error);
            old_error = error;
            t = t + 1;
            # eta = eta / 2;
        print('learning process finished with iter:', t, 'and convergence: ', diff_error);
        return w, norm_t, error; 

def LR_learnKNN(x, y, eta, iter_max, eps, norm):
        x = norm.transform(x);
        x = np.c_[np.ones(x.shape[0]), x];
        w = np.zeros(x.shape[1]);
        y = np.dot(x, w);
        t = 0;
        old_error = 10 ** 3;
        diff_error = 10 ** 3;
        
        while (t < iter_max and diff_error > eps):
            h = np.dot(x, w);
            r = y - h;
            f = np.power(r, 2);
            error = 0.5 * np.mean(f);
            grad_f = - 2 * np.multiply(r, x.transpose());
            grad_error = 0.5 * np.mean(grad_f, axis=1);
            w = w - eta * grad_error;
            diff_error = np.abs(old_error * error);
            old_error = error;
            t = t + 1;
        print('learning process finished with iter:', t, 'and convergence: ', diff_error);
        return w, error;
