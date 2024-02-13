#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from LR_learn_c import LR_learn
from LR_predict_c import LR_predict

boston = datasets.load_boston();
# print(boston)
xt = boston.data[0::2,:]; # Even lines for learning, odd ones for verification
yt = boston.target[0::2];
xq = boston.data[1::2,:];
yq = boston.target[1::2];

# TODO
eta = 0.001;
iter_max = 10000;
eps = 0.00001;

regressor, norm_t, error = LR_learn(xt, yt, eta, iter_max, eps);
# lam = 0.001
# regressor, norm_t = LR_learn(xt, yt, eta, lam, iter_max, eps)
norm_model = StandardScaler();
norm_t = norm_model.fit(xt);
predicted = LR_predict(xq, regressor, norm_t);

fig, ax = plt.subplots();
ax.scatter(yq, predicted, edgecolors=(0, 0, 0));
ax.plot([yq.min(), yq.max()], [yq.min(), yq.max()], 'k--', lw=4);
ax.set_xlabel('Measured');
ax.set_ylabel('Predicted');
plt.show();
