#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from LR_learn_c import LR_learn, LR_learnKNN
from LR_predict_c import LR_predict

breastcancer = datasets.load_breast_cancer();
# print(breastcancer);
x = breastcancer.data;
y = breastcancer.target;

xq, xt, yq, yt = train_test_split(x, y, test_size = 0.5, random_state = 0);

#scale data
norm = StandardScaler();
norm.fit(xq);
xq = norm.transform(xq);
xt = norm.transform(xt);

# classification
logit = linear_model.LogisticRegression();
logit.fit(xq, yq);
predicted = logit.predict(xt);

# results evaluation
cf = confusion_matrix(yt, predicted);
percent_true_prediction = np.sum(predicted == yt)/len(yt);
percent_error = np.sum(predicted != yt)/len(yt);
print('array', cf);
print(percent_true_prediction);
print(percent_error);

# KNN
KNN_model = KNeighborsClassifier(n_neighbors=2);
KNN_model.fit(xq, yq);
KNN_prediction = KNN_model.predict(xt);

eta = 0.001;
iter_max = 10000;
eps = 0.00001;

# y = np.dot(x, regressor);

regressor, error = LR_learnKNN(xt, yt, eta, iter_max, eps, norm);

cfknn = confusion_matrix(yt, KNN_prediction);
print(cfknn); 

print(accuracy_score(KNN_prediction, yt));
print(classification_report(KNN_prediction, yt));

#SVM
SVC_model = SVC();
SVC_model.fit(xq, yq);
SVC_prediction = SVC_model.predict(xt);

cfsvc = confusion_matrix(yt, SVC_prediction);
print(cfsvc);

print(accuracy_score(SVC_prediction, yt));
print(confusion_matrix(SVC_prediction, yt));

#MLP
MLP_model = MLPClassifier();
MLP_model.fit(xq, yq);
MLP_prediction = MLP_model.predict(xt);

cfmlp = confusion_matrix(yt, MLP_prediction);
print(cfmlp);
#

# # TODO
# eta = 0.001;
# iter_max = 10000;
# eps = 0.00001;

# regressor, norm_t, error = LR_learn(xt, yt, eta, iter_max, eps);
# norm_model = StandardScaler();
# norm_t = norm_model.fit(xt);
# predicted = LR_predict(xq, regressor, norm_t);

# fig, ax = plt.subplots();
# # ax.scatter(yq, predicted, edgecolors=(0, 0, 0));
# ax.scatter(yq, KNN_prediction, edgecolors=(0, 0, 0));
# ax.plot([yq.min(), yq.max()], [yq.min(), yq.max()], 'k--', lw=4);
# ax.set_xlabel('Measured');
# ax.set_ylabel('Predicted');
# plt.show();
