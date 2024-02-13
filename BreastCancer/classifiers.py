# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:55:23 2021

@author: Pierre-Antoine
"""

import logging
import numpy as np
from optparse import OptionParser
##

for clf_name, clf in list_of_classifiers:
    t0 = time();
    clf.fit(xq, yq);
    train_time = time() - t0;
    
    t0 = time();
    pred = clf.predict(xt);
    test_time = time() - t0;
    
    cf = confusion_matrix(yt, pred);
    conf_matrix.append((clf_name, cf));
    acc = metrics.accuracy_score(yt, pred);
    results.append((clf_name, acc, train_time, test_time));
