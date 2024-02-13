# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:30:57 2021

@author: Pierre-Antoine
"""
# pip install tensorflow
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_functions import build_model

print(tf.__version__);
model = keras.Sequential(name="FFNN");
model.add(keras.layers.Dense(8, name="first_layer", input_shape=(16,)));
model.add(keras.layers.Dense(4,  name="second_layer"));
model.summary();

(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.boston_housing.load_data();
print(train_features.shape);

model = build_model();

model.summary();

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50);
history = model.fit(
  train_features,
  train_labels,
  epochs=1000,
  verbose=0,
  validation_split=0.1,
  callbacks=[early_stop, PrintDot()]);
