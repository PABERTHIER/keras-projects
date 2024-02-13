# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:12:30 2021

@author: Pierre-Antoine
"""
import tensorflow as tf
import tensorflow.keras as keras

(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.boston_housing.load_data();

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(20, activation='relu', input_shape=[len(train_features[0])]),
        keras.layers.Dense(1)
    ]);
    model.compile(optimizer='SGD', loss='mse', metrics=['mae', 'mse']);
    return model;
    