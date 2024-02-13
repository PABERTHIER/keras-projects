#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import tensorflow.keras as keras
print(tf.__version__)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='relu', input_shape=[len(train_features[0])]),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

def plot_history(hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [Thousand Dollars$^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.legend()
    plt.ylim([0,50])

(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.boston_housing.load_data()

# get per-features statistics (mean, standard deviation) from the training set to normalize by 
train_mean = np.mean(train_features, axis=0)
train_std = np.std(train_features, axis=0)
train_features = (train_features - train_mean) / train_std

model = build_model()
# model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(
    train_features,
    train_labels,
    epochs=1000,
    verbose=0,
    validation_split=0.1,
    callbacks=[early_stop])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist)
rmse_final = np.sqrt(float(hist['val_mse'].tail(1)))
print()
print('Final Room Mean Square Error on validation set : {}'.format(round(rmse_final, 3)))
plot_history(hist)

model = tf.keras.Sequential(name="FFNN")
# Addition of a first dense layer with: 16 inputs and 8 outputs
# First layer, always define the number of inputs
model.add(tf.keras.layers.Dense(8, activation="relu", name="first_layer", input_shape=(16,)))
# For the following layers, we do not put the number of inputs we automatically take the previous number of outputs
model.add(tf.keras.layers.Dense(4, name="second_layer"))
# Shows model configuration
model.summary()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# reshape images to specify that it's a single channel
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

def preprocess_images(imgs): # should word for both a single image and multiple images
    sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
    assert sample_img.shape in [(28, 28, 1), (28, 28)], sample_img.shape
    return imgs / 255.0

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])

model = keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(train_images, train_labels, epochs=5)

print(test_images.shape)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy : ', test_acc)
