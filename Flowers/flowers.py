# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:55:13 2021

@author: Pierre-Antoine
"""

# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import PIL
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd

from tensorflow import keras
print(tf.__version__)

# Download data
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# Explore data
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# The different classes are distributed in associated directories
os.listdir(data_dir)

# Show sample image
list_images = list(data_dir.glob('tulips/*'))
im_path = list_images[1]
print(im_path)
PIL.Image.open(str(im_path))

# Images have different sizes. they all need to be resized to the same size
list_images = list(data_dir.glob('tulips/*'))
im_path = list_images[2]
im = PIL.Image.open(str(im_path))
print(im.size)

# These parameters can be modified
batch_size = 32
img_height = 180
img_width = 180

# Prepare the train and validation set: divide the set in 2, resize the images and group them into batches...
train_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Get class names
class_names = train_ds.class_names
print(class_names)

# View a batch of images after preparation
plt.figure(figsize=(15, 15))
for images, labels in train_ds.take(1):
  print(len(images))
  for i in range(len(images)):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

# Optimization to pre-load images and keep them in memory and avoid going back and forth to disk
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ** Don't forget to normalize ! **
#
# A little tip: to avoid a possible forgetting problem,
# image normalization can be integrated into the neural network in the form of a preprocessing layer (see the following line of code)
normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)

def plot_history(model_history):
    axis = plt.subplots(1,2,figsize=(15,5))

    axis[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axis[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axis[0].set_title('Model Accuracy')
    axis[0].set_ylabel('Accuracy')
    axis[0].set_xlabel('Epoch')
    axis[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axis[0].legend(['train', 'val'], loc='best')

    axis[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axis[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axis[1].set_title('Model Loss')
    axis[1].set_ylabel('Loss')
    axis[1].set_xlabel('Epoch')
    axis[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axis[1].legend(['train', 'val'], loc='best')
    plt.show()

model = keras.Sequential(name="Flower")

# Convolution
input_size = (img_width, img_height)

# Convolution 1
model.add(Conv2D(32, (3, 3), input_shape=(img_width,img_height,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution 2
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution 3
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution 4
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatenning
model.add(Flatten())

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=5, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=val_ds)
model.summary()

# history = model.fit(train_ds, class_names, epochs=5)

test_loss, test_acc = model.evaluate(train_ds)
print('Test loss : ', test_loss)
print('Test accuracy : ', test_acc)

plot_history(history)
