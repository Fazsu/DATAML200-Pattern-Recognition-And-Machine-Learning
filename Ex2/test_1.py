# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras 

# Helper libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn as sk
from sklearn.model_selection import train_test_split


print('Tensorflow version: ', tf.__version__)
print('Sklearn version: ', sk.__version__)

dir = os.getcwd()
images = os.path.join(dir,'GTSRB_subset_2')

train_ds = tf.keras.utils.image_dataset_from_directory(
  images,
  validation_split=0.2,
  subset="training",
  color_mode="rgb",
  seed=123,
  image_size=(64, 64),
  batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  images,
  validation_split=0.2,
  subset="validation",
  color_mode="rgb",
  seed=123,
  image_size=(64, 64),
  batch_size=32)

train_class_names = train_ds.class_names
print("Class names:")
print(train_class_names)

print("Training data")

train_images = []
train_labels = []
test_images = []
test_labels = []

for image_batch, labels_batch in train_ds:
  for img in image_batch:
    train_images.append(img)
  for label in labels_batch:
    train_labels.append(label.numpy())

for image_batch, labels_batch in val_ds:
  for img in image_batch:
    test_images.append(img)
  for label in labels_batch:
    test_labels.append(label.numpy())

print('Training data')
print(train_images[0].shape)
print(len(train_labels))
print(train_labels)
print('Test data')
print(test_images[0].shape)
print(len(test_labels))

print('Original labels')
print(train_labels[0:9])
train_labels_onehot = tf.keras.utils.to_categorical(train_labels, num_classes=2)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=2)
print('One-hot-encoded labels')
print(train_labels_onehot[0:9,:])
foo = np.argmax(train_labels_onehot,axis=1)
print('Sanity check where one-hot-encoded are converted back to the original digits')
print(foo[0:9])

# Simple Sequential structure
model = tf.keras.models.Sequential()

# Flatten 2D input image to a 1D vector
model.add(tf.keras.layers.Flatten(input_shape=(64,64,3)))
print(model.output_shape)

# Add a single layer of 100 neurons each connected to each input
model.add(tf.keras.layers.Dense(100,activation='sigmoid'))
print(model.output_shape)

# Add a single layer of 100 neurons each connected to each input
model.add(tf.keras.layers.Dense(100,activation='sigmoid'))
print(model.output_shape)

# Add a single layer of 10 neurons each connected to each input
model.add(tf.keras.layers.Dense(10,activation='sigmoid'))
print(model.output_shape)

# Print summary
print(model.summary())

model.compile(optimizer='SGD',
              loss=tf.keras.losses.MeanSquaredError())