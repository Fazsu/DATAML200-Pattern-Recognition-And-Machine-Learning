##############################################################
#   COURSE: DATA.ML.200                                      #
#           Pattern Recognition and Machine Learning         #
#   Exercise Set 2: Multi-layer Percepteron (MLP)            #
#   CREATOR: MISKA R.                                        #  
#   STUDENT NUMBER: **********                               #
#   EMAIL: miska.romppainen@tuni.fi                          #
##############################################################

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

# Load the dataset
data = []; labels = []; classes = 2
for i in range(classes):
    class_path = f'class{i+1}'
    img_path = os.path.join(images,class_path) 
    for img in os.listdir(img_path):
        im = Image.open(img_path +'/'+ img)
        im = np.array(im)
        data.append(im)
        labels.append(i)
    
data = np.array(data); labels = np.array(labels)

x = data.astype('float32') / 255.0
y = np.array(labels)

# Split data into two parts - 80% for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, test_size=0.2, shuffle=True)

# Preprocess the data
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print('Training data')
print(x_train.shape)
print(len(y_train))
print(y_train)
print('Test data')
print(x_test.shape)
print(len(y_test))

# Simple Sequential structure
model = tf.keras.models.Sequential()

# Flatten 2D input image to a 1D vector; size of 12288
model.add(tf.keras.layers.Flatten(input_shape=(64,64,3)))
#print(model.output_shape) 

# Add a single layer of 10 neurons each connected to each input
model.add(tf.keras.layers.Dense(10,activation='relu'))
#print(model.output_shape)

# Add a single layer of 10 neurons each connected to each input
model.add(tf.keras.layers.Dense(10,activation='relu'))
#print(model.output_shape)

# Add a single layer of 10 neurons each connected to each input
model.add(tf.keras.layers.Dense(2,activation='softmax'))
#print(model.output_shape)

#print(model.summary())

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32)
plt.plot(history.history['loss'])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
plt.show()
