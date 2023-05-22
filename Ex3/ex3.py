##############################################################
#   COURSE: DATA.ML.200                                      #
#           Pattern Recognition and Machine Learning         #
#   Exercise Set 3: Convolutional Neural Network(CNN)        #
#   CREATOR: MISKA ROMPPAINEN                                #  
#   STUDENT NUMBER: H274426                                  #
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

# The first layer is 2D convolution layer of 10 filters of the size 3 × 3 with stride 2 and ReLU activation function.
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=2, activation='relu', input_shape=(64,64,3)))

# The first layer is followed by a 2 × 2 max pooling layer.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# The max pooling layer is followed by another convolutional layer with the same parameters as the first.
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=2, activation='relu'))

# The second convolutional layer is followed by another max pooling layer of the same parameters.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# The second max pooling layer is “Flattened” and followed by a fullconnected (dense) layer of two neurons with sigmoid activation function.
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

print(model.summary())

# Compile the model
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
plt.plot(history.history['loss'])

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
plt.show()
