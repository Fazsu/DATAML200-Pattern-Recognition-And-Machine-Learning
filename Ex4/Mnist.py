import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

tf.config.list_physical_devices('GPU')

with tf.device('/GPU:0'):

    # load data
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    # normalize data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # add noise to images
    noise_factor = 0.2
    train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape)
    test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape)

    # clip values to (0,1) range
    train_images_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
    test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)

    # The CNN classifier model for the MNIST Fashion
    CNNmodel = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

    # Compile the model
    CNNmodel.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    # Train the model with clean images
    CNNmodel.fit(train_images, train_labels,
                  epochs=10)

    # Evaluate the model with clean test images
    test_loss, test_acc = CNNmodel.evaluate(test_images, test_labels)
    print('Classification accuracy for the clean test images:', test_acc)
    # Evaluate the model with noisy test images
    test_loss, test_acc = CNNmodel.evaluate(test_images_noisy, test_labels)
    print('Classification accuracy for the noisy test images:', test_acc)
    

    # Autoencoder for denoising
    class Denoise(Model):
        def __init__(self):
            super(Denoise, self).__init__()
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same')
            ])
            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    autoencoder = Denoise()



    # Compile and train the autoencoder model using the noisy images as both input and target
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    history = autoencoder.fit(train_images_noisy, train_images,
                    # epochs 5-10 should be enough
                    epochs=5,
                    shuffle=True,
                    validation_data=(test_images_noisy, test_images))
    
    # Check the summary to see what they do
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()
    
    # Plot the Loss
    plt.plot(history.history['loss'])
    plt.show()
    
    # Use the autoencoder to denoise the noisy test images
    denoised_test_images = autoencoder.predict(test_images_noisy)
    # Evaluate the classifier model with denoised test images
    test_loss, test_acc = CNNmodel.evaluate(denoised_test_images, test_labels)
    print('Classification accuracy for the autoencoder denoised test images:', test_acc)
    
    encoded_imgs = autoencoder.encoder(test_images).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    # Visually check the quality of denoised images on unseen test images
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):

        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(test_images_noisy[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(tf.squeeze(decoded_imgs[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()

    # Train the model with noisy images
    CNNmodel.fit(train_images_noisy, train_labels, 
                 epochs=10)
    
    test_loss, test_acc = CNNmodel.evaluate(test_images_noisy, test_labels)
    print('Classification accuracy for the noisy test images after training CNN with noisy images:', test_acc)

