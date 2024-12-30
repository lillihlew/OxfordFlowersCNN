__author__ = "Lilli Lewis"
__organization__ = "COSC420, University of Otago"
__email__ = "lewli942@student.otago.ac.nz"

#imports
import numpy as np
import os
import show_methods
from PIL import Image
from load_oxford_flowers102 import load_oxford_flowers102
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout, Flatten, Dense, Reshape
from keras.models import Model, Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle, gzip
from sklearn.metrics import mean_squared_error
from keras.regularizers import l2
from keras import optimizers

#so I know when the program begins (I keep forgetting to close the output images and then the program doesn't begin)
print("STARTING again")

#Import and load images
from load_oxford_flowers102 import load_oxford_flowers102
data_train, data_validation, data_test, class_names = load_oxford_flowers102(imsize=32, fine=False) #coarse

# Explicit normalization of input (this was my work)
data_train['images'] = data_train['images'].astype('float')
data_train['images'] /= 255
data_test['images'] = data_test['images'].astype('float')
data_test['images'] /= 255
data_validation['images'] = data_validation['images'].astype('float')
data_validation['images'] /= 255

# A lot of the structure of this code, like the if statements and saving the files, is from example 3.
load_from_file = False

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
  os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'oxford_flowers102') 
net_save_name = save_name + '_2a_cnn_net.h5'
history_save_name = save_name + '_2a_cnn_net.hist'


if load_from_file and os.path.isfile(net_save_name):

  # *************************************************
  # * Loading previously trained autoencoding model *
  # *************************************************

  # Load the model from file
  print("Loading neural network from %s..." % net_save_name)
  net = tf.keras.models.load_model(net_save_name)

  # Load the training history - since it should have been created right after
  # saving the model
  if os.path.isfile(history_save_name):
     with gzip.open(history_save_name) as f:
        history = pickle.load(f)
  else:
     history = []
else:


  # ****************************************
  # * Creating and training an autoencoder *
  # ****************************************

    #I borrowed this structure (encoder.add) from chat GPT, but I'm making it my own: I picked the 
    # activation, padding, filters, number of layers, types of layers, where to put them, etc.
    #define input shape
    input_shape = (32, 32, 3)

    # Encoder
    encoder = Sequential()

    #Convolutional layers 1-2: activation is ReLU, padding is same, 32 by 32 filters, 3 channels. First input shape is 32, 32, 3
    encoder.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape))
    encoder.add(Conv2D(32, 3, activation='relu', padding='same'))
    #Batch Normalizing Layer 1
    encoder.add(tf.keras.layers.BatchNormalization())
    #Max Pooling Layer 1
    encoder.add(MaxPooling2D(pool_size=(2, 2)))

    #Convolutional layers 3-4: activation is ReLU, padding is same, 64 by 64 filters, 3 channels. 
    encoder.add(Conv2D(64, 3, activation='relu', padding='same'))
    encoder.add(Conv2D(64, 3, activation='relu', padding='same'))
    #Batch Normalizing Layer 2
    encoder.add(tf.keras.layers.BatchNormalization())
    #Max Pooling Layer 2
    encoder.add(MaxPooling2D(pool_size=(2, 2)))

    #Convolutional layers 5-6: activation is ReLU, padding is same, 128 by 128 filters, 3 channels. 
    encoder.add(Conv2D(128, 3, activation='relu', padding='same'))
    encoder.add(Conv2D(128, 3, activation='relu', padding='same'))
    #Batch Normalizing Layer 3
    encoder.add(tf.keras.layers.BatchNormalization())
    #Max Pooling Layer 3
    encoder.add(MaxPooling2D(pool_size=(2, 2)))



    # Decoder
    decoder = Sequential()

    #Convolutional Layer 1: 32 filters, 3x3 window, 2x2 strides, ReLU activation, same padding
    decoder.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    #Look at shape of model after conv layer 1
    # print("Decoder input after Conv2DTranspose 1:", decoder.input_shape)
    # print("Decoder output after Conv2DTranspose 1:", decoder.output_shape)

    #Convolutional Layer 2: 128 filters, 3x3 window, 2x2 strides, ReLU activation, same padding
    decoder.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    #Look at shape of model after conv layer 2
    # print("Decoder input after Conv2DTranspose 2:", decoder.input_shape)
    # print("Decoder output after Conv2DTranspose 2:", decoder.output_shape)

    #Convolutional Layer 3: 512 filters, 3x3 window, 2x2 strides, ReLU activation, same padding
    decoder.add(tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    #Look at shape of model after conv layer 3
    # print("Decoder input after Conv2DTranspose 3:", decoder.input_shape)
    # print("Decoder output after Conv2DTranspose 3:", decoder.output_shape)

    #Convolutional Layer 4: 3 filters, 3x3 window, 2x2 strides, sigmoid activation, same padding
    decoder.add(tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # Output layer with desired dimensions
    #Look at shape of model after conv layer 4
    # print("Output shape of decoder:", decoder.output_shape)

    #Define autoencoder
    autoencoder = tf.keras.models.Sequential([encoder, decoder])

    #From Stack Overflow!# 
    optm = optimizers.Adam(learning_rate=0.001)
    #From Stack Overflow!#

    # Compile autoencoder model
    autoencoder.compile(optimizer=optm, loss='mse')  # Use appropriate optimizer and loss function

    # Train the model for 100 epochs, shuffle the data into different batches after every epoch, batch size 32
    train_info = autoencoder.fit(data_train['images'], data_train['images'], epochs=100, shuffle=True, batch_size=32, validation_data=(data_validation['images'],data_validation['images']))

    # Save the model to file
    print("Saving autoencoder to %s..." % net_save_name)
    autoencoder.save(net_save_name)

    # Save training history to file
    history = train_info.history
    with gzip.open(history_save_name, 'w') as f:
        pickle.dump(history, f)

# All of the below code was from chat GPT
# Print autoencoder model summary
autoencoder.summary()

# Generate decoded images
decoded_images = autoencoder.predict(data_test['images'])

# Calculate pixel-wise error
pixelwise_error = np.abs(data_test['images'] - decoded_images)

# Compute mean and standard deviation
mean_error = np.mean(pixelwise_error)
std_dev_error = np.std(pixelwise_error)

# Report findings
print("Mean error per pixel:", mean_error)
print("Standard deviation of error per pixel:", std_dev_error)

#Display images
show_methods.show_data_images(images = data_test['images'][:16])
show_methods.show_data_images(images = decoded_images[:16])


# So I know program finishes without errors
print("done with ptTwoA!")