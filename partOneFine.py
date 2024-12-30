__author__ = "Lilli Lewis"
__organization__ = "COSC420, University of Otago"
__email__ = "lewli942@student.otago.ac.nz"

#Import and load images
from load_oxford_flowers102 import load_oxford_flowers102
data_train, data_validation, data_test, class_names = load_oxford_flowers102(imsize=96, fine=True) #fine

#imports from example3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import show_methods
import os
import pickle, gzip
#import from Stack Overflow to help customize my optimizer
from keras import optimizers

#Lots of the below code is from example3.py. I've denoted which work I've put in, but I started with 1 layer and worked
# up from there. 
load_from_file = True

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
   os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'oxford_flowers102') 
net_save_name = save_name + '_pt1_fine_cnn_net.h5'
history_save_name = save_name + '_pt1_fine_cnn_net.hist'

# Show 16 train images with the corresponding labels
show_methods.show_data_images(images=data_train['images'][:16],labels=data_train['labels'][:16],class_names=class_names,blocking=False)

#define n_classes
n_classes = len(class_names)

#explicit normalization of input
data_train['images'] = data_train['images'].astype('float')
data_train['images'] /= 255
data_test['images'] = data_test['images'].astype('float')
data_test['images'] /= 255
data_validation['images'] = data_validation['images'].astype('float')
data_validation['images'] /= 255

#From example 3
if load_from_file and os.path.isfile(net_save_name):

   # ***************************************************
   # * Loading previously trained neural network model *
   # ***************************************************

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

   # ************************************************
   # * Creating and training a neural network model *
   # ************************************************

   # Create feed-forward network
   net = tf.keras.models.Sequential()

   ################################################## MY WORK ##################################################

   # Convolutional Layer 1: 3x3 window, 32 filters, input size 96x96x3, padding="same", activation is ReLU
   net.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), activation='relu',
                                  input_shape=(96, 96, 3),padding='same'))
   
   # Convolutional Layer 2: 3x3 window, 32 filters, input size 96x96x3, padding="same", activation is ReLU
   net.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), activation='relu',
                                  input_shape=(96, 96, 3),padding='same'))
   
   #Batch Normalizing Layer 1
   net.add(tf.keras.layers.BatchNormalization())

   # Max Pooling Layer 1: 2x2 window, implicit arguments - padding="valid"
   net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))



   # Convolutional Layer 3: 3x3 window, 64 filters, input size 96x96x3, padding="same", activation is ReLU
   net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), activation='relu',
                                  input_shape=(96, 96, 3),padding='same'))
   
   # Convolutional Layer 4: 3x3 window, 64 filters, input size 96x96x3, padding="same", activation is ReLU
   net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), activation='relu',
                                  input_shape=(96, 96, 3),padding='same'))
   
   #Batch Normalizing Layer 2
   net.add(tf.keras.layers.BatchNormalization())

   # Max Pooling Layer 2: 2x2 window, implicit arguments - padding="valid"
   net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))



   # Convolutional Layer 5: 3x3 window, 128 filters, input size 96x96x3, padding="same", activation is ReLU
   net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), activation='relu',
                                  input_shape=(96, 96, 3),padding='same'))
   
   # Convolutional Layer 6: 3x3 window, 128 filters, input size 96x96x3, padding="same", activation is ReLU
   net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), activation='relu',
                                  input_shape=(96, 96, 3),padding='same'))
   
   #Batch Normalizing Layer 3
   net.add(tf.keras.layers.BatchNormalization())
   
   # Max Pooling Layer 3: 2x2 window, implicit arguments - padding="valid"
   net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))



   # Flatten the output maps for fully connected layer
   net.add(tf.keras.layers.Flatten())

   #taken from example 4
   reg_wdecay = tf.keras.regularizers.l2(.2)
   #taken from example 4

   # First fully connected layer of 512 neurons 
   net.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=reg_wdecay))

   # Dropout layer 1: 
   net.add(tf.keras.layers.Dropout(0.2))

   # Second fully connected layer of 512 neurons 
   net.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=reg_wdecay))

   # Third fully connected layer with number of output neurons the same
   # as the number of classes
   net.add(tf.keras.layers.Dense(units=n_classes,activation='softmax', kernel_regularizer=reg_wdecay)) 

   # Dropout layer 2:
   net.add(tf.keras.layers.Dropout(0.2))

   ################################################## MY WORK ##################################################


   #From Stack Overflow!# 
   optm = optimizers.Adam(learning_rate=0.0001)
   #From Stack Overflow!#

   # Define training regime: Adam optimiser, sparse categorical crossentropy loss function, and accuracy metrics
   net.compile(optimizer=optm,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

#    x_valid = data_validation['images'][:16]
#    y_hat_valid = data_validation['labels'][:16]
   
   # Train the model for 300 epochs, shuffle the data into different batches after every epoch, batch size is 32 (I've played with these values a lot and gotten the most success this way)
   train_info = net.fit(data_train['images'], data_train['labels'], validation_data=(data_validation['images'], data_validation['labels']), batch_size=32, epochs=300, shuffle=True)


   # Save the model to file
   print("Saving neural network to %s..." % net_save_name)
   net.save(net_save_name)

   # Save training history to file
   history = train_info.history
   with gzip.open(history_save_name, 'w') as f:
      pickle.dump(history, f)


# *********************************÷
# *********************************************************
# * Evaluating the neural network model within tensorflow *
# *********************************************************

loss_train, accuracy_train = net.evaluate(data_train['images'],  data_train['labels'], verbose=0)
loss_test, accuracy_test = net.evaluate(data_test['images'], data_test['labels'], verbose=0)
loss_validation, accuracy_validation = net.evaluate(data_validation['images'], data_validation['labels'], verbose=0)

#Print accuracy values and model summary
print("Train accuracy (tf): %.2f" % accuracy_train)
print("Test accuracy  (tf): %.2f" % accuracy_test)
print("Val accuracy (tf): %.2f" % accuracy_validation)
print("Number of parameters: \n")
print(net.summary())

# Compute output for 16 test images
y_test = net.predict(data_test['images'][:16])
y_test = np.argmax(y_test, axis=1)

# Show true labels and predictions for 16 test images
show_methods.show_data_images(images=data_test['images'][:16],
                              labels=data_test['labels'][:16],predictions=y_test,
                              class_names=class_names,blocking=True)






# Print Confusion matrix (from Chat GPT) AS A GRAPHIC
from sklearn.metrics import confusion_matrix
import pandas as pd

# Obtain predictions from the model
y_pred = net.predict(data_test['images'])
y_pred = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(data_test['labels'], y_pred)

# Visualize the confusion matrix using Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
plt.yticks(np.arange(len(class_names)), class_names)
plt.tight_layout()

# # Add counts in the cells
# for i in range(len(class_names)):
#     for j in range(len(class_names)):
#         plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='white')

plt.show()

# Now doing Confusion matrix as a table (also from chat GPT)

# Convert confusion matrix to a pandas DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Add axis titles
conf_matrix_df.index.name = 'Actual'
conf_matrix_df.columns.name = 'Predicted'

# Print the confusion matrix table
print("Confusion Matrix:")
print(conf_matrix_df)

#End chat GPT


#Ensure file runs correctly
print("Done with file!")

