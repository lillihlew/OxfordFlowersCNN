__author__ = "Lilli Lewis"
__organization__ = "COSC420, University of Otago"
__email__ = "lewli942@student.otago.ac.nz"

#Imports 
import tensorflow as tf
import numpy as np
import os
import pickle, gzip
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#so I know when the program begins (I keep forgetting to close the output images and then the program doesn't begin)
print("STARTING")

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
load_from_file = True

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
    os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'oxford_flowers102') 
net_save_name = save_name + '_2b_cnn_net.h5'
history_save_name = save_name + '_2b_cnn_net.hist'

if load_from_file and os.path.isfile(net_save_name):

    # **********************************************
    # * Loading previously trained diffusion model *
    # **********************************************

    # Load the model from file
    print("Loading diffusion model from %s..." % net_save_name)
    model = tf.keras.models.load_model(net_save_name)

    # Load the training history - since it should have been created right after
    # saving the model
    if os.path.isfile(history_save_name):
        with gzip.open(history_save_name) as f:
            history = pickle.load(f)
    else:
        history = []
else:
    
    # *******************************************
    # * Creating and training a diffusion model *
    # *******************************************
    # This model is exactly the same as my autoencoder, so I removed the comments

    #define input shape
    input_shape = (32, 32, 3)

    # Define diffusion model architecture
    model = Sequential()
    # Add convolutional layers for diffusion steps (my exact encoder model)
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add convolutional transpose layers for reverse diffusion steps (my exact decoder model)
    model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')) 

    #From Stack Overflow!# 
    # Define the optimizer
    optm = Adam(learning_rate=0.001)
    #From Stack Overflow!# 

    # Compile autoencoder model
    model.compile(optimizer=optm, loss='mse')  # Use appropriate optimizer and loss function

    #Making noisy data (This is from The Keras Blog, I added the validation lines)
    noise_factor = 0.5
    x_train_noisy = data_train['images'] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data_train['images'].shape) 
    x_test_noisy = data_test['images'] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data_test['images'].shape) 
    x_val_noisy = data_validation['images'] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data_validation['images'].shape)     
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    x_val_noisy = np.clip(x_val_noisy, 0., 1.)

    #Everything below originated from chat GPT or example 3, I have edited it to suit my purposes.

    # Train the diffusion model with noisy and clean image pairs, batch size of 32, 100 epochs
    train_info = model.fit(x_train_noisy, data_train['images'], epochs=100, batch_size=32, validation_data=(x_val_noisy, data_validation['images']))

    # Create a model that takes noisy images as input and outputs denoised images at different stages
    model_denoise = tf.keras.models.Model(inputs=model.input, 
                                           outputs=[model.layers[5].output, model.layers[9].output, model.output]) #saving images at layers 5 and 9

    # Generate denoised images using the model
    denoised_images_1, denoised_images_2, denoised_images_final = model_denoise.predict(x_test_noisy)

    # Save the model to file
    print("Saving diffusion model to %s..." % net_save_name)
    model.save(net_save_name)

    # Save the model and training history
    history = train_info.history
    with gzip.open(history_save_name, 'wb') as f:
        pickle.dump(history, f)

    # Combine channels into a single image for visualization #################################### indented below here
    n_examples = 5
    plt.figure(figsize=(14, 6))
    for i in range(n_examples):
        # Clean image
        plt.subplot(n_examples, 5, 5*i + 1)
        plt.imshow(data_test['images'][i])
        plt.title('Clean Image')
        plt.axis('off')

        # Original noisy image
        plt.subplot(n_examples, 5, 5*i + 2)
        plt.imshow(x_test_noisy[i])
        plt.title('Noisy Image')
        plt.axis('off')

        # Intermediate denoised image 1
        combined_image = np.mean(denoised_images_1[i], axis=-1)
        plt.subplot(n_examples, 5, 5*i + 3)
        plt.imshow(combined_image, cmap='viridis')
        plt.title('Intermediate Denoised: Layer 5')
        plt.axis('off')

        # Intermediate denoised image 2
        combined_image = np.mean(denoised_images_2[i], axis=-1)
        plt.subplot(n_examples, 5, 5*i + 4)
        plt.imshow(combined_image, cmap='viridis')
        plt.title('Intermediate Denoised: Layer 9')
        plt.axis('off')

        # Final denoised image
        plt.subplot(n_examples, 5, 5*i + 5)
        plt.imshow(denoised_images_final[i])
        plt.title('Final Denoised')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate pixel-wise error
    pixelwise_error = np.abs(data_test['images'] - denoised_images_final)

    # Compute mean and standard deviation
    mean_error = np.mean(pixelwise_error)
    std_dev_error = np.std(pixelwise_error)

    # Report findings
    print("Mean error per pixel:", mean_error)
    print("Standard deviation of error per pixel:", std_dev_error)

# Print diffusion model summary
model.summary()

#Everything below here I found using chat GPT, but I've corrected and modified it to work for my specifications

# Generate a random image
random_image = np.random.rand(1, 32, 32, 3)  # Assuming input shape is (32, 32, 3)

# Use the trained model to denoise the noisy image
denoised_random_image = model.predict(random_image)

#I know creating this model (below line) is a repeat line of code, but I want it to happen in either case in this program, and I only want the  
#   images extracted from the model above to appear when load_from_file = False, so I'm adding it again (instead of moving the line outside)
#   of the if block
# Create a model that takes noisy images as input and outputs denoised images at different stages
model_denoise = tf.keras.models.Model(inputs=model.input, 
                                           outputs=[model.layers[5].output, model.layers[9].output, model.output]) #saving images at layers 5 and 9

# Another repeat line, same reason!
# Use the trained model to denoise the noisy image and get intermediate denoised images at layers 5 and 9
denoised_random_image_layer_5, denoised_random_image_layer_9, denoised_random_image_final = model_denoise.predict(random_image)


# Visualize the original random image, the noisy image, and the intermediate denoised images at layers 5 and 9, and the final denoised image
plt.figure(figsize=(14, 5))
plt.subplot(1, 4, 1)
plt.imshow(random_image.squeeze())
plt.title('Original Random Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(denoised_random_image_layer_5[0, :, :, 0], cmap='viridis')
plt.title('Intermediate Denoised: Layer 5') 
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(denoised_random_image_layer_9[0, :, :, 0], cmap='viridis')
plt.title('Intermediate Denoised: Layer 9')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(denoised_random_image_final.squeeze())
plt.title('Final Denoised Random Image')
plt.axis('off')

plt.tight_layout()
plt.show()


#Repeating the calculations specific to the random image 
# Calculate and report pixel-wise error
pixelwise_error = np.abs(random_image - denoised_random_image) #(this was my code!)

# Compute mean and standard deviation
mean_error = np.mean(pixelwise_error)
std_dev_error = np.std(pixelwise_error)

# Report findings
print("Mean error per pixel:", mean_error)
print("Standard deviation of error per pixel:", std_dev_error)

# So I know program finishes without errors
print("done with ptTwoB!")