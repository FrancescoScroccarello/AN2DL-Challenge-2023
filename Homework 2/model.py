seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(seed)

import logging

import random
random.seed(seed)

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


with tf.device('/GPU:0'):
    training_data = np.load("Dataset/cut_data.npy")
    valid_periods = np.load("Dataset/valid_periods.npy")
    categories = np.load("Dataset/categories.npy")

    training_data = training_data.reshape((training_data.shape[0],training_data.shape[1],1))
    categories = np.array([ord(i)-65 for i in categories])

    # shuffle of ordered data
    permutation = np.random.permutation(training_data.shape[0])
    training_data = training_data[permutation]
    categories = categories[permutation]

    train_val_data, test_set, train_val_categories, test_categories = train_test_split(
        training_data, categories, random_state=seed, test_size=.20, stratify=categories
    )

    training_data = train_val_data[:,:-9]
    val_data = train_val_data[:,-9:]

    test_in = test_set[:,:-9]
    test_out = test_set[:,-9:]

    input_shape = training_data.shape[1:]
    output_shape = val_data.shape[1:]
    batch_size = 256
    epochs = 200

    # time series classification
    # Build the neural network layer by layer
    # Define the input layer with the specified shape
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Add a Bidirectional LSTM layer with 64 units
    x = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True, name='lstm'), name='bidirectional_lstm')(input_layer)

    # Add a 1D Convolution layer with 128 filters and a kernel size of 3
    x = tfkl.Conv1D(128, 3, padding='same', activation='relu', name='conv')(x)

    # Add a final Convolution layer to match the desired output shape
    output_layer = tfkl.Conv1D(output_shape[1], 3, padding='same', name='output_layer')(x)

    # Calculate the size to crop from the output to match the output shape
    crop_size = output_layer.shape[1] - output_shape[0]

    # Crop the output to the desired length
    output_layer = tfkl.Cropping1D((0,crop_size), name='cropping')(output_layer)
    output_layer = tfkl.Reshape((9,))(output_layer)

    # Construct the model by connecting input and output layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='CONV_LSTM_model')

    # Compile the model with Mean Squared Error loss and Adam optimizer
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())

    model.summary()

    history = model.fit(
        x=training_data,
        y=val_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=.1,
        callbacks=[
            tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=12, restore_best_weights=True),
            tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=10, factor=0.1, min_lr=1e-5)
        ]
    ).history

    best_epoch = np.argmin(history['val_loss'])
    plt.figure(figsize=(17, 4))
    plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Mean Squared Error')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(18, 3))
    plt.plot(history['lr'], label='Learning Rate', alpha=.8, color='#ff7f0e')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    model.save("Forecaster")

    predictions = model.predict(test_in, verbose=0)

    # Predict the test set using the model
    predictions = model.predict(test_in, verbose=0)

    # Print the shape of the predictions
    print(f"Predictions shape: {predictions.shape}")

    # Calculate and print Mean Squared Error (MSE)
    mean_squared_error = tfk.metrics.mean_squared_error(test_out.flatten(), predictions.flatten()).numpy()
    print(f"Mean Squared Error: {mean_squared_error}")

    # Calculate and print Mean Absolute Error (MAE)
    mean_absolute_error = tfk.metrics.mean_absolute_error(test_out.flatten(), predictions.flatten()).numpy()
    print(f"Mean Absolute Error: {mean_absolute_error}")