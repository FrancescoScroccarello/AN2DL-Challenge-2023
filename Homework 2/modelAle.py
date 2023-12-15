import time

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

    # huh

    # shuffle of ordered data
    permutation = np.random.permutation(training_data.shape[0])
    training_data = training_data[permutation]
    categories = categories[permutation]

    train_val_data, test_set, train_val_categories, test_categories = train_test_split(
        training_data, categories, random_state=seed, test_size=.10, stratify=categories
    )

    training_data = train_val_data[:,:-9]
    val_data = train_val_data[:,-9:]

    test_in = test_set[:,:-9]
    test_out = test_set[:,-9:]

    input_shape = training_data.shape[1:]
    output_shape = val_data.shape[1:]
    batch_size = 256
    epochs = 500

    input_layer = tfk.layers.Input(shape=input_shape, name='Input')

    conv = tfk.layers.Conv1D(120,6,padding="same",strides=1,name="Conv")(input_layer)

    lstm1 = tfk.layers.Bidirectional(tfk.layers.LSTM(256, return_sequences=True))(conv)

    lstm2 = tfk.layers.Bidirectional(tfk.layers.LSTM(256, return_sequences=True))(lstm1)

    dropout2 = tfk.layers.Dropout(0.1)(lstm2)

    gap = tfk.layers.Flatten()(dropout2)

    output_layer = tfk.layers.Dense(output_shape[0], activation='linear', name='output_layer')(gap)

    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(1e-4))

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

    tstr = time.strftime("%d%m%Y-%H%M%S")
    model.save("Forecaster" + tstr)

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