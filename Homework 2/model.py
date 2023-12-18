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
from sklearn.model_selection import train_test_split


with tf.device('/GPU:0'):
    data = np.load("Dataset/cut_data.npy")
    valid_periods = np.load("Dataset/valid_periods.npy")
    categories = np.load("Dataset/categories.npy")

    data = data.reshape((data.shape[0], data.shape[1], 1))
    # shuffle of ordered data
    permutation = np.random.permutation(data.shape[0])
    data = data[permutation]
    categories = categories[permutation]

    '''
    rawdata = []
    for i in range(data.shape[0]):
        if categories[i]=='A':
            rawdata.append(data[i])
    data = np.array(rawdata)
    '''

    # split data into training set and test set (validation set provided within fit function)
    data, test_data = train_test_split(
        data, random_state=seed, test_size=.10
    )

    train_in = data[:, :-9] # input series
    train_out = data[:, -9:] # last 9 samples to predict (literally the output to check)

    test_in = test_data[:,:-9]
    test_out = test_data[:,-9:]

    input_shape = train_in.shape[1:]
    output_shape = train_out.shape[1:]
    batch_size = 256
    epochs = 500

    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # ARCHITECTURE 1
    # Encoder
    encoder1 = tfkl.LSTM(64, return_sequences=True, return_state=True, name='encoder1')(input_layer)
    # Decoder
    decoder1 = tfkl.LSTM(64, return_sequences=True, return_state=False, name='decoder1')(encoder1)
    # Attention
    attention1 = tfkl.Attention(name='attention1')([encoder1[0], decoder1])
    context1 = tfkl.Concatenate(name='context1')([decoder1, attention1])

    # ARCHITECTURE 2
    encoder2 = tfkl.LSTM(128, return_sequences=True, return_state=True, name='encoder2')(input_layer)
    decoder2 = tfkl.LSTM(128, return_sequences=True, return_state=False, name='decoder2')(encoder2)
    attention2 = tfkl.Attention(name='attention2')([encoder2[0], decoder2])
    context2 = tfkl.Concatenate(name='context2')([decoder2, attention2])

    # Join threads
    join = 0.5*tfkl.Add()([context1, context2])
    flattening = tfkl.Flatten(name='flattening')(join)
    dropout = tfkl.Dropout(0.1)(flattening)

    dense = tfkl.Dense(256, activation='relu', name='dense1')(dropout)
    dense2 = tfkl.Dense(64, activation='relu', name='dense2')(dense)
    # Prediction
    output_layer = tfkl.Dense(output_shape[0], activation='linear', name='output_layer')(dense2)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(1e-3))
    model.summary()

    history = model.fit(
        x=train_in,
        y=train_out,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=.1,
        callbacks=[
            tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True),
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