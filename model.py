import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import numpy as np
import os
import warnings
import random

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

with tf.device('/GPU:0'):
    data = np.load("AugumentedSet/augumented_dataset.npz")
    images = data['data']
    labels = data['labels']

    # Fix randomness and hide warnings
    seed = 42

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    np.random.seed(seed)
    random.seed(seed)

    # model
    mobile = tfk.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg',
    )

    # Create an input layer
    inputs = tfk.Input(shape=(96, 96, 3))
    # Connect MobileNetV2 to the input
    contrast_layer = tf.keras.Sequential([tfkl.RandomContrast([0, 0.15])])

    x = contrast_layer(inputs)
    x = mobile(x)

    # Add a Dense layer with 2 units and softmax activation as the classifier
    outputs = tfkl.Dense(2, activation='softmax')(x)

    # Create a Model connecting input and output
    model = tfk.Model(inputs=inputs, outputs=outputs, name='model')

    # Compile the model with Categorical Cross-Entropy loss and Adam optimizer
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(1e-5), metrics=['accuracy'])

    history = model.fit(
        x=img_train,  # We need to apply the preprocessing thought for the MobileNetV2 network
        y=label_train,
        batch_size=8,  # 16
        epochs=500,
        validation_data=(img_val, label_val),  # We need to apply the preprocessing thought for the MobileNetV2 network
        callbacks=[
            tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=50, restore_best_weights=True)]
    ).history

    plt.figure(figsize=(15,5))
    plt.plot(history['loss'], alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_loss'], label='Vanilla CNN', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')

    plt.show()

    plt.figure(figsize=(15,5))
    plt.plot(history['accuracy'], alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_accuracy'], label='Vanilla CNN', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')

    plt.show()

    predictions = model.predict(img_test, verbose=0)
    cm = confusion_matrix(np.argmax(label_test, axis=-1), np.argmax(predictions, axis=-1))
    accuracy = accuracy_score(np.argmax(label_test, axis=-1), np.argmax(predictions, axis=-1))
    precision = precision_score(np.argmax(label_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    recall = recall_score(np.argmax(label_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    f1 = f1_score(np.argmax(label_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    print('Accuracy:', accuracy.round(4))
    print('Precision:', precision.round(4))
    print('Recall:', recall.round(4))
    print('F1:', f1.round(4))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.T, xticklabels=list(('unhealthy','healthy')), yticklabels=list(('unhealthy','healthy')), cmap='Blues', annot=True)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()