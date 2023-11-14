import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras_cv
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers as tfkl
from tensorflow import keras as tfk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
seed = 42
np.random.seed(seed)

with tf.device('/GPU:0'):
    data = np.load("UndersampledDataset/undersampled_dataset.npz")
    images = data['data']
    labels = data['labels']

    for i in range(len(labels)):
        if labels[i] == 'healthy':
            labels[i] = 0;
        else:
            labels[i] = 1;

    labels = tfk.utils.to_categorical(labels, num_classes=2)

    img_train_val, img_test, label_train_val, label_test = train_test_split(
        images, labels, random_state=seed, test_size=.10, stratify=labels
    )

    img_train, img_val, label_train, label_val = train_test_split(
        img_train_val, label_train_val, random_state=seed, test_size=.20, stratify=label_train_val
    )

    input_shape = img_train.shape[1:]
    output_shape = label_train.shape[1:]
    batch_size = 4
    epochs = 100

    callbacks = [
        tfk.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='auto'),
    ]

    convolution = tfk.applications.ConvNeXtBase(
        model_name="convnext_base",
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=(96, 96, 3),
        pooling="avg",
        classes=2,
        classifier_activation="softmax",
    )
    convolution.trainable=True

    preprocessing = tfk.Sequential([
        tfkl.RandomTranslation(0.3, 0.3, seed=seed),
        tfkl.RandomFlip(seed=seed),
        tfkl.RandomRotation(0.25, seed=seed),
        keras_cv.layers.AutoContrast(value_range=(0, 255)),
        keras_cv.layers.RandomGaussianBlur(1, factor=(0.2, 0.5), seed=seed)
    ])

    input_layer = tfkl.Input(shape=input_shape, name='Input')

    pre_layer = preprocessing(input_layer)

    mid_layer = convolution(pre_layer)

    output_layer = tfkl.Dense(units=2, activation='softmax', name='Output')(mid_layer)

    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='CNN')

    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(1e-5), metrics=['accuracy'])

    # in order to have everything normalized in the same way
    img_train = tfk.Sequential(
        tfkl.BatchNormalization()
    )(img_train)

    history = model.fit(
        x=img_train,
        y=label_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(img_val, label_val),
        callbacks=callbacks
    ).history

    plt.figure(figsize=(15, 5))
    plt.plot(history['loss'], alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_loss'], label='ConvNextBase', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')

    plt.figure(figsize=(15, 5))
    plt.plot(history['accuracy'], alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_accuracy'], label='ConvNextBase', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')

    plt.show()

    model.save('MyModel')


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

