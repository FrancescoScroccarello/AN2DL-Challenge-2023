import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

data = np.load("AgumentedSet/agumented_dataset.npz")
images = data['data']
labels = data['labels']

# Fix randomness and hide warnings
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

import random
random.seed(seed)

tf.random.set_seed(seed)

img_train_val, img_test, label_train_val, label_test = train_test_split(
    images, labels, random_state=seed, test_size=0.10, stratify=labels
)

img_train, img_val, label_train, label_val = train_test_split(
    img_train_val, label_train_val, random_state=seed, test_size=0.25, stratify=label_train_val
)

out_translation = {'unhealthy' : 0,
                   'healthy' : 1}

for i in range(0,len(label_train)):
    if label_train[i] == 'healthy':
        label_train[i]=0
    else:
        label_train[i]=1


for j in range(0,len(label_val)):
    if label_val[j] == 'healthy':
        label_val[j] = 0
    else:
        label_val[j] = 1

for k in range(0,len(label_test)):
    if label_test[k] == 'healthy':
        label_test[k] = 0
    else:
        label_test[k] = 1

input_shape = img_train.shape[1:]
output_shape = label_train.shape[1:]
batch_size = 32
epochs = 1000

label_train = tfk.utils.to_categorical(label_train, num_classes=2)
label_val = tfk.utils.to_categorical(label_val, num_classes=2)
label_test = tfk.utils.to_categorical(label_test, num_classes=2)

callbacks = [
    tfk.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True, mode='auto'),
]

input_layer = tfkl.Input(shape=input_shape, name='Input')

inception = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(96,96,3),
    pooling='avg',
    classes=1000,
    classifier_activation="softmax",
)
inception.trainable = False
x = inception(input_layer)

output_layer = tfkl.Dense(units=2, activation='softmax',name='Output')(x)

model = tfk.Model(inputs=input_layer, outputs=output_layer, name='CNN')

model.compile(loss=tfk.losses.BinaryCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(
    x = img_train,
    y = label_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = (img_val, label_val),
    callbacks = callbacks
).history

plt.figure(figsize=(15,5))
plt.plot(history['loss'], alpha=.3, color='#ff7f0e', linestyle='--')
plt.plot(history['val_loss'], label='Vanilla CNN', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Categorical Crossentropy')
plt.grid(alpha=.3)

plt.figure(figsize=(15,5))
plt.plot(history['accuracy'], alpha=.3, color='#ff7f0e', linestyle='--')
plt.plot(history['val_accuracy'], label='Vanilla CNN', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Accuracy')
plt.grid(alpha=.3)

plt.show()

model.save('first_model')

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
