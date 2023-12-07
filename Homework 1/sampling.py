import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import layers as tfkl
import cv2
import numpy as np
from tqdm import trange
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler

import warnings
import logging

seed = 42
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
np.random.seed(seed)
random.seed(seed)
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


def print_balance(labels):
    new_labels = np.zeros(len(labels), dtype=int)

    for k in range(len(labels)):
        new_labels[k] = 0 if labels[k] == 'healthy' else 1

    healthy_num, unhealthy_num = np.bincount(new_labels)

    print(
        f'Dataset balance: \n{" " * 20}Healthy samples - {healthy_num}, \n{" " * 20}Unhealthy samples - {unhealthy_num}')


def undersampling(images, labels):
    nsamples, nx, ny, nz = images.shape
    d2_train_dataset = images.reshape((nsamples, nx * ny * nz))
    undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    images_resampled, labels_resampled = undersampler.fit_resample(d2_train_dataset, labels)
    images_resampled = images_resampled.reshape((images_resampled.shape[0], nx, ny, nz))

    # Save samples deleted during the UnderSampling process
    deleted_images = np.zeros((len(images) - len(images_resampled), nx, ny, nz), dtype=int)
    deleted_labels = np.zeros((len(labels) - len(labels_resampled)), dtype=str)
    sampled_indices = undersampler.sample_indices_
    deleted_index = []
    for i in range(images.shape[0]):
        if i not in sampled_indices:
            deleted_index.append(i)
    for i in range(len(deleted_index)):
        deleted_images[i] = images[deleted_index[i]]
        deleted_labels[i] = labels[deleted_index[i]]
    np.savez("UndersampledDataset/deleted_samples.npz", data=deleted_images, labels=deleted_labels)
    print(f'Deleted samples shapes: \n{" "*20}Images - {deleted_images.shape}, \n{" "*20}Labels - {deleted_labels.shape}')

    permutation = np.random.permutation(images_resampled.shape[0])
    print(f'Undersampled dataset shapes: \n{" "*20}Images - {images_resampled.shape}, \n{" "*20}Labels - {labels_resampled.shape}')
    print_balance(labels_resampled)
    return images_resampled[permutation], labels_resampled[permutation]


def smote(images, labels):
    nsamples, nx, ny, nz = images.shape
    d2_train_dataset = images.reshape((nsamples, nx * ny * nz))
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    images_resampled, labels_resampled = smote.fit_resample(d2_train_dataset, labels)
    images_resampled = images_resampled.reshape((images_resampled.shape[0], nx, ny, nz))
    print_balance(labels_resampled)
    print(f'Oversampled (SMOTE) dataset shapes: \n{" " * 20}Images - {images_resampled.shape}, \n{" " * 20}Labels - {labels_resampled.shape}')
    permutation = np.random.permutation(images.shape[0])
    return images_resampled[permutation], labels_resampled[permutation]


def random_oversampling(images, labels):
    nsamples, nx, ny, nz = images.shape
    d2_train_dataset = images.reshape((nsamples, nx * ny * nz))
    random_oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    images_resampled, labels_resampled = random_oversampler.fit_resample(d2_train_dataset, labels)
    images_resampled = images_resampled.reshape((images_resampled.shape[0], nx, ny, nz))
    print(f'Oversampled (Random) dataset shapes: \n{" " * 20}Images - {images_resampled.shape}, \n{" " * 20}Labels - {labels_resampled.shape}')
    print_balance(labels_resampled)
    permutation = np.random.permutation(images.shape[0])
    return images_resampled[permutation], labels_resampled[permutation]


def oversampling(images, labels):

    unhealthy = []

    for i in trange(images.shape[0]):
        if labels[i] == 'unhealthy':
            unhealthy.append(images[i])

    preprocessing = tf.keras.Sequential([
        tfkl.RandomRotation(0.3),
        tfkl.RandomFlip(),
    ])

    length = len(images) - 2 * len(unhealthy) + random.randint(-100, 100)
    to_prepare = random.sample(unhealthy, length)
    prepared = preprocessing(to_prepare)
    new_labels = []
    for i in trange(length):
        new_labels.append('unhealthy')
        filename = f"{i}_augmented.png"
        file_path = "AugmentedSet/" + filename
        r = prepared[i, :, :, 0]
        g = prepared[i, :, :, 1]
        b = prepared[i, :, :, 2]
        rgb_image = np.dstack((r, g, b))
        cv2.imwrite(file_path, rgb_image)

    images = np.concatenate((images, np.array(prepared)), axis=0)
    labels = np.concatenate((labels, np.array(new_labels)), axis=0)
    permutation = np.random.permutation(images.shape[0])
    print(f'Oversampled (Custom) dataset shapes: \n{" " * 20}Images - {images.shape}, \n{" " * 20}Labels - {labels.shape}')
    print_balance(labels)
    return images[permutation], labels[permutation]



data = np.load("Dataset/clean_dataset.npz", allow_pickle=True)
images = data['data']
labels = data['labels']

new_labels = np.zeros(len(labels), dtype=int)

for k in range(len(labels)):
    new_labels[k] = 0 if labels[k] == 'healthy' else 1

healthy_num, unhealthy_num = np.bincount(new_labels)

print(f'Dataset shapes: \n{" "*20}Images - {images.shape}, \n{" "*20}Labels - {labels.shape}')

print_balance(labels)

smp_images, smp_labels = oversampling(images, labels)
np.savez("OversampledDataset/oversampled_dataset", data=smp_images, labels=smp_labels)
#
smp_images, smp_labels = undersampling(images, labels)
np.savez("UndersampledDataset/undersampled_dataset", data=smp_images, labels=smp_labels)
#
smp_images, smp_labels = random_oversampling(images, labels)
np.savez("OversampledDataset/RandomOvr_dataset", data=smp_images, labels=smp_labels)

smp_images, smp_labels = smote(images, labels)
np.savez("OversampledDataset/SMOTE_dataset", data=smp_images, labels=smp_labels)




