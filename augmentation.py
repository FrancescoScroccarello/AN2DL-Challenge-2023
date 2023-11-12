import random
import tensorflow as tf
from keras import layers as tfkl
import cv2
import numpy as np
from tqdm import trange


data = np.load("Dataset/clean_dataset.npz", allow_pickle=True)
images = data['data']
labels = data['labels']
unhealthy = []
for i in trange(images.shape[0]):
    if labels[i] == 'unhealthy':
        unhealthy.append(images[i])

preprocessing = tf.keras.Sequential([
    tfkl.RandomFlip('horizontal_and_vertical'),
    tfkl.RandomTranslation(height_factor=(-0.1,0.1), width_factor=(-0.1,0.1),fill_mode='reflect', interpolation='nearest'),
    tfkl.RandomRotation(0.2,fill_mode='reflect', interpolation='nearest')
])

len = len(images) - 2*len(unhealthy) + random.randint(-100,100)
to_prepare=random.sample(unhealthy,len)
prepared=preprocessing(to_prepare)
new_labels=[]
for i in trange(len):
    new_labels.append('unhealthy')
    filename = f"{i}_augmented.png"
    file_path = "AugmentedSet/" + filename
    r = prepared[i, :, :, 0]
    g = prepared[i, :, :, 1]
    b = prepared[i, :, :, 2]
    rgb_image = np.dstack((r, g, b))
    cv2.imwrite(file_path, rgb_image)

images = np.concatenate((images,np.array(prepared)),axis=0)
labels = np.concatenate((labels,np.array(new_labels)),axis=0 )
permutation = np.random.permutation(images.shape[0])
new_images = images[permutation]
new_labels = labels[permutation]
np.savez("AugmentedSet/augmented_dataset",data=new_images,labels=new_labels)