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
    tfkl.RandomBrightness(0.2, value_range=(0,255)),
    tfkl.RandomZoom(0.1),
])

to_prepare=random.sample(unhealthy,1000)
prepared=preprocessing(to_prepare)
new_labels=[]
for i in trange(1000):
    new_labels.append('unhealthy')
    filename = f"{i}_agumented.png"
    file_path = "AgumentedSet/" + filename
    r = prepared[i, :, :, 0]
    g = prepared[i, :, :, 1]
    b = prepared[i, :, :, 2]
    rgb_image = np.dstack((r, g, b))
    cv2.imwrite(file_path, rgb_image)

images=np.concatenate((images,np.array(prepared)),axis=0)
labels=np.concatenate((labels,np.array(new_labels)),axis=0 )
permutation=np.random.permutation(images.shape[0])
new_images=images[permutation]
new_labels=labels[permutation]
np.savez("AgumentedSet/agumented_dataset",data=new_images,labels=new_labels)
