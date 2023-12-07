import cv2
import numpy as np
from tqdm import trange

data = np.load("Dataset/public_data.npz", allow_pickle=True)
images = data['data']
labels = data['labels']

clean_set = []
clean_labels = []
# there are some faulty elements in the dataset
shrek = images[880]
troll = images[898]

output_directory = 'Dataset'
for i in trange(images.shape[0]):
    # this check is to have a clean dataset
    if not np.array_equal(images[i], shrek) and not np.array_equal(images[i], troll):
        clean_set.append(images[i])
        clean_labels.append(labels[i])
        filename = f"{i}_{labels[i]}.png"
        file_path = output_directory + "/" + filename
        r = images[i, :, :, 0]
        g = images[i, :, :, 1]
        b = images[i, :, :, 2]
        rgb_image = cv2.merge((r, g, b))
        cv2.imwrite(file_path, rgb_image)

clean_set = np.array(clean_set)
clean_labels = np.array(clean_labels)
print(clean_set.shape[0])
np.savez("CleanDS/clean_dataset.npz", data=clean_set, labels=clean_labels)