import matplotlib.pyplot as plt
import numpy as np

data=np.load('UndersampledDataset/undersampled_dataset.npz', allow_pickle=True)
labels=data['labels']

plt.hist(labels)
plt.show()


