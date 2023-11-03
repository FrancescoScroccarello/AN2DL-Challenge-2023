import matplotlib.pyplot as plt
import numpy as np

data=np.load('Dataset/public_data.npz',allow_pickle=True)
labels=data['labels']

plt.hist(labels)
plt.show()


