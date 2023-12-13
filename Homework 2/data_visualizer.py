import sys
import numpy as np
from tqdm import trange


np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)

training_data = np.load("Dataset/training_data.npy")
valid_periods = np.load("Dataset/valid_periods.npy")
categories = np.load("Dataset/categories.npy")

f = open('DataVisualization.txt', 'w')
sys.stdout = f

for i in trange(training_data.shape[0]):
    print(training_data[i])
    print(valid_periods[i])
    print(categories[i])
    print(" ")