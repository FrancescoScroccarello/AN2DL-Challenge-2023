import numpy as np
from tqdm import trange

training_data = np.load("Dataset/training_data.npy")
valid_periods = np.load("Dataset/valid_periods.npy")
categories = np.load("Dataset/categories.npy")


count = 0
count_less = 0
samples = training_data.shape[0]
for i in trange(valid_periods.shape[0]):
    count += valid_periods[i][0]
    if valid_periods[i][0] < 2500:
        count_less += 1

print(f"Mean starting time: {count/samples}\nPercentage less than the mean: {count_less/samples*100}")