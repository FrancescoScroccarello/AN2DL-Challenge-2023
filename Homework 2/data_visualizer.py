import numpy as np
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter

training_data = np.load("Dataset/training_data.npy")
valid_periods = np.load("Dataset/valid_periods.npy")
categories = np.load("Dataset/categories.npy")

labels = []

label_counts = Counter(categories)
for label, count in label_counts.items():
    print(f'{label}: {count}')

target_ratios = {'A': 1.2, 'B': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 5.0}

desired_samples = {}
for label, count in label_counts.items():
    desired_samples[label] = int(count * (target_ratios[label] / 1.0))

smote = SMOTE(sampling_strategy=desired_samples, random_state=42)
training_data1, categories = smote.fit_resample(training_data, categories)

plt.hist(categories)

for i in range(training_data.shape[0]):
    if valid_periods[i][1]-valid_periods[i][0]>=209:
        labels.append(categories[i])


data = np.array(labels)
plt.hist(data)
plt.show()
