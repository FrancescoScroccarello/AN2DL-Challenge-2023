import numpy as np
from tqdm import trange

training_data = np.load("Dataset/training_data.npy")
new_data=[]
for i in trange(training_data.shape[0]):
    new_data.append(training_data[i][2776-209:])

new_data = np.array(new_data)
print(new_data.shape)
np.save("Dataset/cut_data.npy",new_data)