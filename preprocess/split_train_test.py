import sys
import os
import numpy as np

data_path = "../data/train_mano.npz"
data = np.load(data_path)

train_path = "../data/train_train_mano.npz"
test_path = "../data/test_train_mano.npz"
validation_path = "../data/validation_train_mano.npz"

#train
#validation
#test

test_size = 500
validation_size = test_size
train_size = data['train_data'].shape[0] - test_size - validation_size

indices = np.random.permutation(data['train_data'].shape[0])

train_idx = indices[:train_size]
validation_idx = indices[train_size: train_size + validation_size]
test_idx = indices[train_size + validation_size:]

test_set_data = data['train_data'][test_idx]
validation_set_data = data['train_data'][validation_idx]
train_set_data = data['train_data'][train_idx]

test_set_angles = data['train_angles'][test_idx]
validation_set_angles = data['train_angles'][validation_idx]
train_set_angles = data['train_angles'][test_idx]

#shape = data['train_data'].shape

np.savez(train_path, train_data=train_set_data, train_angles=train_set_angles)
np.savez(test_path, train_data=test_set_data, train_angles=test_set_angles)
np.savez(validation_path, train_data=validation_set_data, train_angles=validation_set_angles)

pass

#dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['train_data'])) #, data['train_angles']