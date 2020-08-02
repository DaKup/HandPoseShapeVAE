import sys
import os
import numpy as np
import scipy.misc

# data_path = "../data/preprocessed/rendered/train_msra_merged.npz"
data_path = "../data/train_mano_shape_new.npz"
data = np.load(data_path)

output_dir = "../data/plot_train_msra_merged/"


for idx, data in enumerate(data[data.files[1]]):

    array = np.array(data[0])
    filename = "{}.jpg".format(idx)
    scipy.misc.imsave(output_dir + filename, array)

