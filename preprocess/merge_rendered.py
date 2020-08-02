import sys
import os
import numpy as np
import glob

merged_train_gt3D = None
merged_train_com = None
merged_train_data = None
merged_train_gt3DCrop = None

# train_gt3D, train_com, train_data

sMode = "msra"
# sMode = "unlabeled"
# sMode = "labeled"

if sMode == 'msra':
    # msra files:
    sOutputFile = "../data/msra_merged.npz"
    lInputFiles = glob.glob("../data/msra_*.npz")

elif sMode == 'unlabeled':
    # unlabeled files:
    sOutputFile = "../data/train_mano_merged.npz"
    lInputFiles = [
        "../data/train_mano.npz",
        "../data/train_mano_spread.npz",
        "../data/train_mano_close-open.npz",
        "../data/train_mano_shape.npz"
        ]

elif sMode == 'labeled':
    # labeled files:
    sOutputFile = "../data/train_mano_merged_new.npz"
    lInputFiles = [
        "../data/train_mano_pose_new.npz",
        "../data/train_mano_spread_new.npz",
        "../data/train_mano_close-open_new.npz",
        "../data/train_mano_shape_new.npz",
        "../data/train_mano_rot_new.npz"
        ]

for input_dataset in lInputFiles:
    dataset = np.load(input_dataset)

    train_data = dataset['train_data']
    
    if merged_train_data is None:
        merged_train_data = train_data
    else:
        merged_train_data = np.concatenate((merged_train_data, train_data))


    if 'train_gt3D' in dataset and 'train_com' in dataset:

        train_gt3D = dataset['train_gt3D']
        train_com = dataset['train_com']

        if merged_train_gt3D is None:
            merged_train_gt3D = train_gt3D
            merged_train_com = train_com
        else:
            merged_train_gt3D = np.concatenate((merged_train_gt3D, train_gt3D))
            merged_train_com = np.concatenate((merged_train_com, train_com))

        if 'train_gt3DCrop' in dataset:

            train_gt3DCrop = dataset['train_gt3DCrop']

            if merged_train_gt3DCrop is None:
                merged_train_gt3DCrop = train_gt3DCrop
            else:
                merged_train_gt3DCrop = np.concatenate((merged_train_gt3DCrop, train_gt3DCrop))

if merged_train_gt3DCrop is not None:
    np.savez(sOutputFile, train_gt3D=merged_train_gt3D, train_gt3DCrop=merged_train_gt3DCrop, train_com=merged_train_com, train_data=merged_train_data)
elif merged_train_gt3D is not None:
    np.savez(sOutputFile, train_gt3D=merged_train_gt3D, train_com=merged_train_com, train_data=merged_train_data)
else:
    np.savez(sOutputFile, train_data=merged_train_data)


print("Finished!")
