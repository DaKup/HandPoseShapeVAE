import sys
import os
import numpy as np
import glob

## deprecated: use merge_rendered

merged_frames = None
merged_labels = None

for input_dataset in glob.glob("../data/msra_*.npz"):
    print(input_dataset)
    dataset = np.load(input_dataset)
    frames = dataset[dataset.files[1]]
    labels = dataset[dataset.files[0]]
    
    if merged_frames is None:
        merged_frames = frames
        merged_labels = labels
    else:
        merged_frames = np.concatenate((merged_frames, frames))
        merged_labels = np.concatenate((merged_labels, labels))

np.savez("../data/msra_merged.npz", labels=merged_labels, frames=merged_frames)

print("Finished!")
