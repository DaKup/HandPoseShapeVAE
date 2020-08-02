from __future__ import print_function, division

import os
import sys
import gc
import random
import argparse
from multiprocessing import freeze_support

import torch
import glob
import pandas as pd
from skimage import io, transform
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np

from data.importers import MSRA15Importer


def main():
    
    # args:
    parser = argparse.ArgumentParser(description='HandPoseShapeVAE Preprocess Rendering')
    parser.add_argument('--input-dataset', type=str)
    parser.add_argument('--label-name', type=str)
    parser.add_argument('--output-dir', type=str, default="../data")
    parser.add_argument('--train', type=float, default=0.0)
    parser.add_argument('--validate', type=float, default=0.0)
    parser.add_argument('--test', type=float, default=0.0)
    parser.add_argument('--no-random', action='store_true', default=False)
    args = parser.parse_args()

    max_persons = -1
    normalize_input = True

    seqList = [x for x in os.listdir(args.input_dataset) if os.path.isdir(os.path.join(args.input_dataset, x))]
    if max_persons >= 0:
        seqList = seqList[:max_persons]

    importer = MSRA15Importer(args.input_dataset, useCache=False)

    test = []

    for seq in seqList:
        # test.append(np.load(os.path.join(args.output_dir, "msra_{}.npz".format(seq))))
        # print(test[-1][test[-1].files[1]].shape[0])
        # continue
        if normalize_input == True:
            tmp_sequence = importer.loadSequence(seq, docom=True)
            tmp_dpt = np.stack( [depth.dpt for depth in tmp_sequence.data], axis=0 )
            tmp_gt3Dcrop = np.stack( [depth.gt3Dcrop for depth in tmp_sequence.data], axis=0 )
            tmp_com = np.stack( [depth.com for depth in tmp_sequence.data], axis=0 )


            zero_idx = np.where(tmp_dpt == 0)
            tmp_dpt[zero_idx] = tmp_com[zero_idx[0],2] + (tmp_sequence.config['cube'][2] / 2.)
            tmp_dpt = (tmp_dpt.transpose() - tmp_com.transpose()[2]).transpose()
            tmp_dpt = tmp_dpt / (tmp_sequence.config['cube'][2] / 2.)
            tmp_dpt = np.expand_dims(tmp_dpt, axis=1) # unsqueeze

            # for depth_frame in tmp_sequence.data: # DepthFrame
            #     depth_frame.dpt[depth_frame.dpt == 0] = depth_frame.com[2] + (tmp_sequence.config['cube'][2] / 2.)
            #     depth_frame.dpt[:] = depth_frame.dpt[:] - depth_frame.com[2]
            #     depth_frame.dpt[:] = depth_frame.dpt[:] / (tmp_sequence.config['cube'][2] / 2.)

            train_path = os.path.join(args.output_dir, "msra_{}.npz".format(seq))
            np.savez(train_path, labels=tmp_gt3Dcrop, frames=tmp_dpt)

            
            # if dpt is None:
            #     dpt = tmp_dpt
            #     gt3Dcrop = tmp_gt3Dcrop
            # else:
            #     dpt = np.append(dpt, tmp_dpt, axis=0)
            #     gt3Dcrop = np.append(gt3Dcrop, tmp_gt3Dcrop, axis=0)
    print("Done!")
    return

    # load dataset:
    input_dataset = np.load(args.input_dataset)
    dataset_size = input_dataset[input_dataset.files[1]].shape[0]

    # train test validate sizes:
    if args.train > 0.0 and args.test > 0.0 and args.validate > 0.0:
        assert(args.train + args.test + args.validate == 1.0)
        train_ratio = args.train
        test_ratio = args.test
        validate_ratio = args.validate
    elif args.train > 0.0 and args.test > 0.0:
        assert(args.train + args.test < 1.0)
        train_ratio = args.train
        test_ratio = args.test
        validate_ratio = 1.0 - args.train - args.test
    elif args.train > 0.0 and args.validate > 0.0:
        assert(args.train + args.validate < 1.0)
        train_ratio = args.train
        validate_ratio = args.validate
        test_ratio = 1.0 - args.train - args.validate
    elif args.test > 0.0 and args.validate > 0.0:
        assert(args.test + args.validate < 1.0)
        test_ratio = args.test
        validate_ratio = args.validate
        train_ratio = 1.0 - args.test - args.validate
    else:
        train_ratio = 0.6
        test_ratio = 0.2
        validate_ratio = 0.2

    assert(train_ratio + test_ratio + validate_ratio == 1.0)
    train_size = int(train_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    validate_size = dataset_size - train_size - test_size
    assert(train_size > 0 and test_size > 0 and validate_size > 0)
    

    # randomize:
    if not args.no_random:
        indices = np.random.permutation(dataset_size)
    else:
        indices = np.arange(0, dataset_size)


    # split:
    train_idx = indices[:train_size]
    validate_idx = indices[train_size: train_size + validate_size]
    test_idx = indices[train_size + validate_size:]

    test_set_frames = input_dataset[input_dataset.files[1]][test_idx]
    validate_set_frames = input_dataset[input_dataset.files[1]][validate_idx]
    train_set_frames = input_dataset[input_dataset.files[1]][train_idx]

    test_set_labels = input_dataset[input_dataset.files[0]][test_idx]
    validate_set_labels = input_dataset[input_dataset.files[0]][validate_idx]
    train_set_labels = input_dataset[input_dataset.files[0]][train_idx]


    # save:
    train_path = os.path.join(args.output_dir, "train_{}.npz".format(args.label_name))
    test_path = os.path.join(args.output_dir, "test_{}.npz".format(args.label_name))
    validate_path = os.path.join(args.output_dir, "validate_{}.npz".format(args.label_name))

    np.savez(train_path, labels=train_set_labels, frames=train_set_frames)
    np.savez(test_path, labels=test_set_labels, frames=test_set_frames)
    np.savez(validate_path, labels=validate_set_labels, frames=validate_set_frames)


if __name__ == '__main__':
    main()
    print("Finished!")
