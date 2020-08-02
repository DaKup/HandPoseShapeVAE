from __future__ import print_function, division

import os
import sys
sys.path.append(".")
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
    parser = argparse.ArgumentParser(description='HandPoseShapeVAE Import MSRA')
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

    # 'dpt', 'gtorig', 'gtcrop', 'T', 'gt3Dorig', 'gt3Dcrop', 'com', 'fileName', 'subSeqName', 'side', 'extraData'

    for seq in seqList:
        if normalize_input == True:
            tmp_sequence = importer.loadSequence(seq, docom=True)
            tmp_dpt = np.stack( [depth.dpt for depth in tmp_sequence.data], axis=0 )
            tmp_gt3Dcrop = np.stack( [depth.gt3Dcrop for depth in tmp_sequence.data], axis=0 )
            tmp_gt3D = np.stack( [depth.gt3Dorig for depth in tmp_sequence.data], axis=0 )
            tmp_com = np.stack( [depth.com for depth in tmp_sequence.data], axis=0 )

            zero_idx = np.where(tmp_dpt == 0)
            tmp_dpt[zero_idx] = tmp_com[zero_idx[0],2] + (tmp_sequence.config['cube'][2] / 2.)
            tmp_dpt = (tmp_dpt.transpose() - tmp_com.transpose()[2]).transpose()
            tmp_dpt = tmp_dpt / (tmp_sequence.config['cube'][2] / 2.)
            tmp_dpt = np.expand_dims(tmp_dpt, axis=1) # unsqueeze

            # tmp_gtorig
            # tmp_gtcrop
            # tmp_T
            # tmp_fileName
            # tmp_subSeqName
            # tmp_side
            # tmp_extraData

            train_path = os.path.join(args.output_dir, "msra_{}.npz".format(seq))
            np.savez(
                train_path,
                # general:
                train_gt3D=tmp_gt3D, # gt3Dorig
                train_gt3DCrop=tmp_gt3Dcrop, # gt3Dcrop
                train_com=tmp_com, # com
                train_data=tmp_dpt # dpt
                # msra specific:
                # ,train_gtorig=tmp_gtorig,
                # train_gtcrop=tmp_gtcrop,
                # train_T=tmp_T,
                # train_fileName=tmp_fileName,
                # train_subSeqName=tmp_subSeqName,
                # train_side=tmp_side,
                # train_extraData=tmp_extraData
            )

            # train_angles, train_gt3D, train_com, train_data


if __name__ == '__main__':
    main()
    print("Finished!")
