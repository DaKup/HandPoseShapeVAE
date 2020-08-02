import sys
import os
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from data.importers import MSRA15Importer
from data.dataset import MSRA15Dataset, MSRA15ImporterWrapper

from util.handdetector import HandDetector


def create_dataloader(data_dir: Path, batch_size: int, normalize_input=True, max_persons=-1, train_mode="mixed", pose_dict_path=None, shuffle=False):
    
    if os.path.isfile(data_dir):
        data = np.load(data_dir)

        if 'train_gt3DCrop_norm' in data:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['train_data']), torch.from_numpy(data['train_gt3DCrop_norm']))
        else:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['train_data']))

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            #collate_fn=default_collate,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None
        )
        return dataloader

    importer = MSRA15ImporterWrapper(MSRA15Importer(data_dir), normalize_input=normalize_input, max_persons=max_persons)

    if pose_dict_path != None:
        dataset = MSRA15Dataset(importer, train_mode=train_mode, pose_dict_path=pose_dict_path, batch_size=batch_size)
        batch_size = 1
    else:
        dataset = MSRA15Dataset(importer)
        
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        #collate_fn=default_collate,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None
    )
    return dataloader


def load_frame(filename: Path, com=None, size=(250, 250, 250), dsize=(128, 128), docom=False, cube=None):

    # No ImporterWrapper here, because we don't want to load all sequences, we just need a way to load frames from single files:
    importer = MSRA15Importer(basepath=None)

    dpt = importer.loadDepthMap(filename)
    hand_detector = HandDetector(dpt, importer.fx, importer.fy, refineNet=importer.refineNet, importer=importer)
    if not hand_detector.checkImage(1.):
        sys.exit("No hand detected")

    #try:
    cropped_hand_depth, joint_transf_mat, com = hand_detector.cropArea3D(com=None, size=size, dsize=dsize, docom=docom) # size=config['cube']
    # except UserWarning:
    #     #sys.exit("Skipping file {}, no hand detected".format(filename))
    #     print("Skipping file {], no hand detected".format(filename))
    #     return None

    if cube == None:
        cube = [] # Min/Max
    else:
        # normalize input [-1, 1]
        cropped_hand_depth[cropped_hand_depth == 0] = com[2] + (cube[2] / 2.)
        cropped_hand_depth = cropped_hand_depth - com[2]
        cropped_hand_depth = cropped_hand_depth / (cube[2] / 2.)

    return cropped_hand_depth, joint_transf_mat, com
    # input = torch.from_numpy(cropped_hand_depth)
    # batch_input = input.unsqueeze(0) # add 1 for the batch dimension
    # batch_input = batch_input.to(args.device)

def save_wavefront(filename: Path, dpt):

    with open(filename, "w") as obj:
        
        for xyz in dpt:
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
            obj.write("v {} {} {}\n".format(x, y, z))
