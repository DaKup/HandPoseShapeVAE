from __future__ import print_function, division
import os
import gc
import torch
import glob
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data.importers import DepthImporter, MSRA15Importer


class MSRA15ImporterWrapper:
    
    # loading all data at once will probably fail or time-out.
    # idea: preprocessing => create pkl cache for subSeq (will take very long, but is only done once)
    def __init__(self, importer: MSRA15Importer, normalize_input=True, max_persons=-1):


        # else:
        #     imgD = numpy.asarray(imgSeq.data[i].dpt.copy(), 'float32')
        #     imgD[imgD == 0] = imgSeq.data[i].com[2] + (imgSeq.config['cube'][2] / 2.)
        #     imgD -= imgSeq.data[i].com[2]
        #     imgD /= (imgSeq.config['cube'][2] / 2.)

        #     imgStack[i] = imgD
        #     labelStack[i] = numpy.asarray(imgSeq.data[i].gt3Dcrop, dtype='float32') / (imgSeq.config['cube'][2] / 2.)
        
        basepath = importer.basepath
        self._seqList = [x for x in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, x))]

        # TODO: debug: only load 2 persons for testing
        if max_persons >= 0:
            self._seqList = self._seqList[:max_persons]

        # self._data = dict()
        # for seq in self._seqList:
        #     if normalize_input == True:
        #         self._data[seq] = importer.loadSequence(seq, docom=True)
        #         for depth_frame in self._data[seq].data: # DepthFrame
        #             depth_frame.dpt[depth_frame.dpt == 0] = depth_frame.com[2] + (self._data[seq].config['cube'][2] / 2.)
        #             depth_frame.dpt[:] = depth_frame.dpt[:] - depth_frame.com[2]
        #             depth_frame.dpt[:] = depth_frame.dpt[:] / (self._data[seq].config['cube'][2] / 2.)
        #     else:
        #         self._data[seq] = importer.loadSequence(seq, docom=False)
        #     gc.collect()

        self.dpt = None
        self.gt3Dcrop = None

        for seq in self._seqList:
            if normalize_input == True:
                tmp_sequence = importer.loadSequence(seq, docom=True)
                tmp_dpt = np.stack( [depth.dpt for depth in tmp_sequence.data], axis=0 )
                tmp_gt3Dcrop = np.stack( [depth.gt3Dcrop for depth in tmp_sequence.data], axis=0 )
                tmp_com = np.stack( [depth.com for depth in tmp_sequence.data], axis=0 )


                zero_idx = np.where(tmp_dpt == 0)
                tmp_dpt[zero_idx] = tmp_com[zero_idx[0],2] + (tmp_sequence.config['cube'][2] / 2.)
                tmp_dpt = (tmp_dpt.transpose() - tmp_com.transpose()[2]).transpose()
                tmp_dpt = tmp_dpt / (tmp_sequence.config['cube'][2] / 2.)

                # for depth_frame in tmp_sequence.data: # DepthFrame
                #     depth_frame.dpt[depth_frame.dpt == 0] = depth_frame.com[2] + (tmp_sequence.config['cube'][2] / 2.)
                #     depth_frame.dpt[:] = depth_frame.dpt[:] - depth_frame.com[2]
                #     depth_frame.dpt[:] = depth_frame.dpt[:] / (tmp_sequence.config['cube'][2] / 2.)
                
                if self.dpt is None:
                    self.dpt = tmp_dpt
                    self.gt3Dcrop = tmp_gt3Dcrop
                else:
                    self.dpt = np.append(self.dpt, tmp_dpt, axis=0)
                    self.gt3Dcrop = np.append(self.gt3Dcrop, tmp_gt3Dcrop, axis=0)
            else:
                tmp_sequence = importer.loadSequence(seq, docom=False)
            gc.collect()

class MSRA15Dataset(Dataset):
    
    def __init__(self, importer: MSRA15ImporterWrapper, train_mode: str = "mixed", pose_dict_path: str = None, batch_size: int = None):
        self._importer = importer
        self._batch_size = batch_size
        self._train_mode = train_mode
        
        self._minibatch = None

        if train_mode == "mixed":
            #raise Exception("Not implemented => Use a 'simple' dataset and pytorch dataloader with batch_size > 0")
            self._batch_size = 1 # for mixed train mode: set batch_size in pytorch dataloader
            self._minibatch = []
            #self._minibatch = [all frames]
            for shape_name, shape_data in importer._data.items():
                for frame in shape_data.data:
                    self._minibatch.append(frame.dpt)
        elif train_mode == "pose": # => fixed shape, variable pose => aka the default file structure

            # max minibatch_size ~= 19 * 500 ==> all frames of one shape
            # within any minibatch the same shape must be used
            # if batch_size >= 0 or None: _minibatch = full sequence
            self._minibatch = []
            # for each sequence/shape/person: add a list of minibatches 
            
            # the important principal is that tmp_batch (which itself is a list of frames)
            # is guaranteed to contain only frames of the same person
            for shape_name, shape_data in importer._data.items():
                # tmp = (up to) batch_size frames of that sequence
                tmp_batch = []
                # for frame in seq:
                # for batch_idx in range(num_of_minibatches_in_this_sequence):
                # alternatively start a new tmp_batch every batch_size'th frame:
                for frame in shape_data.data:
                    if batch_size != None and batch_size > 0 and len(tmp_batch) >= batch_size:
                        #self._minibatch.append(np.array(tmp_batch))
                        self._minibatch.append(tmp_batch)
                        tmp_batch = []
                    tmp_batch.append(frame.dpt)

                if tmp_batch != []:
                    #self._minibatch.append(np.array( tmp_batch )) # add a list of mini_batches to self._minibatch
                    self._minibatch.append(tmp_batch) # add a list of mini_batches to self._minibatch
        elif train_mode == "shape":
            assert(pose_dict_path != None)
            pose_dict = torch.load(pose_dict_path)

            # max minibatch_size ~= 8 ==> all frames of one pose (we only have ~8 persons who did this one pose)
            # within any minibatch the same pose must be used
            # if batch_size >= 0 or None: _minibatch = full sequence
            self._minibatch = []
            # for each pose: add a list of minibatches (we have about 19*500 poses, each of about 8 frames)
            
            #tmp_batch = []
            # tmp_batch = np.empty(shape=(cur_batch_size, 128, 128), dtype=float)
            # the important principal is that tmp_batch (which itself is a list of frames)
            # is guaranteed to contain only frames of the same pose
                        
            for subSeq, _ in pose_dict.items():
                for rotation in pose_dict[subSeq]:
                    
                    tmp_batch = []
                    # as a result we will get A LOT(!) of very small minibatches here
                    for shape, _ in pose_dict[subSeq][rotation].items():
                        if shape not in self._importer._data:
                            continue # pose_dict's shape might be from test set

                        filename = pose_dict[subSeq][rotation][shape]['filename']
                        if filename == None:
                            continue # the gesture is missing for this person

                        idx = int(filename[:6])
                        assert( os.path.basename( self._importer._data[shape].data[idx].fileName ) == filename )

                        if batch_size != None and batch_size > 0 and len(tmp_batch) >= batch_size:
                            #self._minibatch.append(np.array(tmp_batch))
                            self._minibatch.append(tmp_batch)
                            tmp_batch = []

                        frame = self._importer._data[shape].data[idx]
                        tmp_batch.append(frame.dpt)

                    #self._minibatch.extend(tmp_batch) # add a list of mini_batches to self._minibatch
                    if tmp_batch != []:
                        #self._minibatch.append(np.array( tmp_batch )) # add a list of mini_batches to self._minibatch
                        self._minibatch.append(tmp_batch) # add a list of mini_batches to self._minibatch

            # for shape_name, shape_data in importer._data.items():
            #     # tmp = (up to) batch_size frames of that sequence
            #     tmp_batch = []
            #     # for frame in seq:
            #     # for batch_idx in range(num_of_minibatches_in_this_sequence):
            #     # alternatively start a new tmp_batch every batch_size'th frame:
            #     for frame in range(shape_data.data):
            #         if batch_size != None and batch_size > 0 and len(tmp_batch) >= batch_size:
            #             self._minibatch.append(tmp_batch)
            #             tmp_batch = []                      
            #         tmp_batch.append(frame.dpt)
            #     self._minibatch.extend(tmp_batch) # add a list of mini_batches to self._minibatch
        else:
            raise Exception("unknown train_mode")
        # load all sequences once and then for each minibatch put the frames together


    def __len__(self):
        return len(self._minibatch)
    
    def __getitem__(self, idx):
        # idx := minibatch_index

        #test = self._minibatch[idx]
        #return test
        #return self._minibatch[idx]
        # TODO: make a method which expects minibatch to contain either indices or shallow copies and generates numpy arrays on the fly (to save memory):
        #return np.array(self._minibatch[idx])
        return [np.array(self._minibatch[idx])]
        # Labels = None # to comply with pytorch's way of input + ground_truth
        # return [np.array(self._minibatch[idx]), Labels]
        #return [torch.from_numpy(self._minibatch[idx]).unsqueeze(1)]
        #return self._sequence.data[idx].dpt

