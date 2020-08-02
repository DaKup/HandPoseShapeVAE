# from torch.utils.data import Dataset, DataLoader
import sys
import os
import glob
import csv
from data.importers import DepthImporter, MSRA15Importer

from log.progress import progress

#from torchvision.utils import save_image

import scipy.misc

import numpy as np
import shutil
import copy

def generateMatchingPoseList(basepath: str):
    
    seqList = [x for x in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, x))]
    subSeqList = [x for x in os.listdir(os.path.join(basepath, seqList[0])) if os.path.isdir(os.path.join(basepath, seqList[0], x))]
    filenameList = [x for x in os.listdir(os.path.join(basepath, seqList[0], subSeqList[0])) if os.path.isfile(os.path.join(basepath, seqList[0], subSeqList[0], x)) and x.lower().endswith(".bin")]
    
    pose_dict = dict()

    total = len(seqList) * len(subSeqList)

    for subSeq in subSeqList:
        pose_dict[subSeq] = dict()
        for i in range(len(filenameList)):
            pose_dict[subSeq][i] = dict()
            for seq in seqList:
                pose_dict[subSeq][i][seq] = dict() # filename, distance
                pose_dict[subSeq][i][seq]['filename'] = None
                pose_dict[subSeq][i][seq]['distance'] = None
                #pose_dict[subSeq][filename][seq]['filenameList'] = dict()

    importer = MSRA15Importer(basepath)
    count = 0
    for subSeq in subSeqList:
        #pose_dict[subSeq] = dict()
        seq1 = seqList[0]
        sequence1 = importer.loadSequence(seq1, [subSeq])
        for seq2 in seqList[1:]:
            progress(count, total, status='')
            count += 1
            sequence2 = importer.loadSequence( seq2, [subSeq] )
            #for idx1 in range(len(filenameList)):
            for idx1 in range(len(sequence1.data)):
                pose_dict[subSeq][idx1][seq1]['filename'] = os.path.basename(sequence1.data[idx1].fileName)
                pose_dict[subSeq][idx1][seq1]['distance'] = -1.0
                pose_dict[subSeq][idx1][seq1]['distanceList'] = [('', -1.0)] * 500
                pose_dict[subSeq][idx1][seq2]['distanceList'] = [('', -1.0)] * 500

                joints1 = sequence1.data[idx1].gt3Dcrop
                best_idx2 = None
                best_dist = None
                # for idx2 in range(len(filenameList)):
                for idx2 in range(len(sequence2.data)):
                    joints2 = sequence2.data[idx2].gt3Dcrop

                    #dist = np.sqrt(np.sum((joints1 - joints2)**2))
                    dist = np.linalg.norm(joints1 - joints2, axis=1)
                    #max_dist = np.mean(dist)
                    max_dist = np.max(dist)

                    pose_dict[subSeq][idx1][seq2]['distanceList'][idx2] = (os.path.basename(sequence2.data[idx2].fileName), max_dist)

                    if best_dist == None or max_dist < best_dist:
                        best_dist = max_dist
                        best_idx2 = idx2

                pose_dict[subSeq][idx1][seq2]['filename'] = os.path.basename(sequence2.data[best_idx2].fileName)
                pose_dict[subSeq][idx1][seq2]['distance'] = best_dist

    progress(total, total, status='')
    return pose_dict


def saveTrainTestDataset(basepath_input: str, basepath_output: str, test_seq: str, test_subSeq: str):
    
    seqList = [f.path for f in os.scandir(basepath_input) if f.is_dir()]
    for seq in seqList:
        subSeqList = [f.path for f in os.scandir(seq) if f.is_dir()]
        for subSeq in subSeqList:
            source = subSeq
            if os.path.basename(seq) == test_seq and os.path.basename(subSeq) == test_subSeq:
                target = "test"
            elif os.path.basename(seq) == test_seq and os.path.basename(subSeq) != test_subSeq:
                target = "test_shape"
            elif os.path.basename(seq) != test_seq and os.path.basename(subSeq) == test_subSeq:
                target = "test_pose"
            else:
                target = "train"

            target = os.path.join(basepath_output, target, os.path.basename(seq), os.path.basename(subSeq))
            os.makedirs(os.path.join(basepath_output, target, os.path.basename(seq)), exist_ok=True)

            shutil.copytree(source, target)

    
def saveReorderedDataset(pose_dict, basepath_input: str, basepath_output: str, test_seq: str, test_subSeq: str):
    
    os.mkdir(os.path.join(basepath_output, "train"))
    os.mkdir(os.path.join(basepath_output, "test"))
    os.mkdir(os.path.join(basepath_output, "test_pose"))
    os.mkdir(os.path.join(basepath_output, "test_shape"))

    total = len(pose_dict)
    count = 0
    
    for subSeq, _ in pose_dict.items():
        progress(count, total, status='')
        count += 1
        for idx, _ in pose_dict[subSeq].items():
            for seq, _ in pose_dict[subSeq][idx].items():
                filename = pose_dict[subSeq][idx][seq]['filename']
                if filename == None:
                    continue
                src_file = os.path.join(basepath_input, seq, subSeq, filename)
                if subSeq != test_subSeq and seq != test_seq:
                    subdir = "train"
                elif subSeq == test_subSeq and seq == test_seq:
                    subdir = "test"
                elif subSeq != test_subSeq and seq == test_seq:
                    subdir = "test_shape"
                else:
                    subdir = "test_pose"

                dest_file = os.path.join(basepath_output, subdir, seq, subSeq, filename)
                if not os.path.isdir( os.path.join( basepath_output, subdir, seq, subSeq ) ):
                    if not os.path.isdir( os.path.join( basepath_output, subdir, seq ) ):
                        os.mkdir( os.path.join( basepath_output, subdir, seq ) )
                    os.mkdir( os.path.join( basepath_output, subdir, seq, subSeq ) )                    

                shutil.copy2(src_file, dest_file)
                shutil.copy2(src_file[:-3] + "jpg", dest_file[:-3] + "jpg")
                
    progress(total, total, status='')

def saveGestureSet(pose_dict_small, basepath: str, subSeq, idx: int, output_dir: str):
    
    importer = MSRA15Importer(basepath)
    for seq, tmp in pose_dict_small[subSeq][idx].items():
        sequence = importer.loadSequence(seq, [subSeq])
        depth_frame = sequence.data[idx].dpt
        depth_frame8 = (depth_frame * 255 / np.max(depth_frame)).astype('uint8')
        src_file = os.path.join(basepath, seq, subSeq, tmp['filename'][:-3] + "jpg")
        dest_file = os.path.join(output_dir, seq + ".jpg")
        #save_image(depth_frame8, dest_file[:-3] + "png")
        scipy.misc.imsave(dest_file[:-3] + "png", depth_frame8)
        shutil.copy2(src_file, dest_file)

# bug: don't save from pose_dict, because many frames are missing due to overlapping closest similarities
# fix: copy directories from source_dir, basically ignoring the pose_dict
def saveSortedPoseMatches(pose_dict, basepath: str, seq1: str, seq2: str, subSeq: str, idx: int, output_dir: str):

    importer = MSRA15Importer(basepath)
    original_frame = os.path.join(basepath, seq1, subSeq, pose_dict[subSeq][idx][seq1]['filename'][:-3] + "jpg")
    shutil.copy2(original_frame, os.path.join(output_dir, "original.jpg"))
    sequence = importer.loadSequence(seq1, [subSeq])
    depth_frame = sequence.data[idx].dpt
    depth_frame8 = (depth_frame * 255 / np.max(depth_frame)).astype('uint8')
    scipy.misc.imsave(os.path.join(output_dir, "original.png"), depth_frame8)

    distanceList = pose_dict[subSeq][idx][seq2]['distanceList']
    #distanceList.sort(key=lambda tup: tup[1])  # sorts in place
    sortedList = sorted(distanceList, key=lambda tup: tup[1])
    counter = 0
    sequence = importer.loadSequence(seq2, [subSeq])
    for (filename, dist) in sortedList:
        counter += 1
        if dist < 0.0:
            continue
        src_file = os.path.join(basepath, seq2, subSeq, os.path.basename(filename)[:-3] + "jpg")
        dest_file = os.path.join(output_dir, str(counter).zfill(5) + ".jpg")
        shutil.copy2(src_file, dest_file)
        idx2 = int(os.path.basename(filename)[:-10])
        depth_frame = sequence.data[idx2].dpt
        depth_frame8 = (depth_frame * 255 / np.max(depth_frame)).astype('uint8')
        scipy.misc.imsave(dest_file[:-3] + "png", depth_frame8)


def getPoseDictSmall(pose_dict):
    
    pose_dict_small = dict()
    for subSeq, _ in pose_dict.items():
        pose_dict_small[subSeq] = dict()
        for idx, _ in pose_dict[subSeq].items():
            pose_dict_small[subSeq][idx] = dict()
            for seq, _ in pose_dict[subSeq][idx].items():
                pose_dict_small[subSeq][idx][seq] = dict()
                pose_dict_small[subSeq][idx][seq]['filename'] = pose_dict[subSeq][idx][seq]['filename']
                pose_dict_small[subSeq][idx][seq]['distance'] = pose_dict[subSeq][idx][seq]['distance']
        
    return pose_dict_small
