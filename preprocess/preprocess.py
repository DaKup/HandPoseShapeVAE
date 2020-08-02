import os
import sys
import argparse
import random
from multiprocessing import freeze_support

import torch
from torchvision.utils import save_image

from preprocess.find_pose import generateMatchingPoseList, saveGestureSet, saveSortedPoseMatches, getPoseDictSmall, saveReorderedDataset, saveTrainTestDataset

# train:
# sys.argv.append('--basedir')
# sys.argv.append('../data/MSRA15_Small')

# sys.argv.append('--basedir_output')
# sys.argv.append('../data/MSRA15_Small_Preprocessed')

# sys.argv.append('--pose_dict')
# sys.argv.append('small_pose_dict.pth')

# sys.argv.append('--pose_dict_small')
# sys.argv.append('small_pose_dict_small.pth')


def main():
    
    # args:
    parser = argparse.ArgumentParser(description='HandPoseShapeAE preprocess')
    parser.add_argument('--basedir', type=str, default='../data/MSRA15')
    parser.add_argument('--basedir_output', type=str, default='../data/MSRA15_Preprocessed')
    parser.add_argument('--pose_dict', type=str, default='pose_dict.pth')
    parser.add_argument('--pose_dict_small', type=str, default='pose_dict_small.pth')
    parser.add_argument('--test_shape', type=str, default='P0')
    parser.add_argument('--test_pose', type=str, default='1')
    args = parser.parse_args()

    pose_dict = None
    pose_dict_small = None

    # pose_dict & pose_dict_small:
    if args.pose_dict != '':
        if os.path.isfile(args.pose_dict):
            print("loading pose_dict '{0}'...".format(args.pose_dict))
            pose_dict = torch.load(args.pose_dict)
            print("done!")
        else:
            print("creating pose_dict '{0}'...".format(args.pose_dict))
            torch.save({}, args.pose_dict) # test if writing is possible before processing
            os.remove(args.pose_dict)
            pose_dict = generateMatchingPoseList(args.basedir)
            print("saving pose_dict '{0}'...".format(args.pose_dict))
            torch.save(pose_dict, args.pose_dict)
            print("done!")

    if args.pose_dict_small != '':
        if os.path.isfile(args.pose_dict_small):
            if pose_dict != None:
                pose_dict_small = pose_dict
            else:
                print("loading pose_dict_small '{0}'...".format(args.pose_dict_small))
                pose_dict_small = torch.load(args.pose_dict_small)
                print("done!")
        elif args.pose_dict != '':
            print("creating pose_dict_small '{0}'...".format(args.pose_dict_small))
            pose_dict_small = getPoseDictSmall(pose_dict)
            print("saving pose_dict_small '{0}'...".format(args.pose_dict_small))
            torch.save(pose_dict_small, args.pose_dict_small)
            print("done!")

    # split train, test, test_pose, test_shape
    if not os.path.exists(args.basedir_output):
        print("creating train, test, test_pose, test_shape datasets in '{0}'...".format(args.basedir_output))
        os.mkdir(args.basedir_output)
        #saveReorderedDataset(pose_dict_small, args.basedir, args.basedir_output, args.test_shape, args.test_pose)
        saveTrainTestDataset(args.basedir, args.basedir_output, args.test_shape, args.test_pose)
        print("done!")

    # save gesture example
    if not os.path.exists(os.path.join(args.basedir_output, "example")):
        os.mkdir(os.path.join(args.basedir_output, "example"))

    if args.pose_dict != '':
        if not os.path.exists(os.path.join(args.basedir_output, "example", "sorted")):
            os.mkdir(os.path.join(args.basedir_output, "example", "sorted"))
        rand_subSeq = random.choice(list(pose_dict))
        rand_idx = random.choice(list(pose_dict[rand_subSeq]))
        # rand_seq1 = random.choice(list(pose_dict[rand_subSeq][rand_idx]))
        rand_seq1 = 'P0'
        rand_seq2 = random.choice(list(pose_dict[rand_subSeq][rand_idx]))
        counter = 0
        while rand_seq2 == rand_seq1:
            rand_seq2 = random.choice(list(pose_dict[rand_subSeq][rand_idx]))
            counter += 1
            if counter >= 15:
                break
        if counter < 15:
            print("saving example sorted gestures by similarity '{0}':{1} of '{2}' and '{3}'...".format(rand_subSeq, rand_idx, rand_seq1, rand_seq2))
            with open(os.path.join(args.basedir_output, "example", "sorted", "example.txt"), "w") as oFile:
                oFile.write("sorted gestures by similarity '{0}':{1} of '{2}' and '{3}'...".format(rand_subSeq, rand_idx, rand_seq1, rand_seq2))
            saveSortedPoseMatches(pose_dict, args.basedir, rand_seq1, rand_seq2, rand_subSeq, rand_idx, os.path.join(args.basedir_output, "example", "sorted"))
            print("done!")

    rand_subSeq = random.choice(list(pose_dict_small))
    rand_idx = random.choice(list(pose_dict_small[rand_subSeq]))
    print("saving example gesture '{0}':{1}...".format(rand_subSeq, rand_idx))
    with open(os.path.join(args.basedir_output, "example", "example.txt"), "w") as oFile:
        oFile.write("gesture '{0}':{1}...".format(rand_subSeq, rand_idx))
    saveGestureSet(pose_dict_small, args.basedir, rand_subSeq, rand_idx, os.path.join(args.basedir_output, "example"))
    print("done!")

    print( "preprocessing completed!" )


if __name__ == '__main__':
    main()
