import os
import sys
import random
import argparse
import numpy as np
from multiprocessing import freeze_support

# old: train_angles, train_data
# new: train_angles, train_gt3D, train_com, train_data

def main():
    
    # args:
    parser = argparse.ArgumentParser(description='HandPoseShapeVAE Preprocess Rendering')
    parser.add_argument('--input-dataset', type=str)
    parser.add_argument('--label-name', type=str)
    parser.add_argument('--output-dir', type=str, default="../data/preprocessed")
    # parser.add_argument('--train', type=float, default=0.0)
    parser.add_argument('--train', type=float, default=0.005)
    parser.add_argument('--validate', type=float, default=0.0)
    # parser.add_argument('--test', type=float, default=0.0)
    parser.add_argument('--test', type=float, default=0.4)
    parser.add_argument('--random', action='store_true', default=False)
    args = parser.parse_args()


    # load dataset:
    input_dataset = np.load(args.input_dataset)
    dataset_size = input_dataset['train_data'].shape[0]

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
    if args.random:
        indices = np.random.permutation(dataset_size)
    else:
        indices = np.arange(0, dataset_size)

    # save paths:
    train_path = os.path.join(args.output_dir, "train_{}.npz".format(args.label_name))
    test_path = os.path.join(args.output_dir, "test_{}.npz".format(args.label_name))
    validate_path = os.path.join(args.output_dir, "validate_{}.npz".format(args.label_name))

    # split:
    train_idx = indices[:train_size]
    validate_idx = indices[train_size: train_size + validate_size]
    test_idx = indices[train_size + validate_size:]

    test_set_train_data = input_dataset['train_data'][test_idx]
    validate_set_train_data = input_dataset['train_data'][validate_idx]
    train_set_train_data = input_dataset['train_data'][train_idx]

    # train_gt3D, train_com available:
    if 'train_gt3D' in input_dataset and 'train_com' in input_dataset:
        
        train_gt3D = input_dataset['train_gt3D'].reshape((-1, 21, 3))
        test_set_train_gt3D = train_gt3D[test_idx]
        validate_set_train_gt3D = train_gt3D[validate_idx]
        train_set_train_gt3D = train_gt3D[train_idx]

        train_com = input_dataset['train_com']
        test_set_train_com = train_com[test_idx]
        validate_set_train_com = train_com[validate_idx]
        train_set_train_com = train_com[train_idx]

        test_set_train_gt3DCrop = None
        validate_set_train_gt3DCrop = None
        train_set_train_gt3DCrop = None

        if 'train_gt3DCrop' in input_dataset:
            # MSRA:
            train_gt3DCrop = input_dataset['train_gt3DCrop'].reshape((-1, 21, 3))
            test_set_train_gt3DCrop = train_gt3DCrop[test_idx]
            validate_set_train_gt3DCrop = train_gt3DCrop[validate_idx]
            train_set_train_gt3DCrop = train_gt3DCrop[train_idx]
            
            # generous range:
            min_x = -150.0
            min_y = -150.0
            min_z = -150.0

            max_x = 200.0
            max_y = 200.0
            max_z = 200.0
        else:
            # Rendered:
            train_gt3DCrop = train_gt3D - train_com[:,np.newaxis,:]
            test_set_train_gt3DCrop = train_gt3DCrop[test_idx]
            validate_set_train_gt3DCrop = train_gt3DCrop[validate_idx]
            train_set_train_gt3DCrop = train_gt3DCrop[train_idx]

            # generous range:
            min_x = -110.0
            min_y = -110.0
            min_z = -110.0

            max_x = 110.0
            max_y = 110.0
            max_z = 110.0

        # specific range:
        # min_x = np.min( np.min(train_gt3DCrop, axis=1)[:, 0] )
        # min_y = np.min( np.min(train_gt3DCrop, axis=1)[:, 1] )
        # min_z = np.min( np.min(train_gt3DCrop, axis=1)[:, 2] )

        # max_x = np.max( np.max(train_gt3DCrop, axis=1)[:, 0] )
        # max_y = np.max( np.max(train_gt3DCrop, axis=1)[:, 1] )
        # max_z = np.max( np.max(train_gt3DCrop, axis=1)[:, 2] )

        train_gt3DCrop_norm = np.empty_like(train_gt3DCrop)

        # norm (-1, 1)
        train_gt3DCrop_norm[:,:,0] = 2 * (train_gt3DCrop[:,:,0] - min_x) / (max_x - min_x) - 1
        train_gt3DCrop_norm[:,:,1] = 2 * (train_gt3DCrop[:,:,1] - min_y) / (max_y - min_y) - 1
        train_gt3DCrop_norm[:,:,2] = 2 * (train_gt3DCrop[:,:,2] - min_z) / (max_z - min_z) - 1

        # # test:
        # min_x = np.min( np.min(train_gt3DCrop_norm, axis=1)[:, 0] )
        # min_y = np.min( np.min(train_gt3DCrop_norm, axis=1)[:, 1] )
        # min_z = np.min( np.min(train_gt3DCrop_norm, axis=1)[:, 2] )

        # max_x = np.max( np.max(train_gt3DCrop_norm, axis=1)[:, 0] )
        # max_y = np.max( np.max(train_gt3DCrop_norm, axis=1)[:, 1] )
        # max_z = np.max( np.max(train_gt3DCrop_norm, axis=1)[:, 2] )

        bounding_box = np.array(
            [
                [min_x, max_x],
                [min_y, max_y],
                [min_z, max_z]
            ]
        )

        test_set_train_gt3DCrop_norm = train_gt3DCrop_norm[test_idx]
        validate_set_train_gt3DCrop_norm = train_gt3DCrop_norm[validate_idx]
        train_set_train_gt3DCrop_norm = train_gt3DCrop_norm[train_idx]

        if (
            not os.path.isfile( train_path ) and
            not os.path.isfile( test_path ) and
            not os.path.isfile( validate_path )
        ):
            np.savez(train_path, train_gt3D=train_set_train_gt3D, train_gt3DCrop=train_set_train_gt3DCrop, train_gt3DCrop_norm=train_set_train_gt3DCrop_norm, train_com=train_set_train_com, train_data=train_set_train_data, bounding_box=bounding_box)
            np.savez(test_path, train_gt3D=test_set_train_gt3D, train_gt3DCrop=test_set_train_gt3DCrop, train_gt3DCrop_norm=test_set_train_gt3DCrop_norm, train_com=test_set_train_com, train_data=test_set_train_data, bounding_box=bounding_box)
            np.savez(validate_path, train_gt3D=validate_set_train_gt3D, train_gt3DCrop=validate_set_train_gt3DCrop, train_gt3DCrop_norm=validate_set_train_gt3DCrop_norm, train_com=validate_set_train_com, train_data=validate_set_train_data, bounding_box=bounding_box)

    else:
        if (
            not os.path.isfile( train_path ) and
            not os.path.isfile( test_path ) and
            not os.path.isfile( validate_path )
        ):
            np.savez(train_path, train_data=train_set_train_data)
            np.savez(test_path, train_data=test_set_train_data)
            np.savez(validate_path, train_data=validate_set_train_data)


if __name__ == '__main__':
    main()
    print("Finished!")
