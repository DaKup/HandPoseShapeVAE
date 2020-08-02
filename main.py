import sys, os
import argparse
import glob
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np

from models.models import VAE

from process.solver import Solver
from process.loss import LossFunction, LossBVAE, LossBVAE2, LossTCVAE, LossJoints
from log.logger import Logger

from data.dataloader import load_frame, create_dataloader

# from process.experiments import evaluate, traverse_latents, save_graph, plot_3d_model, run, getDisentanglementMetric
from process.experiments import (
    traverse_latents,
    save_graph,
    plot_3d_model,
    run,
    getDisentanglementMetric,
    svr_joints,
    train_joints,
    save_plots,
    compare_joint_training,
    plot_datasets
)

from torchsummary import summary

# todo: tc-vae loss
# todo: tc-vae metric


def main():

    # arguments
    args = parse_args()
    print_config(args)

    if args.save_plots:

        save_plots()
        return

    if args.compare_joint_training:
        compare_joint_training()
        return

    if args.plot_datasets:

        plot_datasets()
        return
    
    # init solver
    solver = init_solver(args)

    # print model statistics
    # solver.model.to("cuda")
    # summary(solver.model, (1, 128, 128))

    # load checkpoint:
    if args.load_net_path != '':
        solver.load_checkpoint(args.load_net_path)
    elif args.load_epoch >= 0:
        auto_load(solver, os.path.join(args.model_dir, args.basename), args.model, args.num_latents, args.load_epoch)
    elif args.auto_load:
        auto_load(solver, os.path.join(args.model_dir, args.basename), args.model, args.num_latents)

    # train:
    if args.train:
        solver.train(args.train_dir, args.validation_dir, args.batch_size, args.epochs, args.checkpoint_interval, args.model_dir, args.overwrite, max_persons=args.max_persons, shuffle=args.shuffle)
        if args.save_net_path != "":
            if args.overwrite or not os.path.isfile(args.save_net_path):
                solver.save_checkpoint(args.save_net_path)
        else:
            filename = "{}_{}.pt".format(os.path.join(args.model_dir, solver.logger.basename), solver.logger.epoch)
            if args.overwrite or not os.path.isfile(filename):
                solver.save_checkpoint(filename)
    
    # test:
    if args.test:
        solver.test(args.test_dir, args.test_batch_size)
    
    # experiments:
    if args.experiments:
        dataset_index = 15
        input_frame = get_test_frame(args.input_frame, args.test_dir, dataset_index)

        if args.run:
            run(solver, input_frame)

        if args.plot3d:
            plot_3d_model(solver, input_frame)

        if args.traverse_latents:
            # traverse_latents(solver, input_frame, latents=torch.arange(-3, 3.1, 1/6.), log_label="traverse_{}".format(dataset_index))

            dataloader = create_dataloader(args.test_dir, args.batch_size, normalize_input=True, max_persons=-1, shuffle=True)
            for i in range(25):
                input_frame = get_test_frame(args.input_frame, args.test_dir, i)
                traverse_latents(solver, dataloader, input_frame, latents=torch.arange(-3, 3.1, 1/6.), log_label="traverse_{}".format(i))

        if args.plot_graph:
            # save_graph(solver, input_frame)
            save_graph(solver)

        if args.train_joints:
            train_joints(solver, args.train_dir, args.validation_dir)

    if args.evaluate:
        # evaluate(solver)
        # trainloader = create_dataloader(args.train_dir, args.batch_size, normalize_input=True, max_persons=-1, shuffle=True)
        # getDisentanglementMetric(solver, trainloader)

        svr_joints(solver, args.train_dir, args.validation_dir)
        
    return # end of application


def get_test_frame(filename: Path=None, dataset: Path=None, dataset_index=0):
    
    frame = None
    if filename:
        # load frame from file

        print("using image {} for experiments".format(filename))

        if not os.path.isfile(filename):
            print("Cannot load {}".format(filename))
            return frame

        # if msra:
        default_cubes = {
            'P0': (200, 200, 200),
            'P1': (200, 200, 200),
            'P2': (200, 200, 200),
            'P3': (180, 180, 180),
            'P4': (180, 180, 180),
            'P5': (180, 180, 180),
            'P6': (170, 170, 170),
            'P7': (160, 160, 160),
            'P8': (150, 150, 150)}

        # todo: allow passing optional args.input_cube instead of default cube
        cube = None
        for p, c in default_cubes.items():
            if p in filename:
                cube = c
                break

        dpt, M, com = load_frame(filename, docom=True, cube=cube)
        frame = torch.from_numpy(dpt)

    elif dataset:
        # load frame from dataset
        print("using image from dataset {} for experiments".format(dataset))

        if os.path.isfile(dataset):
            data = np.load(dataset)
            frame = torch.from_numpy(data['train_data'])[dataset_index].squeeze(0)

        else:
            dataloader = create_dataloader(dataset, batch_size=1, normalize_input=True, shuffle=False)
            for batch_idx, batch_input in enumerate(dataloader):

                if batch_idx == dataset_index:
                    batch_input = batch_input[0]#.to(self.device) # batch_input[1] contains the target data (if this wasn't an autoencoder...)
                    if batch_input.dim() == 4:
                        batch_input = batch_input.squeeze(1)
                    
                    frame = batch_input[0]
                    break

    else:
        # generate frame from random data
        print("using random image for experiments")
        frame = torch.randn(128, 128, dtype=torch.float)

    return frame


def auto_load(solver: Solver, basename: Path, model: str, num_latents: int, epoch: int = None):
    
    filenamebase = "{}_{}_{}_".format(basename, model, num_latents)

    if epoch != None:
        solver.load_checkpoint("{}_{}.pt".format(filenamebase, epoch))
        return

    max_epoch = None
    filename = None
    for tmp in glob.glob("{}*.pt".format(filenamebase)):
        try:
            epoch = int(tmp[len(filenamebase):-len(".pt")])
            if os.path.isfile(tmp):
                if max_epoch == None:
                    max_epoch = epoch
                    filename = tmp
                elif epoch > max_epoch:
                    max_epoch = epoch
                    filename = tmp
        except:
            pass

    if filename != None:
        solver.load_checkpoint(filename)
    else:
        print("Auto-Load {}*.pt skipped".format(filenamebase))


def print_config(args):

    sMode = "mode: "
    if args.train:
        sMode += "train; "
    if args.test:
        sMode += "test; "
    if args.experiments:
        sMode += "experiments; "
    if args.evaluate:
        sMode += "evaluate; "
    if sMode == "mode: ":
        sMode += "nothing selected"
    print(sMode)

    print("device: {}".format(args.device))
    if args.seed > 0:
        print("manual seed: {}".format(args.seed))

    print("model: {}".format(args.model))
    print("z-dim: {}".format(args.num_latents))
    
    if args.model == 'bvae':
        print("beta: {}".format(args.beta))
    elif args.model == 'bvae2':
        print("gamma: {}; max-capacity: {}; warmup-iterations: {}".format(args.gamma, args.max_capacity, args.capacity_increments))
    elif args.model == 'tcvae':
        pass


def init_solver(args):

    # seed: random, manual:
    if args.seed > 0:
        torch.manual_seed(args.seed)
        
    torch.manual_seed(torch.initial_seed())

    # logger:
    if args.no_logging:
        logger = None
    else:
        logger = Logger(args.model, log_interval=args.log_interval, basename="{}_{}_{}".format(args.basename, args.model.lower(), args.num_latents))
        logger.add_text("CommandLine", ' '.join(sys.argv))
        # total_commandline = sys.argv[0]
        for arg in vars(args):
            logger.add_text("arg/" + arg, str(getattr(args, arg)))
            #print(arg, getattr(args, arg))
            #total_commandline += "--" + arg.replace('_', '-') + " " + str(getattr(args, arg))

    output_dist = None
    if args.output_dist:
        output_dist = args.output_dist

    milestones = []

    # model:
    if args.model == 'vae':
        model = VAE(args.num_latents, output_dist=output_dist)
        loss_function = LossBVAE(logger, beta=1.0)
    elif args.model == 'bvae':
        model = VAE(args.num_latents, output_dist=output_dist)
        loss_function = LossBVAE(logger, beta=args.beta)
    elif args.model == 'bvae2':
        model = VAE(args.num_latents, output_dist=output_dist)
        loss_function = LossBVAE2(logger, max_capacity=args.max_capacity, gamma=args.gamma)
    elif args.model == 'tcvae':
        model = VAE(args.num_latents, output_dist=output_dist)
        loss_function = LossTCVAE(logger, beta=args.beta)
        # milestones = [3500, 5000]
        # milestones = [99999999]
    elif args.model == 'cnn':
        model = CNN()
        loss_function = LossJoints(logger)
    else:
        sys.exit("Unknown model: {}".format(args.model))

    loss_function.output_dist = model.output_dist

    # torch.nn.init.xavier_uniform.apply(model)

    # optimizer and learning rate scheduler:
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100, 200, 300, 400, 500], gamma=0.1)
    scheduler = None
    if len(milestones) > 0:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # solver:
    return Solver(model, loss_function, optimizer, scheduler, args.device, logger)


def parse_args():
    
    print(*sys.argv, sep=' ')
    parser = argparse.ArgumentParser(description='HandPoseShapeVAE')

    # general:
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--seed', type=int, default=-1, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--basename', type=str, default='HandPoseShapeVAE')
    parser.add_argument('--model-dir', type=str, default='../models')

    # logging:
    parser.add_argument('--log-interval', type=int, default=80, metavar='N', help='how many batches to wait before logging training status')
    # traverse latents:
    parser.add_argument('--no-logging', action='store_true', default=False)
    
    # loading:
    parser.add_argument('--load-net-path', type=str, default='')
    parser.add_argument('--load_epoch', type=int, default=-1)
    parser.add_argument('--auto-load', action='store_true', default=False)
    
    # saving:
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--save-net-path', type=str, default='')
    parser.add_argument('--auto-save', action='store_true', default=False)
    parser.add_argument('--checkpoint-interval', type=int, default=10)

    parser.add_argument('--save-examples', action='store_true', default=False)
    
    # architecture:
    parser.add_argument('--model', type=str, default='bvae2', choices=["vae", "bvae", "bvae2", "tcvae", "cnn"])#, "ae", "dcign", "sdvae"])
    parser.add_argument('--num-latents', type=int, default=10) # 10, 120, 200, 32 for CelebA
    #parser.add_argument('--output-params', type=int, default=1) # 1 (=no distr.), 2 (=normal distr.)
    parser.add_argument('--output-dist', type=str, default='', choices=["normal", "bernoulli", "none", "fake_normal"])#, "normal", "bernoulli", "" => None

    # train:
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--train-dir', type=str)
    # parser.add_argument('--train-dir', type=str, default='../data/preprocessed/msra/train')
    # parser.add_argument('--train-dir', type=str, default='../data/preprocessed/rendered/train_merged.npz')
    # parser.add_argument('--train-dir', type=str, default='../data/preprocessed/rendered/train_angles.npz')
    # parser.add_argument('--train-dir', type=str, default='../data/preprocessed/rendered/train_spread.npz')
    # parser.add_argument('--train-dir', type=str, default='../data/preprocessed/rendered/train_open.npz')
    # parser.add_argument('--train-dir', type=str, default='../data/preprocessed/rendered/train_shape.npz')
    parser.add_argument('--max-persons', type=int, default=-1, help='Restrict training to a subset for faster testing (default: <=0  <==> train all)')
    parser.add_argument('--batch-size', type=int, default=600, metavar='N', help='input batch size for training (default: 80)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--learning-rate', type=float, default=5e-4, metavar='LR', help='learning rate for Adam optimizer (default: 5e-4)')

    # validate:
    parser.add_argument('--validation-dir', type=str)
    # parser.add_argument('--validation-dir', type=str, default='../data/preprocessed/msra/validate')
    # parser.add_argument('--validate-dir', type=str, default='../data/preprocessed/rendered/validate_merged.npz')
    # parser.add_argument('--validate-dir', type=str, default='../data/preprocessed/rendered/validate_angles.npz')
    # parser.add_argument('--validate-dir', type=str, default='../data/preprocessed/rendered/validate_spread.npz')
    # parser.add_argument('--validate-dir', type=str, default='../data/preprocessed/rendered/validate_open.npz')
    # parser.add_argument('--validate-dir', type=str, default='../data/preprocessed/rendered/validate_shape.npz')

    # test:
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test-dir', type=str)
    # parser.add_argument('--test-dir', type=str, default='../data/preprocessed/msra/test')
    # parser.add_argument('--test-dir', type=str, default='../data/preprocessed/msra/test_pose')
    # parser.add_argument('--test-dir', type=str, default='../data/preprocessed/msra/test_shape')
    # parser.add_argument('--test-dir', type=str, default='../data/preprocessed/rendered/test_merged.npz')
    # parser.add_argument('--test-dir', type=str, default='../data/preprocessed/rendered/test_angles.npz')
    # parser.add_argument('--test-dir', type=str, default='../data/preprocessed/rendered/test_spread.npz')
    # parser.add_argument('--test-dir', type=str, default='../data/preprocessed/rendered/test_open.npz')
    # parser.add_argument('--test-dir', type=str, default='../data/preprocessed/rendered/test_shape.npz')
    parser.add_argument('--test-batch-size', type=int, default=650, metavar='N', help='input batch size for testing (default: 100)')

    # experiments:
    parser.add_argument('--experiments', action='store_true', default=False, help='run experiments')
    parser.add_argument('--input-frame', type=str, default='')

    # train joints:
    parser.add_argument('--train-joints', action='store_true', default=False)

    # evaluation:
    parser.add_argument('--evaluate', action='store_true', default=False)

    # run:
    parser.add_argument('--run', action='store_true', default=False, help='process a single frame')
    parser.add_argument('--output-frame', type=str, default='../data/MSRA15/example/output.bin')
    parser.add_argument('--output-image', type=str, default='../data/MSRA15/example/output.jpg')
    parser.add_argument('--output-joints', type=str, default='../data/MSRA15/example/output.tsv')
    
    # traverse:
    parser.add_argument('--traverse-latents', action='store_true', default=False)

    # plot 3d model:
    parser.add_argument('--plot3d', action='store_true', default=False)

    # plot nn graph:
    parser.add_argument('--plot-graph', action='store_true', default=False)

    # plots etc for report:

    # save plots:
    parser.add_argument('--save-plots', action='store_true', default=False)

    # compare joint learning:
    parser.add_argument('--compare-joint-training', action='store_true', default=False)

    # plot datasets:
    parser.add_argument('--plot-datasets', action='store_true', default=False)

    # architecture-specific arguments:

    # dcign (deprecated): Kulkarni, Tejas D.; Whitney, Will; Kohli, Pushmeet; Tenenbaum, Joshua B. (2015): Deep Convolutional Inverse Graphics Network.
    # parser.add_argument('--disentangle', action='store_true', default=False)
    # parser.add_argument('--pose-dict', type=str, default='pose_dict_small.pth')
    
    # bvae: β-VAE. LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK (2017).
    parser.add_argument('--beta', type=float, default=15.0)

    # bvae2: Burgess, Christopher P.; Higgins, Irina; Pal, Arka; Matthey, Loic; Watters, Nick; Desjardins, Guillaume; Lerchner, Alexander (2018): Understanding disentangling in $β$-VAE.
    parser.add_argument('--max-capacity', type=float, default=250.0) # 25nats for dSprites, 50nats for CelebA /linear increment over 100.000 iterations
    parser.add_argument('--gamma', type=float, default=1000.0) # chosen to be large enough to ensure the actual KL is always close to the target KL
    parser.add_argument('--capacity-increments', type=int, default=1500000.0)

    # tcvae: Chen, Tian Qi; Li, Xuechen; Grosse, Roger; Duvenaud, David (2018): Isolating Sources of Disentanglement in Variational Autoencoders.
    # todo: implement
    
    args = parser.parse_args()

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    args.model = args.model.lower()

    if not args.auto_save:
        args.checkpoint_interval = -1

    return args


if __name__ == '__main__':
    main()
    print("Finished!")
