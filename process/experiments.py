import os
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from process.solver import Solver
import torch
from torch import Tensor
import torchvision
import torch.optim as optim

from data.dataloader import load_frame
from postprocess.postprocess import postprocess_frame

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tensorboardX import SummaryWriter
from data.dataloader import create_dataloader, load_frame, save_wavefront
from postprocess.postprocess import postprocess_frame
from data.importers import MSRA15Importer
import torchvision

from log.logger import Logger

from process.loss import LossJoints, LossFunction, LossBVAE, LossBVAE2, LossTCVAE

from models.models import FullyConnected, JointModel

# from process.loss import logsumexp, normal_log_density


from array2gif import write_gif

import scipy.misc

from sklearn.svm import LinearSVR


from models.models import VAE

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


from tqdm import tnrange, tqdm_notebook, trange
from sklearn import metrics

from scipy.spatial import minkowski_distance
from sklearn.cluster import spectral_clustering
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from sklearn.preprocessing import normalize

import hiddenlayer as hl

# requires supervised latents
# # def mutual_info_metric_shapes(vae, shapes_dataset):
# #     dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

# #     N = len(dataset_loader.dataset)  # number of data samples
# #     K = vae.z_dim                    # number of latent variables
# #     nparams = vae.q_dist.nparams
# #     vae.eval()

# #     print('Computing q(z|x) distributions.')
# #     qz_params = torch.Tensor(N, K, nparams)

# #     n = 0
# #     for xs in dataset_loader:
# #         batch_size = xs.size(0)
# #         xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
# #         qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
# #         n += batch_size

# #     qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda())
# #     qz_samples = vae.q_dist.sample(params=qz_params)

# #     print('Estimating marginal entropies.')
# #     # marginal entropies
# #     marginal_entropies = estimate_entropies(
# #         qz_samples.view(N, K).transpose(0, 1),
# #         qz_params.view(N, K, nparams),
# #         vae.q_dist)

# #     marginal_entropies = marginal_entropies.cpu()
# #     cond_entropies = torch.zeros(4, K)

# #     print('Estimating conditional entropies for scale.')
# #     for i in range(6):
# #         qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
# #         qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

# #         cond_entropies_i = estimate_entropies(
# #             qz_samples_scale.view(N // 6, K).transpose(0, 1),
# #             qz_params_scale.view(N // 6, K, nparams),
# #             vae.q_dist)

# #         cond_entropies[0] += cond_entropies_i.cpu() / 6

# #     print('Estimating conditional entropies for orientation.')
# #     for i in range(40):
# #         qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
# #         qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

# #         cond_entropies_i = estimate_entropies(
# #             qz_samples_scale.view(N // 40, K).transpose(0, 1),
# #             qz_params_scale.view(N // 40, K, nparams),
# #             vae.q_dist)

# #         cond_entropies[1] += cond_entropies_i.cpu() / 40

# #     print('Estimating conditional entropies for pos x.')
# #     for i in range(32):
# #         qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
# #         qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

# #         cond_entropies_i = estimate_entropies(
# #             qz_samples_scale.view(N // 32, K).transpose(0, 1),
# #             qz_params_scale.view(N // 32, K, nparams),
# #             vae.q_dist)

# #         cond_entropies[2] += cond_entropies_i.cpu() / 32

# #     print('Estimating conditional entropies for pox y.')
# #     for i in range(32):
# #         qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
# #         qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

# #         cond_entropies_i = estimate_entropies(
# #             qz_samples_scale.view(N // 32, K).transpose(0, 1),
# #             qz_params_scale.view(N // 32, K, nparams),
# #             vae.q_dist)

# #         cond_entropies[3] += cond_entropies_i.cpu() / 32

# #     metric = compute_metric_shapes(marginal_entropies, cond_entropies)
# #     return metric, marginal_entropies, cond_entropies


# def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None, device="cuda"):
#     """Computes the term:
#         E_{p(x)} E_{q(z|x)} [-log q(z)]
#     and
#         E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
#     where q(z) = 1/N sum_n=1^N q(z|x_n).
#     Assumes samples are from q(z|x) for *all* x in the dataset.
#     Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

#     Computes numerically stable NLL:
#         - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

#     Inputs:
#     -------
#         qz_samples (K, N) Variable
#         qz_params  (N, K, nparams) Variable
#         weights (N) Variable
#     """

#     # Only take a sample subset of the samples
#     if weights is None:
#         # qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
#         qz_samples = qz_samples.index_select(1, torch.randperm(qz_samples.size(1))[:n_samples].to(device))
#     else:
#         sample_inds = torch.multinomial(weights, n_samples, replacement=True)
#         qz_samples = qz_samples.index_select(1, sample_inds)

#     K, S = qz_samples.size()
#     N, _, nparams = qz_params.size()
#     # assert(nparams == q_dist.nparams)
#     assert(nparams == 2)
#     assert(K == qz_params.size(1))

#     if weights is None:
#         weights = -math.log(N)
#     else:
#         weights = torch.log(weights.view(N, 1, 1) / weights.sum())

#     entropies = torch.zeros(K).to(device)

#     pbar = tqdm(total=S)
#     k = 0
#     while k < S:
#         batch_size = min(10, S - k)
#         sample = qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size]
#         mu = mu.view(N, K, 1).expand(N, K, S)[:, :, k:k + batch_size]
#         logsigma = logsigma.view(N, K, 1).expand(N, K, S)[:, :, k:k + batch_size]
#         logqz_i = normal_log_density( sample, mu, logsigma )
#         # logqz_i = q_dist.log_density(
#         #     qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
#         #     qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
#         k += batch_size

#         # computes - log q(z_i) summed over minibatch
#         # entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
#         entropies += - logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
#         pbar.update(batch_size)
#     pbar.close()

#     entropies /= S

#     return entropies


# def evaluate(solver: Solver, dataloader: DataLoader, device="cuda"):

#     N = len(dataloader.dataset)  # number of data samples
#     # K = vae.z_dim                    # number of latent variables
#     K = solver.model.num_latents
#     # nparams = vae.q_dist.nparams
#     nparams = solver.model.num_params
#     # vae.eval()
#     solver.model.eval()

#     print('Computing q(z|x) distributions.')
#     qz_params = torch.Tensor(N, K, nparams)

#     n = 0
#     for xs in dataloader:
#         batch_size = xs.size(0)
#         # xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
#         xs = xs.view(batch_size, 1, 128, 128).to(device)
#         # qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
#         (mu, logsigma) = solver.model.encode(xs).view(batch_size, K, nparams)
#         qz_params[n:n + batch_size] = (mu, logsigma)
#         n += batch_size

#     # qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda())
#     qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams).to(device)
#     # qz_samples = vae.q_dist.sample(params=qz_params)
#     qz_samples = solver.model.reparameterize( mu, logsigma )

#     print('Estimating marginal entropies.')
#     # marginal entropies
#     marginal_entropies = estimate_entropies(
#         qz_samples.view(N, K).transpose(0, 1),
#         qz_params.view(N, K, nparams),
#         vae.q_dist)

#     marginal_entropies = marginal_entropies.cpu()
#     cond_entropies = torch.zeros(4, K)

#     print('Estimating conditional entropies for scale.')
#     for i in range(6):
#         qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
#         qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

#         cond_entropies_i = estimate_entropies(
#             qz_samples_scale.view(N // 6, K).transpose(0, 1),
#             qz_params_scale.view(N // 6, K, nparams),
#             vae.q_dist)

#         cond_entropies[0] += cond_entropies_i.cpu() / 6

#     print('Estimating conditional entropies for orientation.')
#     for i in range(40):
#         qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
#         qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

#         cond_entropies_i = estimate_entropies(
#             qz_samples_scale.view(N // 40, K).transpose(0, 1),
#             qz_params_scale.view(N // 40, K, nparams),
#             vae.q_dist)

#         cond_entropies[1] += cond_entropies_i.cpu() / 40

#     print('Estimating conditional entropies for pos x.')
#     for i in range(32):
#         qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
#         qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

#         cond_entropies_i = estimate_entropies(
#             qz_samples_scale.view(N // 32, K).transpose(0, 1),
#             qz_params_scale.view(N // 32, K, nparams),
#             vae.q_dist)

#         cond_entropies[2] += cond_entropies_i.cpu() / 32

#     print('Estimating conditional entropies for pox y.')
#     for i in range(32):
#         qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
#         qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

#         cond_entropies_i = estimate_entropies(
#             qz_samples_scale.view(N // 32, K).transpose(0, 1),
#             qz_params_scale.view(N // 32, K, nparams),
#             vae.q_dist)

#         cond_entropies[3] += cond_entropies_i.cpu() / 32

#     metric = compute_metric_shapes(marginal_entropies, cond_entropies)
#     return metric, marginal_entropies, cond_entropies


def to_var(x, gpu_mode = True, **kwargs):
    if torch.cuda.is_available() & gpu_mode:
        x = x.cuda()
    return torch.Variable(x, **kwargs)


def to_np(x):
    # if isinstance(x, torch.Variable):
    #     x = x.data.cpu()
    # elif not torch.is_tensor(x):
    #         raise TypeError('We need tensor here.')
    return x.cpu().numpy()

# gen_noise_Gaussian = None
# long_tensor_to_onehot = None
# noise_dim = None
# code_dim = None
# num_unblock = None
# num_labels = None
# gen_random_labels = None

# def get_noise(num_ = 1):
#         out_noise = gen_noise_Gaussian(num_, noise_dim)
#         num_noise_unblock = max(num_unblock - code_dim, 0)
#         if num_noise_unblock < noise_dim:
#             out_noise[:, num_noise_unblock:] = 0
#         return out_noise
    
# def get_code(num_ = 1):
#     out_code = gen_noise_Gaussian(num_, code_dim)
#     if num_unblock < code_dim:
#         out_code[:, num_unblock:] = 0
#     return out_code

# def get_labels(num_, random = True):
#     if not random:
#         ll = torch.arange(0, num_labels).repeat(1, num_ // num_labels + 1)[0,:num_].long()
#     else:
#         ll = gen_random_labels( num_, num_labels )
    
#     out_labels = long_tensor_to_onehot(ll, num_labels).float()
#     return ll, out_labels

# def getNoiseThisBatch(this_batch_size):
#     gpu_mode = True
#     var_z_noise = to_var( get_noise(this_batch_size), gpu_mode = gpu_mode )
#     var_z_code = to_var( get_code(this_batch_size), gpu_mode = gpu_mode )
#     var_z_labels = to_var( get_labels(this_batch_size)[1], gpu_mode = gpu_mode )
#     return var_z_noise, var_z_code, var_z_labels   

def getDisentanglementMetric(solver: Solver, trainloader: DataLoader, alpha = 0.5, real_sample_fitting_method = 'LinearRegression', subspace_fitting_method = 'OMP'):
    
    with torch.no_grad():
        z_dim = solver.model.num_latents

        # G = None # model.Generator ? GAN? decoder
        # getNoiseThisBatch = None
        # get_code = None
        
        print('Begin the calculation of the subspace score.')
        
        # self.G.eval()
        solver.model.eval()
        
        num_tries = 5
        # code variantion range
        num_range = 5
        code_variant = torch.linspace(-2, 2, num_range)
        # the number of samples per batch and the result
        num_sample_per_batch = 10
        
        class Reconstructor():
            def __init__(self, reconstruct_method):
                super().__init__()
                self.reconstruct_method = reconstruct_method

                method = {'LinearRegression': LinearRegression(fit_intercept=False, n_jobs = -1),
                            'Ridge': Ridge(alpha=1e-4, fit_intercept=False, tol=1e-2,), 
                            'Lasso': Lasso(alpha=1e-5, fit_intercept=False, warm_start = False, tol=1e-3,),
                            'ElasticNet': ElasticNet(alpha=1e-2, l1_ratio = 0.5, fit_intercept=False, tol=1e-3,),
                            'OMP': OMP(n_nonzero_coefs = num_range * num_sample_per_batch, fit_intercept=False ),
                            }
                self.clf = method[reconstruct_method]

                print('Reconstructor initialized with the method as %s' % reconstruct_method)

            def fit(self,X,Y):
                X, Y = normalize(X, axis=1), normalize(Y, axis=1) # unit length
                Xt, Yt = np.transpose(X), np.transpose(Y)
                # print('Reconstructor fit() called. Begin to fit from %s to %s.' % (str(X.shape), str(Y.shape)) )
                self.clf.fit(Xt, Yt)
                Y_hat = self.clf.coef_ @ X #+ self.clf.intercept_.reshape(1,-1)
                return np.mean(minkowski_distance(Y, Y_hat)), self.clf.coef_, Y_hat


            def fit_self(self,X):
                # print('Reconstructor fit_self() called. Begin to fit %s' % str(X.shape) )
                X = normalize(X, axis=1) # unit length
                num_sample = len(X)
                idx = np.arange(num_sample)
                result_matrix = np.zeros([num_sample, num_sample])
                for i1 in tnrange(num_sample, leave=False):
                    this_idx = np.delete(idx, i1 ).tolist()
                    coef = self.fit(X[this_idx, :], X[i1, :].reshape(1,-1))[1]
                    result_matrix[i1, this_idx] = coef
                return result_matrix
        

        final_result_batch = []
        for i0 in tnrange(num_tries, desc = 'total rounds'):

            # normalizer
            scaler = StandardScaler(with_mean=True, with_std=False)

            # reconstructor 
            print('Real samples fitting method')
            r_fit_real = Reconstructor(real_sample_fitting_method)
            print('Generated samples fitting method')
            r_fit_generated = Reconstructor(subspace_fitting_method)

            # get some of the real samples
            # trainloader_iter = iter(self.trainloader)
            trainloader_iter = iter(trainloader)

            ####### sample from dataloader:
            # num_of_batches = 200
            # num_of_batches = 12800 / 750
            num_of_batches = 2
            part_of_real_samples = torch.cat( [ next(trainloader_iter)[0] for _ in range(num_of_batches) ] )
            # if part_of_real_samples.dim() == 4:
            #     part_of_real_samples = part_of_real_samples.squeeze(1)

            # batch_size = 200?
            part_of_real_samples_np = part_of_real_samples.view(part_of_real_samples.size()[0], -1).numpy()

            #######
            
            part_of_real_samples_np = scaler.fit_transform(part_of_real_samples_np)

            total_samples_batch = []
            total_labels_batch = []
            # generate sequences for every code
            for i1 in tnrange(z_dim):

                ####### TODO: generate noisy samples?
                # in_noise, in_code, in_labels = getNoiseThisBatch( num_sample_per_batch )
                # this_code = get_code( num_sample_per_batch )#.zero_()
                this_code = torch.randn(num_sample_per_batch, z_dim)
                #######

                # this_code = torch.zeros(this_code.size())
                # each code varies
                for i2 in range(num_range):
                    this_code[:,i1] = code_variant[i2]
                    # in_code = to_var(this_code, gpu_mode = gpu_mode, volatile = True)
                    in_code = this_code

                    ####### call decoder

                    (x_mu, x_logsigma) = solver.model.decode(in_code)
                    batch_size = num_sample_per_batch
                    x_mu = x_mu.view(batch_size, solver.model.input_size * solver.model.input_size)
                    x_logsigma = x_logsigma.view(batch_size, solver.model.input_size * solver.model.input_size)

                    if solver.model.output_dist == 'normal':
                        samples = solver.model.reparameterize(x_mu, x_logsigma)
                    elif solver.model.output_dist == 'bernoulli':
                        samples = solver.model.reparameterize_bernoulli(x_mu)
                    else:
                        samples = x_mu             

                    total_samples_batch.append( samples )
                    total_labels_batch.append( torch.ones(num_sample_per_batch) * i1 )

            # all the generated sequence
            total_samples = torch.cat(total_samples_batch)
            total_labels = torch.cat(total_labels_batch)

            
            
            # numpy format
            total_samples_np = to_np( total_samples.view(total_samples.size()[0], -1) ) 
            total_samples_np = scaler.transform(total_samples_np)
            total_labels_np = to_np( total_labels )

            # print('Begin to fit real samples.')
            # the reconstruction accuracy of the real samples from the generated samples
            reconstructed_accuracy = r_fit_real.fit(total_samples_np, part_of_real_samples_np)[0]
            
            # print('Begin to fit generated samples.')
            # fit the total_samples_np with itself
            coefficient_matrix = r_fit_generated.fit_self(total_samples_np)

            # symmetrization
            coefficient_matrix_abs = np.abs(coefficient_matrix)
            coefficient_matrix_sym = coefficient_matrix_abs + np.transpose(coefficient_matrix_abs)
            
            # # show the ``covariance'' matrix
            # plt.imshow( coefficient_matrix_sym / np.max(coefficient_matrix_sym) )
            # plt.show()
            
            # # # # # # self.tbx_writer.add_image('coefficient_matrix', torch.from_numpy(coefficient_matrix_sym).view(1,1,total_samples.size()[0],total_samples.size()[0]), i0)
            
            # subspace clustering 
            label_hat = spectral_clustering(coefficient_matrix_sym, n_clusters = z_dim)

            NMI = metrics.normalized_mutual_info_score(label_hat, total_labels_np)
            
            final_result_this = (1 - reconstructed_accuracy) * alpha + NMI * (1 - alpha)
            
            to_print = 'ROUND {}:\n distance to projection:{}\n NMI:{}\n final result:{}\n'.format(i0, reconstructed_accuracy, NMI, final_result_this)
            # print( to_print )
            # # # # # # # self.tbx_writer.add_text('disentanglement metric', to_print, i0)

            final_result_batch.append(final_result_this)

        to_print = 'final subspace score value: {}+-{}'.format( np.mean(final_result_batch), np.std(final_result_batch) ) 
        print( to_print)
        # # # # # # # # self.tbx_writer.add_text('disentanglement metric', to_print, num_tries)

        return np.mean(final_result_batch)

def svr_joints(solver: Solver, train_dir: str, validation_dir: str):
    

    # with torch.no_grad():
    #     solver.model.eval()
    #     solver.model.to("cuda")

    #     dataloader = create_dataloader(train_dir, 500, normalize_input=True, max_persons=-1, shuffle=False)
    #     merged_z = None
    #     merged_joints = None

    #     for batch_idx, batch_input in enumerate(dataloader):

    #         joints = batch_input[1].to("cuda")
    #         batch_input = batch_input[0].to("cuda")

    #         if batch_input.dim() == 4:
    #             batch_input = batch_input.squeeze(1)

    #         batch_output, x_params, z, z_params = solver.model(batch_input)
            
    #         if merged_z is None:
    #             merged_z = z.cpu().numpy()
    #             merged_joints = joints.cpu().numpy()
    #         else:
    #             merged_z = np.concatenate((merged_z, z.cpu().numpy()))
    #             merged_joints = np.concatenate((merged_joints, joints.cpu().numpy()))

    #     np.savez("../data/merged_z.npz", joints=merged_joints, z_code=merged_z)



    output_dist = None
    # if args.output_dist:
    #     output_dist = args.output_dist

    milestones = [999999]

    # frame_model = FullyConnected(in_size=solver.model.input_size*solver.model.input_size, out_size=21)
    # model:
    z_logger = Logger("FullyConnected", log_interval=10, basename="{}_{}_{}".format("z_joints", "fc", solver.model.num_latents))
    frame_logger = Logger("FullyConnected", log_interval=10, basename="{}_{}_{}".format("z_joints", "fc", solver.model.num_latents))
    z_model = FullyConnected(in_size=solver.model.num_latents)
    frame_model = FullyConnected(in_size=solver.model.input_size*solver.model.input_size)
    z_loss_function = LossJoints(z_logger)
    frame_loss_function = LossJoints(frame_logger)
    # LossJoints

    # loss_function.output_dist = model.output_dist

    # torch.nn.init.xavier_uniform.apply(model)

    # optimizer and learning rate scheduler:
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    z_optimizer = optim.Adam(z_model.parameters(), lr=0.000005, betas=(0.9, 0.999))
    frame_optimizer = optim.Adam(frame_model.parameters(), lr=0.000005, betas=(0.9, 0.999))
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100, 200, 300, 400, 500], gamma=0.1)
    z_scheduler = optim.lr_scheduler.MultiStepLR(z_optimizer, milestones=milestones, gamma=0.1)
    frame_scheduler = optim.lr_scheduler.MultiStepLR(frame_optimizer, milestones=milestones, gamma=0.1)

    z_solver = Solver(z_model, z_loss_function, z_optimizer, z_scheduler, "cuda", z_logger)
    frame_solver = Solver(frame_model, frame_loss_function, frame_optimizer, frame_scheduler, "cuda", frame_logger)

    z_solver.train("../data/merged_z.npz", None, 2500, 1000, 10, "../models", False, max_persons=-1, shuffle=False)
    z_solver.save_checkpoint("test_model_z.pt")
    frame_solver.train(train_dir, None, 500, 1000, 10, "../models", False, max_persons=-1, shuffle=False)
    z_solver.save_checkpoint("test_model_frames.pt")

    print("done")
    return

    # solver:
    # return Solver(z_model, loss_function, optimizer, scheduler, "cuda", solver.logger)

    # with torch.no_grad():
    #     solver.model.eval()
    #     solver.model.to("cuda")
    #     for batch_idx, batch_input in enumerate(dataloader):

            
    #         frames = batch_input[0].to("cuda") # batch_input[1] contains the target data (if this wasn't an autoencoder...)
    #         joints = batch_input[1].to("cuda")
    #         batch_size = frames.shape[0]

    #         (z_mu, z_logsigma) = solver.model.encode(frames)
    #         z_mu = z_mu.view(batch_size, solver.model.num_latents)
    #         z_logsigma = z_logsigma.view(batch_size, solver.model.num_latents)
    #         z_code = solver.model.reparameterize(z_mu, z_logsigma)
        
            
    #         z_code = z_code.view(batch_size, -1).cpu().numpy()
    #         joints = joints.view(batch_size, -1).cpu().numpy()
    #         frames = frames.view(batch_size, -1).cpu().numpy()

    #         regr = LinearSVR(random_state=0, tol=1e-5)
    #         regr.fit(z_code, joints)
    #         regr.fit(frames, joints)
    #         pass

def compute_kld(solver: Solver, dataloader: DataLoader, device="cuda" ):

    with torch.no_grad():

        solver.model.eval()
        solver.model.to(device)
        for batch_idx, batch_input in enumerate(dataloader):

            # joints = None
            # if len(batch_input) > 1:
            #     joints = batch_input[1].to(device)

            batch_input = batch_input[0].to(device) # batch_input[1] contains the target data (if this wasn't an autoencoder...)

            if batch_input.dim() == 4: # if rendered dataset; msra native has dim() == 3
                batch_input = batch_input.squeeze(1)

            # if train:
            #     if self.accumulate_grad:
            #         if count == 0:
            #             optimizer.step()
            #             optimizer.zero_grad()
            #             count = self.batch_multiplier
            #     else:
            #         optimizer.zero_grad()

            # if solver.joint_model is not None:
            #     solver.joint_model.to(device)
            #     with torch.no_grad():
            #         solver.model.eval()
            #         # (z_mu, z_logsigma) = model.encode(batch_input)
            #         # z_mu = z_mu.view(batch_input.size(0), model.num_latents)
            #         # z_logsigma = z_logsigma.view(batch_input.size(0), model.num_latents)
            #         # z = model.reparameterize(z_mu, z_logsigma)

            #         batch_output, x_params, z, z_params, _ = solver.model(batch_input)
            #         (mu, logsigma) = z_params

            #     _, _, _, _, recon_joints = solver.joint_model(z)
            #         # (z_mu, z_logsigma) = z_params
            # else:
            _, _, _, z_params, _ = solver.model(batch_input)
            (mu, logsigma) = z_params

            # plot joint-model:
            # plot_joint_model = False
            # if plot_joint_model and x_params == (None, None) and z is not None:
            #     batch_size = batch_input.size(0)
            #     (x_mu, x_logsigma) = model.vae.decode(z)
            #     x_mu = x_mu.view(-1, model.vae.input_size * model.vae.input_size)
            #     x_logsigma = x_logsigma.view(batch_size, model.vae.input_size * model.vae.input_size)

            #     if model.vae.output_dist == 'normal':
            #         x = model.vae.reparameterize(x_mu, x_logsigma)
            #         x_mu = x_mu.view(batch_size, model.vae.input_size, model.vae.input_size)
            #         x_logsigma = x_logsigma.view(batch_size, model.vae.input_size, model.vae.input_size)
            #     # elif model.vae.output_dist == 'bernoulli':
            #         # x = model.vae.reparameterize_bernoulli(x_mu)
            #     elif model.vae.output_dist == 'fake_normal':
            #         x_logsigma = torch.zeros_like(x_logsigma)
            #         x = model.vae.reparameterize(x_mu, x_logsigma)
            #         x_mu = x_mu.view(batch_size, model.vae.input_size, model.vae.input_size)
            #         x_logsigma = x_logsigma.view(batch_size, model.vae.input_size, model.vae.input_size)
            #     # else:
            #         #x = x_params
            #         # x = x_mu

            #     # x = x.view(batch_size, model.vae.input_size, model.vae.input_size)
            #     x_params = (x_mu, x_logsigma)
            #     # return x, (x_mu, x_logsigma), z, (z_mu, z_logsigma), None
            
            loss_function = LossFunction()
            total_kld, dimwise_kld, mean_kld = loss_function.kld_loss(mu, logsigma)
            # if self.accumulate_grad:
            #     loss_value /= self.batch_multiplier

            # lass_value_item = loss_value.item()
            # loss_sum += lass_value_item

            return total_kld, dimwise_kld, mean_kld

def traverse_latents(solver: Solver, dataloader: DataLoader, input_frame: Tensor, device="cuda", latents=torch.arange(-3, 3.1, 2/3.), log_label = "Traverse"):

    with torch.no_grad():

        solver.model.eval()
        solver.model.to(device)

        # compute kld from a random minibatch:
        total_kld, dimwise_kld, mean_kld = compute_kld(solver, dataloader, device)


        batch_input = input_frame.to(device).unsqueeze(0)
        #batch_output, mu, logsigma = solver.model(batch_input)
        batch_output, x_params, z, z_params, recon_joints = solver.model(batch_input)
        # mu = z_params[:,:int(z_params.size(1) / 2)]
        # logsigma = z_params[:,:int(-z_params.size(1) / 2)]
        #mu = z_params.select(-1, 0)
        #logsigma = z_params.select(-1, 1)
        (mu, logsigma) = z_params

        #z = solver.model.reparameterize(mu, logsigma)
        output = batch_output.squeeze(0)

        # dpt_reconst = output.cpu().numpy()
        # dpt_postprocessed = postprocess_frame(dpt_reconst)
        # joints = None

        #random_z = torch.Tensor(torch.cuda(torch.rand(1, z.shape[1]), True), volatile=True)
        # random_z = torch.Tensor(z.shape).to(device)
        # random_z.random_()


        #test = "../data/MSRA15_Preprocessed/test/P0/1/000445_depth.bin"
        # test = filename
        # dpt, M, com = load_frame(test, docom=True)

        # if cube == None:
        #     cube = [] # Min/Max
        # else:
        #     # normalize input [-1, 1]
        #     dpt[dpt == 0] = com[2] + (cube[2] / 2.)
        #     dpt = dpt - com[2]
        #     dpt = dpt / (cube[2] / 2.)
        # batch_input = torch.from_numpy(dpt).to(device).unsqueeze(0)
        # #x2 = solver.model.encode(batch_input)
        # x2 = solver.model.encode(batch_input)
        # mu2 = solver.model.fc_mean(x2)
        # logsigma2 = solver.model.fc_std(x2)
        # z2 = solver.model.reparameterize(mu2, logsigma2)

        #batch_output = solver.model.decode(z2)
        rgb_input = batch_input[0].unsqueeze(0) # add 1 for the color channel
        rgb_output = batch_output[0].unsqueeze(0) # add 1 for the color channel
        rgb_input_output = torch.stack([rgb_input, rgb_output], dim=0)
        #test = torch.stack(rgb_input, rgb_output)
        #grid_output = torchvision.utils.make_grid(rgb_output, nrow=1, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
        #grid_input = torchvision.utils.make_grid(rgb_input, nrow=1, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
        grid = torchvision.utils.make_grid(rgb_input_output, nrow=2, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
        np_grid = grid.cpu().data.numpy()
        np_grid = np.transpose(np_grid, axes=(1, 2, 0))
        scipy.misc.imsave('gif/{}{}_input_output.jpg'.format(log_label, solver.logger.basename), np_grid)

        # input + unmodified reconstructed output (optional postprocessing for all or none)
        solver.logger.add_image( "{}/Input_Output".format(log_label), grid.cpu().data.numpy(), 0)
        #solver.logger.add_image( "Output", grid_output.cpu().data.numpy(), 0)

        # sort latents by their kld, plot only the n highest latents

        # A) latent meaning is known (e.g number of fingers, rotation etc):
        # x-axis = [-3, 3] (tcvae: [-6, 6], [0, 6], [6, 0])
        # y-axis = example gesture
        # time-step = None?

        # B) visualizing changes:
        # x-axis = [-3, 3]
        # y-axis = latent-idx


        total_grid = torch.Tensor(z.shape[1], len(latents), input_frame.shape[0], input_frame.shape[1]) # 10, 75, 128, 128
        for latent_idx in range(z.shape[1]): # 0; 75
            z_clone = z.repeat(len(latents), 1) # batch of size len(latent value range) # 10, 128, 128
            before = z_clone[0, latent_idx].cpu().numpy()

            for batch_idx, latent_val in enumerate(latents): # -3; 3
                #z_clone = z.clone()
                #z_clone.repeat(z.shape[1], 1)
                #print("before: {}".format(z_clone[0, latent_idx].cpu().numpy()))
                z_clone[batch_idx, latent_idx] = latent_val
                #print("{:.4f} => {:.4f}".format(before, latent_val))
                #print("after: {}".format(z_clone[0, latent_idx].cpu().numpy()))

            #batch_modified = solver.model.decode(z_clone)
            batch_size = z_clone.shape[0]
            #x_params = solver.model.decode(z_clone).view(z_clone.shape[0], input_frame.shape[0] * input_frame.shape[1], solver.model.num_output_params)
            #(x_mu, x_logsigma) = solver.model.decode(z_clone).view(z_clone.shape[0], input_frame.shape[0] * input_frame.shape[1], solver.model.num_output_params)
            (x_mu, x_logsigma) = solver.model.decode(z_clone)#.view(batch_size, self.input_size * self.input_size, self.num_output_params)
            x_mu = x_mu.view(batch_size, solver.model.input_size * solver.model.input_size)
            x_logsigma = x_logsigma.view(batch_size, solver.model.input_size * solver.model.input_size)

            if solver.model.output_dist == 'normal':
                x = solver.model.reparameterize(x_mu, x_logsigma)
            elif solver.model.output_dist == 'bernoulli':
                x = solver.model.reparameterize_bernoulli(x_mu)
            else:
                x = x_mu
            total_grid[latent_idx] = x.view(x.shape[0], input_frame.shape[0], input_frame.shape[1])

        counter = 0
        for batch_idx, latent_val in enumerate(latents): # -3; 3
            rgb_modified = total_grid[:,batch_idx].unsqueeze(1) # add 1 for the color channel
            rgb_modified = rgb_modified.transpose(-1, -2)
            grid = torchvision.utils.make_grid(rgb_modified, nrow=2, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
            grid = grid.transpose(-1, -2)
            np_grid = grid.cpu().data.numpy()
            solver.logger.add_image( "{}/Latents".format(log_label), np_grid, counter)
            np_grid = np.transpose(np_grid, axes=(1, 2, 0))
            scipy.misc.imsave('gif/{}{}_{}.jpg'.format(log_label, solver.logger.basename, str(counter).zfill(2)), np_grid)
            counter += 1

        # save for each dim in z: a row of traversing images up to max_z and max_cols
        # sort rows by kld

        # call gif maker tool
        try:
            subprocess.check_output(["convert", "-loop", "0", "-delay", "6", "gif/{}{}_*.jpg".format(log_label, solver.logger.basename), "gif/{}{}.gif".format(log_label, solver.logger.basename)], shell=True)
        except subprocess.CalledProcessError as e:
            print(e)

            # rgb_modified = batch_modified[:1].unsqueeze(1) # add 1 for the color channel
            # #grid = torchvision.utils.make_grid(rgb_modified, nrow=1, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
            # grid = torchvision.utils.make_grid(rgb_modified, nrow=z.shape[0], padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
            # #solver.logger.add_image( "Latent_{}".format(latent_idx), grid_modified.cpu().data.numpy(), solver.logger.step)
            # solver.logger.add_image( "Traverse/Latent_{}".format(latent_idx), grid.cpu().data.numpy(), counter)
            # counter += 1
            # solver.logger.next_step()
            # solver.logger.add_text( "Latent_{}", "test_str", counter)
            # solver.logger.add_scalars("", )



        # create_gif = False
        # if create_gif:
        #     counter = -1
        #     gif = []
        #     for latent_val in latents:
        #         counter += 1
        #         z_clone = z.clone()
        #         z_clone = z_clone.repeat(z.shape[1], 1)
        #         for latent_idx in range(z.shape[1]):
        #             z_clone[latent_idx, latent_idx] = latent_val
        #             #print("{:.4f} => {:.4f}".format(before, latent_val))

        #         batch_modified = solver.model.decode(z_clone)
        #         rgb_modified = batch_modified.unsqueeze(1) # add 1 for color channel
        #         grid = torchvision.utils.make_grid(rgb_modified, nrow=10, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
        #         solver.logger.add_image( "Traverse/Modified", grid.cpu().data.numpy(), counter)
        #         gif.append(grid.cpu().data.numpy())

        #     gif.extend(list(reversed(gif)))
        #     gif = (np.array(gif) * 256).astype(np.uint8)
        #     write_gif(gif, "test.gif", fps=5)


        # for latent_idx in range(z.shape[1]): # 0; 75
        #     counter = 0
        #     for latent_val in latents: # -3; 3
        #         #z_clone = z.clone()
        #         #z_clone.repeat(z.shape[1], 1)
        #         z_clone = z.repeat(z.shape[1], 1)
        #         #print("before: {}".format(z_clone[0, latent_idx].cpu().numpy()))
        #         before = z_clone[0, latent_idx].cpu().numpy()
        #         z_clone[0, latent_idx] = latent_val
        #         print("{:.4f} => {:.4f}".format(before, latent_val))
        #         #print("after: {}".format(z_clone[0, latent_idx].cpu().numpy()))
        #         batch_modified = solver.model.decode(z_clone)

        #         rgb_modified = batch_modified[:1].unsqueeze(1) # add 1 for the color channel
        #         #grid = torchvision.utils.make_grid(rgb_modified, nrow=1, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
        #         grid = torchvision.utils.make_grid(rgb_modified, nrow=z.shape[1], padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
        #         #solver.logger.add_image( "Latent_{}".format(latent_idx), grid_modified.cpu().data.numpy(), solver.logger.step)
        #         solver.logger.add_image( "Traverse/Latent_{}".format(latent_idx), grid.cpu().data.numpy(), counter)
        #         counter += 1
        #         # solver.logger.next_step()
        #         # solver.logger.add_text( "Latent_{}", "test_str", counter)
        #         # solver.logger.add_scalars("", )

        # #batch_output = solver.model.decode(z)
        # grid = torchvision.utils.make_grid(rgb_modified, nrow=10, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
        # solver.logger.add_image( "Traverse/Latent_{}".format(latent_idx), grid.cpu().data.numpy(), counter)


def save_graph(solver: Solver, input_frame=torch.randn(75, 1, 128, 128, dtype=torch.float)):

    # onnx can't handle max_(un)pool2d with indices
    solver.model.to("cpu")

    # Rather than using the default transforms, build custom ones to group
    # nodes of residual and bottleneck blocks.
    # transforms = [
    #     # Fold Conv, BN, RELU layers into one
    #     # hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    #     hl.transforms.Fold("Conv > BatchNorm > LeakyRelu", "ConvBnLeakyRelu"),
    #     # Fold Conv, BN layers together
    #     # hl.transforms.Fold("Conv > BatchNorm", "ConvBn"),
    #     hl.transforms.Fold("Constant > Unsqueeze", "Constant"),
    #     hl.transforms.Fold("Gather > Unsqueeze", "Gather"),
    #     # Fold bottleneck blocks
    #     hl.transforms.Fold("""
    #         ((ConvBnRelu > ConvBnRelu > ConvBn) | ConvBn) > Add > Relu
    #         """, "BottleneckBlock", "Bottleneck Block"),
    #     # Fold residual blocks
    #     hl.transforms.Fold("""ConvBnRelu > ConvBnRelu > ConvBn > Add > Relu""",
    #                     "ResBlock", "Residual Block"),
    #     # Fold repeated blocks
    #     hl.transforms.FoldDuplicates(),
    # ]
    transforms = []
    
    im = hl.build_graph(solver.model, input_frame, transforms=transforms)
    im.save(path="../plot/my_model.jpg" , format="jpg")
    # torch.onnx.export(solver.model, input_frame, 'model.onnx')
    # solver.logger.add_graph(solver.model, input_frame)


def plot_3d_model(solver: Solver, input_frame: Tensor, title: str=None):
    
    fig = plt.figure()
    if title != None:
	    fig.suptitle(title, fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    
    dpt = input_frame.cpu().numpy()
    x = np.arange(128)
    y = np.arange(128)
    z = dpt[x, y]
    
    x = np.linspace(0, 1, 128)
    y = np.linspace(0, 1, 128)
    xv, yv = np.meshgrid(x, y)
    dem3d=ax.plot_surface(xv,yv,dpt,cmap='afmhot')
    
    plt.show(block=False)


def run(solver: Solver, filename: Path, device="cuda"):

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

    if not os.path.isfile(filename):
        print("Cannot load {}".format(filename))
        return

    with torch.no_grad():
        solver.model.to(device)
        dpt, M, com = load_frame(filename, docom=True, cube=cube)
        batch_input = torch.from_numpy(dpt).to(device).unsqueeze(0)
        #batch_output, mu, logsigma = solver.model(batch_input)
        batch_output, x_params, z, z_params, recon_joints = solver.model(batch_input)
        mu = z_params.select(-1, 0)
        logsigma = z_params.select(-1, 1)
        output = batch_output.squeeze(0)

    dpt_reconst = output.cpu().numpy()
    dpt_postprocessed = postprocess_frame(dpt_reconst)
    joints = None

    # test_pcl = MSRA15Importer.depthToPCL(dpt, M, background_val=0.)
    # test_pcl_reconst = MSRA15Importer.depthToPCL(dpt_reconst, M, background_val=0.)

    #save_wavefront("test.obj", test_pcl)

    # TODO: Save output frame as image
    # TODO: convert to point cloud
    # TODO: joints from encoding

    return (dpt_postprocessed, joints)


# def run(self, filename: Path, device="cuda"):

#     default_cubes = {
#         'P0': (200, 200, 200),
#         'P1': (200, 200, 200),
#         'P2': (200, 200, 200),
#         'P3': (180, 180, 180),
#         'P4': (180, 180, 180),
#         'P5': (180, 180, 180),
#         'P6': (170, 170, 170),
#         'P7': (160, 160, 160),
#         'P8': (150, 150, 150)}
#     # todo: allow passing optional args.input_cube instead of default cube
#     cube = None
#     for p, c in default_cubes.items():
#         if p in filename:
#             cube = c
#             break
    
#     if not os.path.isfile(filename):
#         print("Cannot load {}".format(filename))
#         return

#     with torch.no_grad():
#         self.model.to(device)
#         dpt, M, com = load_frame(filename, docom=True, cube=cube)
#         batch_input = torch.from_numpy(dpt).to(device).unsqueeze(0)
#         batch_output, mu, logsigma = self.model(batch_input)
#         output = batch_output.squeeze(0)

#     dpt_reconst = output.cpu().numpy()
# #     dpt_postprocessed = postprocess_frame(dpt_reconst)
#     joints = None

# #     # test_pcl = MSRA15Importer.depthToPCL(dpt, M, background_val=0.)
# #     # test_pcl_reconst = MSRA15Importer.depthToPCL(dpt_reconst, M, background_val=0.)

#     #save_wavefront("test.obj", test_pcl)

#     # TODO: Save output frame as image
#     # TODO: convert to point cloud
#     # TODO: joints from encoding
    
#     plot3d = False
#     if plot3d:
#         self.plot_depth3(dpt, "Groundtruth")
#         self.plot_depth3(dpt_reconst, "Reconstructed")
#         self.plot_depth3(dpt_postprocessed, "Postprocessed")
#         plt.show()

#     return (dpt_postprocessed, joints)


# def save_graph(self):
#     # onnx can't handle max_(un)pool2d with indices
#     self.model.to("cpu")
#     input = torch.randn(5, 128, 128, dtype=torch.float)
#     self.logger.add_graph(self.model, input)


# def plot_depth3(self, dpt, title=None):

#     fig = plt.figure()
#     if title != None:
#         fig.suptitle(title, fontsize=16)
#     ax = fig.add_subplot(111, projection='3d')

#     x = np.arange(128)
#     y = np.arange(128)
#     z = dpt[x, y]

#     x = np.linspace(0, 1, 128)
#     y = np.linspace(0, 1, 128)
#     xv, yv = np.meshgrid(x, y)
#     dem3d=ax.plot_surface(xv,yv,dpt,cmap='afmhot')

#     plt.show(block=False)

# def traverse_latents(self, filename: Path, device="cuda", cube=None, latents=torch.arange(-3, 3.1, 2/3.)):

#     if not os.path.isfile(filename):
#         print("Cannot load {}".format(filename))
#         return

#     with torch.no_grad():
#         self.model.to(device)
#         dpt, M, com = load_frame(filename, docom=True, cube=cube)
#         batch_input = torch.from_numpy(dpt).to(device).unsqueeze(0)
#         batch_output, mu, logsigma = self.model(batch_input)
#         z = self.model.reparameterize(mu, logsigma)
#         output = batch_output.squeeze(0)

#         dpt_reconst = output.cpu().numpy()
# #         dpt_postprocessed = postprocess_frame(dpt_reconst)
#         joints = None

#         #random_z = torch.Tensor(torch.cuda(torch.rand(1, z.shape[1]), True), volatile=True)
#         # random_z = torch.Tensor(z.shape).to(device)
#         # random_z.random_()


#         #test = "../data/MSRA15_Preprocessed/test/P0/1/000445_depth.bin"
#         # test = filename
#         # dpt, M, com = load_frame(test, docom=True)

#         # if cube == None:
#         #     cube = [] # Min/Max
#         # else:
#         #     # normalize input [-1, 1]
#         #     dpt[dpt == 0] = com[2] + (cube[2] / 2.)
#         #     dpt = dpt - com[2]
#         #     dpt = dpt / (cube[2] / 2.)
#         # batch_input = torch.from_numpy(dpt).to(device).unsqueeze(0)
#         # #x2 = self.model.encode(batch_input)
#         # x2 = self.model.encode(batch_input)
#         # mu2 = self.model.fc_mean(x2)
#         # logsigma2 = self.model.fc_std(x2)
#         # z2 = self.model.reparameterize(mu2, logsigma2)

#         #batch_output = self.model.decode(z2)
#         rgb_input = batch_input[0].unsqueeze(0) # add 1 for the color channel
#         rgb_output = batch_output[0].unsqueeze(0) # add 1 for the color channel
#         rgb_input_output = torch.stack([rgb_input, rgb_output], dim=0)
#         #test = torch.stack(rgb_input, rgb_output)
# #         #grid_output = torchvision.utils.make_grid(rgb_output, nrow=1, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
# #         #grid_input = torchvision.utils.make_grid(rgb_input, nrow=1, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
# #         grid = torchvision.utils.make_grid(rgb_input_output, nrow=2, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)

#         # input + unmodified reconstructed output (optional postprocessing for all or none)
#         self.logger.add_image( "Traverse/Input_Output", grid.cpu().data.numpy(), 0)
#         #self.logger.add_image( "Output", grid_output.cpu().data.numpy(), 0)

#         # sort latents by their kld, plot only the n highest latents

#         # A) latent meaning is known (e.g number of fingers, rotation etc):
#         # x-axis = [-3, 3] (tcvae: [-6, 6], [0, 6], [6, 0])
#         # y-axis = example gesture
#         # time-step = None?

#         # B) visualizing changes:
#         # x-axis = [-3, 3]
#         # y-axis = latent-idx

#         create_gif = False
#         if create_gif:
#             counter = -1
#             gif = []
#             for latent_val in latents:
#                 counter += 1
#                 z_clone = z.clone()
#                 z_clone = z_clone.repeat(z.shape[1], 1)
#                 for latent_idx in range(z.shape[1]):
#                     z_clone[latent_idx, latent_idx] = latent_val
#                     #print("{:.4f} => {:.4f}".format(before, latent_val))

#                 batch_modified = self.model.decode(z_clone)
#                 rgb_modified = batch_modified.unsqueeze(1) # add 1 for color channel
# #                 grid = torchvision.utils.make_grid(rgb_modified, nrow=10, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
#                 self.logger.add_image( "Traverse/Modified", grid.cpu().data.numpy(), counter)
#                 gif.append(grid.cpu().data.numpy())

#             gif.extend(list(reversed(gif)))
#             gif = (np.array(gif) * 256).astype(np.uint8)
#             write_gif(gif, "test.gif", fps=5)
            
#         return


#         # for latent_idx in range(z.shape[1]):
#         #     for latent_val in latents:
#         #         z_clone = z.clone()
#         #         counter = 0
#         #         z_clone.repeat(z.shape[1])
#         #         #print("before: {}".format(z_clone[0, latent_idx].cpu().numpy()))
#         #         before = z_clone[0, latent_idx].cpu().numpy()
#         #         z_clone[0, latent_idx] = latent_val
#         #         print("{:.4f} => {:.4f}".format(before, latent_val))
#         #         #print("after: {}".format(z_clone[0, latent_idx].cpu().numpy()))
#         #         batch_modified = self.model.decode(z_clone)

#         #         rgb_modified = batch_modified[:1].unsqueeze(1) # add 1 for the color channel
# #         #         #grid = torchvision.utils.make_grid(rgb_modified, nrow=1, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
# #         #         grid = torchvision.utils.make_grid(rgb_modified, nrow=z.shape[1], padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
#         #         #self.logger.add_image( "Latent_{}".format(latent_idx), grid_modified.cpu().data.numpy(), self.logger.step)
#         #         self.logger.add_image( "Traverse/Latent_{}".format(latent_idx), grid.cpu().data.numpy(), counter)
#         #         counter += 1
#         #         # self.logger.next_step()
#         #         # self.logger.add_text( "Latent_{}", "test_str", counter)
#         #         # self.logger.add_scalars("", )

#         # #batch_output = self.model.decode(z)
# #         # grid = torchvision.utils.make_grid(rgb_modified, nrow=10, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
#         # self.logger.add_image( "Traverse/Latent_{}".format(latent_idx), grid.cpu().data.numpy(), counter)


def train_joints(solver: Solver, train_dir: str, validation_dir: str):

    # if solver:
    z_logger = Logger("Joint-Trainer", log_interval=10, basename="{}_{}_{}".format("z_joints_new", "fc", solver.model.num_latents))
    z_model = FullyConnected(solver.model.num_latents)
    z_optimizer = optim.Adam(z_model.parameters(), lr=0.00005) #, betas=(0.9, 0.999))
    z_scheduler = None
    # z_model.to("cuda")
    z_solver = Solver(solver.model, LossJoints(z_logger), z_optimizer, z_scheduler, "cuda", z_logger, joint_model=z_model)

    z_solver.train(train_dir, validation_dir, 600, 250, 10, "../models", False, max_persons=-1, shuffle=False)
    z_solver.save_checkpoint("test_model_z.pt")
    # else:
    #     z_logger = Logger("Joint-Trainer", log_interval=10, basename="{}_{}_{}".format("z_joints_new", "fc", solver.model.num_latents))
    #     z_model = FullyConnected(solver.model)
    #     z_optimizer = optim.Adam(z_model.parameters(), lr=0.00005) #, betas=(0.9, 0.999))
    #     z_scheduler = None
    #     z_solver = Solver(z_model, LossJoints(z_logger), z_optimizer, z_scheduler, "cuda", z_logger)

    #     z_solver.train(train_dir, validation_dir, 600, 500, 10, "../models", False, max_persons=-1, shuffle=False)
    #     z_solver.save_checkpoint("test_model_z.pt")
    #     # CNN
    
    # with torch.no_grad():
    #     solver.model.eval()
    #     solver.model.to("cuda")

    #     dataloader = create_dataloader(train_dir, 500, normalize_input=True, max_persons=-1, shuffle=False)
    #     merged_z = None
    #     merged_joints = None

    #     for batch_idx, batch_input in enumerate(dataloader):

    #         joints = batch_input[1].to("cuda")
    #         batch_input = batch_input[0].to("cuda")

    #         if batch_input.dim() == 4:
    #             batch_input = batch_input.squeeze(1)

    #         batch_output, x_params, z, z_params = solver.model(batch_input)
            
    #         if merged_z is None:
    #             merged_z = z.cpu().numpy()
    #             merged_joints = joints.cpu().numpy()
    #         else:
    #             merged_z = np.concatenate((merged_z, z.cpu().numpy()))
    #             merged_joints = np.concatenate((merged_joints, joints.cpu().numpy()))

    #     np.savez("../data/merged_z.npz", joints=merged_joints, z_code=merged_z)



    # output_dist = None
    # # if args.output_dist:
    # #     output_dist = args.output_dist

    # milestones = [999999]

    # # frame_model = FullyConnected(in_size=solver.model.input_size*solver.model.input_size, out_size=21)
    # # model:
    # z_logger = Logger("FullyConnected", log_interval=10, basename="{}_{}_{}".format("z_joints", "fc", solver.model.num_latents))
    # frame_logger = Logger("FullyConnected", log_interval=10, basename="{}_{}_{}".format("z_joints", "fc", solver.model.num_latents))
    # z_model = FullyConnected(in_size=solver.model.num_latents)
    # frame_model = FullyConnected(in_size=solver.model.input_size*solver.model.input_size)
    # z_loss_function = LossJoints(z_logger)
    # frame_loss_function = LossJoints(frame_logger)
    # # LossJoints

    # # loss_function.output_dist = model.output_dist

    # # torch.nn.init.xavier_uniform.apply(model)

    # # optimizer and learning rate scheduler:
    # # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # z_optimizer = optim.Adam(z_model.parameters(), lr=0.000005, betas=(0.9, 0.999))
    # frame_optimizer = optim.Adam(frame_model.parameters(), lr=0.000005, betas=(0.9, 0.999))
    # #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100, 200, 300, 400, 500], gamma=0.1)
    # z_scheduler = optim.lr_scheduler.MultiStepLR(z_optimizer, milestones=milestones, gamma=0.1)
    # frame_scheduler = optim.lr_scheduler.MultiStepLR(frame_optimizer, milestones=milestones, gamma=0.1)

    # z_solver = Solver(z_model, z_loss_function, z_optimizer, z_scheduler, "cuda", z_logger)
    # frame_solver = Solver(frame_model, frame_loss_function, frame_optimizer, frame_scheduler, "cuda", frame_logger)

    # z_solver.train("../data/merged_z.npz", None, 2500, 1000, 10, "../models", False, max_persons=-1, shuffle=False)
    # z_solver.save_checkpoint("test_model_z.pt")
    # frame_solver.train(train_dir, None, 500, 1000, 10, "../models", False, max_persons=-1, shuffle=False)
    # z_solver.save_checkpoint("test_model_frames.pt")

    print("done")
    return

def save_plots():

    torch.manual_seed(42)

    model_dir = "../models"
    dataset_dir = "../data/preprocessed"

    models = ['vae', 'bvae', 'bvae2', 'tcvae_normal', 'tcvae', 'tcvae_fake_normal']
    # models = ['tcvae_normal']
    datasets = ['spread', 'shape', 'pose_new', 'open', 'msra', 'angles', 'merged']
    # datasets = ['merged']
    num_latents = 10
    epochs = 250
    learning_rate = 5e-4
    scheduler = None
    device = "cuda"
    log_interval = 10

    bvae_beta = 15.0

    bvae2_max_capacity = None
    bvae2_gamma = None

    tcvae_beta = 15.0

    joint_model = None

    for modelname in models:

        for datasetname in datasets:

            output_dist = None
            milestones = []

            log_dir = "tensorboard"
            log_basename = "{}_{}".format(modelname, datasetname)
            logger = Logger(
                model = modelname,
                log_interval = log_interval,
                log_dir = log_dir,
                basename = log_basename
            )
            
            if modelname == 'vae':
                loss_function = LossBVAE(logger, beta=1.0)
            elif modelname == 'bvae':
                loss_function = LossBVAE(logger, beta=bvae_beta)
            elif modelname == 'bvae2':
                loss_function = LossBVAE2(logger, max_capacity=bvae2_max_capacity, gamma=bvae2_gamma)
            elif modelname == 'tcvae_fake_normal':
                loss_function = LossTCVAE(logger, beta=tcvae_beta)
                output_dist = 'fake_normal'
            elif modelname == 'tcvae_normal':
                loss_function = LossTCVAE(logger, beta=tcvae_beta)
                output_dist = 'normal'
            elif modelname == 'tcvae':
                loss_function = LossTCVAE(logger, beta=tcvae_beta)
                output_dist = None

            model = VAE(num_latents, output_dist=output_dist)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

            solver = Solver(
                model = model,
                loss_function = loss_function,
                optimizer = optimizer,
                scheduler = scheduler,
                device = device,
                logger = logger,
                joint_model = joint_model
                )

            model_filename = "{}_{}_{}_{}.pt".format(modelname, datasetname, num_latents, epochs)
            solver.load_checkpoint(os.path.join(model_dir, model_filename))
            solver.model.eval()
            solver.model.to(device)

            train_filename = os.path.join(dataset_dir, "train_" + datasetname + ".npz")
            validate_filename = os.path.join(dataset_dir, "validate_" + datasetname + ".npz")
            test_filename = os.path.join(dataset_dir, "test_" + datasetname + ".npz")

            dataloader = create_dataloader(test_filename, 100)

            traverse_image_idx = 0
            if datasetname == 'spread':
                pass
            elif datasetname == 'shape':
                pass
            elif datasetname == 'pose_new':
                pass
            elif datasetname == 'open':
                pass
            elif datasetname == 'msra':
                pass
            elif datasetname == 'angles':
                pass

            # plot experiments
            save_plots_traversals(solver=solver, dataloader=dataloader, device=device, modelname=modelname, datasetname=datasetname, traverse_image_idx=traverse_image_idx)


def save_plots_traversals(solver: Solver, dataloader: DataLoader, device: str, modelname: str, datasetname: str, traverse_image_idx: int):

    input_frame: Tensor
    latents=torch.arange(-3, 3.1, 2/3.) # 10 steps
    latents=torch.arange(-3, 3.1, 4/5.) # 10 steps
    # latents=torch.arange(-3, 3.1, 2*2/3.) # 5 steps
    log_label = "Traverse"

    plot_basedir = "../plot"

    # max_dimensions = 6
    max_dimensions = solver.model.num_latents

    with torch.no_grad():

        solver.model.eval()
        solver.model.to(device)

        # compute kld from a random minibatch:
        total_kld, dimwise_kld, mean_kld = compute_kld(solver, dataloader, device)

        dimwise_kld = dimwise_kld.to("cpu").numpy()

        # load input_frame:
        input_frame = dataloader.dataset[traverse_image_idx][0]
        batch_input = input_frame.to(device).unsqueeze(0)
        batch_output, x_params, z, z_params, recon_joints = solver.model(batch_input)
        (mu, logsigma) = z_params
        (x_mu, x_logsigma) = x_params


        if solver.model.output_dist is not None:

            if solver.model.output_dist == 'normal':
                batch_output = solver.model.reparameterize(x_mu, x_logsigma)
            # elif solver.model.output_dist == 'fake_normal':
            #     batch_output = solver.model.reparameterize(x_mu, x_logsigma)
            elif solver.model.output_dist == 'bernoulli':
                batch_output = solver.model.reparameterize_bernoulli(x_mu)
            else:
                batch_output = x_mu


        output = batch_output.squeeze(0)
        # if modelname == 'tcvae_fake_normal':
        #     output = x_mu.squeeze(0)

        rgb_input = batch_input[0].unsqueeze(0) # add 1 for the color channel
        np_grid = torchvision.utils.make_grid(rgb_input, nrow=1, padding=0, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0).cpu().data.numpy()
        np_grid = np.transpose(np_grid, axes=(1, 2, 0))
        scipy.misc.imsave(os.path.join(plot_basedir, '{}_{}_input.jpg'.format(modelname, datasetname)), np_grid)

        rgb_output = batch_output[0].unsqueeze(0) # add 1 for the color channel
        np_grid = torchvision.utils.make_grid(rgb_output, nrow=1, padding=0, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0).cpu().data.numpy()
        np_grid = np.transpose(np_grid, axes=(1, 2, 0))
        scipy.misc.imsave(os.path.join(plot_basedir, '{}_{}_output.jpg'.format(modelname, datasetname)), np_grid)

        # sort latents by their kld, plot only the n highest latents

        # A) latent meaning is known (e.g number of fingers, rotation etc):
        # x-axis = [-3, 3] (tcvae: [-6, 6], [0, 6], [6, 0])
        # y-axis = example gesture
        # time-step = None?

        # B) visualizing changes:
        # x-axis = [-3, 3]
        # y-axis = latent-idx

        
        total_grid = torch.Tensor(z.shape[1], len(latents), solver.model.input_size, solver.model.input_size) # 10, 75, 128, 128
        # total_grid = torch.Tensor(max_dimensions, len(latents), solver.model.input_size, solver.model.input_size) # 10, 75, 128, 128
        # np_total_grid = None
        for latent_idx in range(z.shape[1]): # 0; 75
            z_clone = z.repeat(len(latents), 1) # batch of size len(latent value range) # 10, 128, 128

            for batch_idx, latent_val in enumerate(latents): # -3; 3
                z_clone[batch_idx, latent_idx] = latent_val

            batch_size = z_clone.shape[0]
            (x_mu, x_logsigma) = solver.model.decode(z_clone)#.view(batch_size, self.input_size * self.input_size, self.num_output_params)
            x_mu = x_mu.view(batch_size, solver.model.input_size * solver.model.input_size)
            x_logsigma = x_logsigma.view(batch_size, solver.model.input_size * solver.model.input_size)

            if solver.model.output_dist == 'normal':
                x = solver.model.reparameterize(x_mu, x_logsigma)
            # elif solver.model.output_dist == 'fake_normal':
            #     x = solver.model.reparameterize(x_mu, x_logsigma)
            elif solver.model.output_dist == 'bernoulli':
                x = solver.model.reparameterize_bernoulli(x_mu)
            else:
                x = x_mu
            # total_grid[latent_idx] = x.view(x.shape[0], 1, solver.model.input_size, solver.model.input_size).transpose()
            total_grid[latent_idx] = x.view(x.shape[0], solver.model.input_size, solver.model.input_size)

        # np_grid = np.transpose(total_grid, axes=(1, 2, 0))
        # scipy.misc.imsave('gif/traverse_{}_{}.jpg'.format(solver.logger.basename, str(counter).zfill(2)), np_grid)

        counter = -1
        # for batch_idx, latent_val in enumerate(latents): # -3; 3
        grid_list = []
        sorted_kld_ids = dimwise_kld.argsort()[::-1]
        for z_dim_idx in range(z.shape[1]):
            counter += 1
            if counter == max_dimensions:
                break
            # rgb_modified = total_grid[:,batch_idx].unsqueeze(1) # add 1 for the color channel
            rgb_modified = total_grid[sorted_kld_ids[z_dim_idx]].unsqueeze(1) # add 1 for the color channel
            grid = torchvision.utils.make_grid(rgb_modified, padding=0, nrow=len(latents), normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
            # np_grid = np.transpose(grid.cpu().data.numpy(), axes=(2, 1, 0))
            np_grid = np.transpose(grid.cpu().data.numpy(), axes=(1, 2, 0))
            grid_list.append(np_grid)

            # solver.logger.add_image( "{}/Latents".format(log_label), np_grid, counter)
            # scipy.misc.imsave('gif/traverse_{}_{}.jpg'.format(solver.logger.basename, str(counter).zfill(2)), np_grid)
            # filename = os.path.join(plot_basedir, '{}_{}_{}.jpg'.format(modelname, datasetname, str(z_dim_idx).zfill(2)))
            # scipy.misc.imsave(filename, np_grid)
            # counter += 1

        # n=5
        # ids = dimwise_kld.argsort()[::-1][:n]
        # grid_list[ids]
        grid_list_sorted = np.vstack(grid_list)
        # test = np.vstack(grid_list[ids.tolist()])[ids]

        grid = torchvision.utils.make_grid(torch.from_numpy(grid_list_sorted), padding=0, nrow=1, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
        # np_grid = np.transpose(grid.cpu().data.numpy(), axes=(1, 2, 0))
        np_grid = grid.cpu().data.numpy()
        # filename = os.path.join(plot_basedir, '{}_{}_{}.jpg'.format(modelname, datasetname, str(z_dim_idx).zfill(2)))
        filename = os.path.join(plot_basedir, '{}_{}.jpg'.format(modelname, datasetname))
        scipy.misc.imsave(filename, np_grid)

        # pass

        # save for each dim in z: a row of traversing images up to max_z and max_cols
        # sort rows by kld

        # call gif maker tool
        # try:
        #     subprocess.check_output(["convert", "-loop", "0", "-delay", "6", "gif/traverse_{}_*.jpg".format(solver.logger.basename), "gif/traverse_{}.gif".format(solver.logger.basename)], shell=True)
        # except subprocess.CalledProcessError as e:
        #     print(e)

def compare_joint_training():
    
    torch.manual_seed(42)

    model_dir = "../models"
    dataset_dir = "../data/preprocessed"

    models = [
        # 'vae',
        # 'bvae',
        # 'bvae2',
        'tcvae_fake_normal'
        ]
    datasets = [
        'spread',
        # 'shape',
        # 'pose_new',
        # 'open',
        # 'msra',
        # 'angles'
        ]
    num_latents = 10
    epochs = 250
    joint_epochs = 150
    learning_rate = 5e-4
    scheduler = None
    device = "cuda"
    log_interval = 10
    batch_size = 400
    batch_size = 15
    validation_batch_size = 400

    bvae_beta = 15.0

    bvae2_max_capacity = 250.0
    bvae2_gamma = 1000.0
    bvae2_capacity_increments = 1500000.0

    tcvae_beta = 15.0

    joint_models = [
        'untrained_unfrozen',
        'trained_frozen',
        'trained_unfrozen'
    ]

    for modelname in models:

        for datasetname in datasets:

            train_filename = os.path.join(dataset_dir, "train_" + datasetname + ".npz")
            validate_filename = os.path.join(dataset_dir, "validate_" + datasetname + ".npz")
            test_filename = os.path.join(dataset_dir, "test_" + datasetname + ".npz")

            train_filename = os.path.join(dataset_dir, "train_" + datasetname + "_new.npz")
            validate_filename = os.path.join(dataset_dir, "validate_" + datasetname + "_new.npz")
            test_filename = os.path.join(dataset_dir, "test_" + datasetname + "_new.npz")

            # dataloader = create_dataloader(train_filename_new, batch_size)

            for joint_model in joint_models:

                solver = create_solver(
                    modelname,
                    datasetname,
                    log_interval,
                    bvae_beta,
                    bvae2_max_capacity,
                    bvae2_gamma,
                    bvae2_capacity_increments,
                    tcvae_beta,
                    num_latents,
                    learning_rate,
                    scheduler,
                    device,
                    joint_model,
                    epochs,
                    model_dir
                    )

                # solver.load_checkpoint( "../models/{}_{}_{}_{}.pt".format(modelname, datasetname, joint_model, joint_epochs) )

                solver.train(train_filename, validate_filename, batch_size, joint_epochs, log_interval, model_dir, False, max_persons=-1, shuffle=False, validation_batch_size=validation_batch_size)
                solver.save_checkpoint(os.path.join(model_dir, "{}_{}_{}_new_large.pt".format(modelname, datasetname, joint_model)))
                
                # save test loss:
                solver.test(test_filename, batch_size, shuffle=False)
                

def create_solver(
    modelname,
    datasetname,
    log_interval,
    bvae_beta,
    bvae2_max_capacity,
    bvae2_gamma,
    bvae2_capacity_increments,
    tcvae_beta,
    num_latents,
    learning_rate,
    scheduler,
    device,
    joint_model,
    epochs,
    model_dir
    ):

    milestones = []

    log_dir = "runs"
    log_basename = "{}_{}_{}".format(modelname, datasetname, joint_model)
    logger = Logger(modelname, log_interval=log_interval, basename=log_basename)

    # logger = Logger(
    #     model = modelname,
    #     log_interval = log_interval,
    #     log_dir = log_dir,
    #     basename = log_basename
    # )
    
    output_dist = None
    if modelname == 'vae':
        loss_function = LossBVAE(logger, beta=1.0)
    elif modelname == 'bvae':
        loss_function = LossBVAE(logger, beta=bvae_beta)
    elif modelname == 'bvae2':
        loss_function = LossBVAE2(logger, max_capacity=bvae2_max_capacity, gamma=bvae2_gamma, iterations=bvae2_capacity_increments)
    elif modelname == 'tcvae_fake_normal':
        loss_function = LossTCVAE(logger, beta=tcvae_beta)
        output_dist = 'fake_normal'

    model = VAE(num_latents, output_dist=output_dist)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # loss_function = LossJoints(logger)
    solver = Solver(
            model = model,
            loss_function = loss_function,
            optimizer = optimizer,
            scheduler = scheduler,
            device = device,
            logger = logger,
            joint_model = None
            )

    if not 'untrained' in joint_model:
        model_filename = "{}_{}_{}_{}.pt".format(modelname, datasetname, num_latents, epochs)
        solver.load_checkpoint(os.path.join(model_dir, model_filename))
        solver.logger.epoch = 0
        solver.logger.step = 0

    if 'unfrozen' in joint_model:
        solver.model = JointModel(solver.model, FullyConnected(solver.model.num_latents))
        solver.optimizer = optim.Adam(solver.model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        solver.model.train()
    else:
        solver.joint_model = FullyConnected(solver.model.num_latents)
        solver.optimizer = optim.Adam(solver.joint_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        solver.model.eval()
        solver.joint_model.train()
        solver.joint_model.to(device)

    # model_filename = "{}_{}_{}_{}.pt".format(modelname, datasetname, num_latents, epochs)
    # solver.load_checkpoint(os.path.join(model_dir, model_filename))
    solver.model.to(device)

    return solver

def plot_datasets():

    # plot 3x3 images from each dataset
    torch.manual_seed(42)

    # model_dir = "../models"
    dataset_dir = "../data/preprocessed"
    plot_basedir = "../plot"

    # models = [
    #     # 'vae',
    #     # 'bvae',
    #     # 'bvae2',
    #     'tcvae_fake_normal'
    #     ]
    datasets = [
        'spread',
        'shape',
        'pose_new',
        'open',
        'msra',
        'angles',
        'merged_new'
        ]
    # num_latents = 10
    # epochs = 250
    # joint_epochs = 50
    # learning_rate = 5e-4
    # scheduler = None
    # device = "cuda"
    # log_interval = 10
    # batch_size = 15
    # validation_batch_size = 400

    # bvae_beta = 15.0

    # bvae2_max_capacity = 250.0
    # bvae2_gamma = 1000.0
    # bvae2_capacity_increments = 1500000.0

    # tcvae_beta = 15.0

    # joint_models = [
    #     'untrained_unfrozen',
    #     'trained_frozen',
    #     'trained_unfrozen'
    # ]

    # for modelname in models:

    num_examples_row_col = 5

    for datasetname in datasets:

        # train_filename = os.path.join(dataset_dir, "train_" + datasetname + ".npz")
        # validate_filename = os.path.join(dataset_dir, "validate_" + datasetname + ".npz")
        test_filename = os.path.join(dataset_dir, "test_" + datasetname + ".npz")

        # train_filename = os.path.join(dataset_dir, "train_" + datasetname + "_new.npz")
        # validate_filename = os.path.join(dataset_dir, "validate_" + datasetname + "_new.npz")
        # test_filename = os.path.join(dataset_dir, "test_" + datasetname + "_new.npz")

        dataloader = create_dataloader(test_filename, batch_size=num_examples_row_col**2)

        for batch_input in dataloader:

            batch_input = batch_input[0]#.to(self.device) # batch_input[1] contains the target data (if this wasn't an autoencoder...)

            if batch_input.dim() == 4: # if rendered dataset; msra native has dim() == 3
                batch_input = batch_input.squeeze(1)

            batch_size = batch_input.size(0)

            tmp_input = batch_input.unsqueeze(1) # color channel
            grid_input = torchvision.utils.make_grid(tmp_input, nrow=num_examples_row_col, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
            # logger.add_image( "{}/Ground-Truth".format(log_name), grid_input.cpu().data.numpy(), log_step) # var.detach().n

            np_grid = grid_input.cpu().data.numpy()
            np_grid = np.transpose(np_grid, axes=(1, 2, 0))
            # filename = os.path.join(plot_basedir, '{}_{}_{}.jpg'.format(modelname, datasetname, str(z_dim_idx).zfill(2)))
            filename = os.path.join(plot_basedir, '{}.jpg'.format(datasetname))
            scipy.misc.imsave(filename, np_grid)

            break

    return