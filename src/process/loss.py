import numpy as np
from numbers import Number
import math

import torch
import torch.nn.functional as F

from log.logger import Logger
from tensorboardX import SummaryWriter

class LossFunction:

    def __init__(self, logger: Logger=None, recon_loss_function=F.mse_loss, dataset_size=-1, output_dist=None):
        self.logger = logger
        self.recon_loss_function = recon_loss_function
        self.log_name = "Train"

        self.dataset_size = dataset_size
        self.output_dist = output_dist


    def set_log_name(self, log_name):
        self.log_name = log_name


    def recon_loss(self, recon_x, x):
        batch_size = x.shape[0]
        loss = self.recon_loss_function(recon_x, x, size_average=False).div(batch_size)
        if self.logger != None:
            self.logger.add_scalar("{}/Reconstruction Loss".format(self.log_name), loss.item(), global_step=self.logger.step)
        return loss


    def kld_loss(self, mu, logsigma): #tcvae
        """Computes KL(q||p) where q is the given distribution and p
        is the standard Normal distribution.
        """
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
        kld = logsigma.mul(2).add(1) - mu.pow(2) - logsigma.exp().pow(2)
        kld.mul_(-0.5)

        mean_kld = kld.mean(1).mean(0, True)
        total_kld = kld.sum(1).mean(0, True)
        dimwise_kld = kld.mean(0)

        if self.logger != None:
            # self.logger.add_scalar("{}/Total KLD".format(self.log_name), total_kld.item(), global_step=self.logger.step)
            # self.logger.add_scalar("{}/Mean KLD".format(self.log_name), mean_kld.item(), global_step=self.logger.step)

            dimwise_kld_dict = {
                "total": total_kld,
                "mean": mean_kld}
            for i in range(dimwise_kld.size(0)):
                dimwise_kld_dict['kld_{}'.format(i)] = dimwise_kld[i].item()
            self.logger.add_scalars("{}/Dimwise-KLD".format(self.log_name), dimwise_kld_dict, global_step=self.logger.step)
        return total_kld, dimwise_kld, mean_kld

    # def kld_loss(self, mu, logsigma): # bvae2
    #     # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    #     # KLD(N(mu, var) || N(0, I))
        
    #     kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())

    #     mean_kld = kld.mean(1).mean(0, True)
    #     total_kld = kld.sum(1).mean(0, True)
    #     dimwise_kld = kld.mean(0)
        
    #     if self.logger != None:
    #         self.logger.add_scalar("{}/Total KLD".format(self.log_name), total_kld.item(), global_step=self.logger.step)
    #         #self.logger.add_scalar("{}/Dimwise KLD".format(self.log_name), dimwise_kld.item(), global_step=self.logger.step)
    #         self.logger.add_scalar("{}/Mean KLD".format(self.log_name), mean_kld.item(), global_step=self.logger.step)
    #     return total_kld, dimwise_kld, mean_kld


    def loss(self, recon_x, x, mu=None, logsigma=None, x_params=None, z=None, recon_joints=None, joints=None):
        loss = self.compute_loss(recon_x, x, mu, logsigma, x_params, z, recon_joints, joints)
        if self.logger != None:
            self.logger.add_scalar("{}/Total Loss".format(self.log_name), loss.item(), global_step=self.logger.step)
        return loss


    def compute_loss(self, recon_x, x, mu=None, logsigma=None, x_params=None, z=None, recon_joints=None, joints=None):
        return self.recon_loss(recon_x, x)

class LossJoints(LossFunction):

    def __init__(self, logger: Logger=None, recon_loss_function=F.mse_loss):
        super(LossJoints, self).__init__(logger, recon_loss_function)

    def compute_loss(self, recon_x, x, mu, logsigma, x_params=None, z=None, recon_joints=None, joints=None):
        batch_size = joints.shape[0]
        loss = self.recon_loss_function(recon_joints, joints, size_average=False).div(batch_size)
        if self.logger != None:
            self.logger.add_scalar("{}/Joints Loss".format(self.log_name), loss.item(), global_step=self.logger.step)
        return loss

class LossBVAE(LossFunction):

    def __init__(self, logger: Logger=None, recon_loss_function=F.mse_loss, beta=1.0):
        super(LossBVAE, self).__init__(logger, recon_loss_function)
        self.beta = beta


    def compute_loss(self, recon_x, x, mu, logsigma, x_params=None, z=None, recon_joints=None, joints=None):
        total_kld, dimwise_kld, mean_kld = self.kld_loss(mu, logsigma)
        beta_kld_loss = self.beta * total_kld
        recon_loss = self.recon_loss(recon_x, x)
        total_loss = recon_loss + beta_kld_loss
        if self.logger != None:
            self.logger.add_scalars("{}/BVAE-Loss".format(self.log_name), {
                "total": total_loss.item(),
                "reconst": recon_loss.item(),
                "kld": total_kld.item(),
                "beta-kld": beta_kld_loss.item()}, global_step=self.logger.step)
        return total_loss


class LossBVAE2(LossFunction):
    
    def __init__(self, logger: Logger=None, recon_loss_function=F.mse_loss, gamma: float = 1000, capacity: float = 0.0, max_capacity: float = 50.0, iterations: int = 100000):
        super(LossBVAE2, self).__init__(logger, recon_loss_function)
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.max_iterations = iterations
        self.capacity = capacity


    def compute_loss(self, recon_x, x, mu, logsigma, x_params=None, z=None, recon_joints=None, joints=None):
        total_kld, dimwise_kld, mean_kld = self.kld_loss(mu, logsigma)
        self.capacity = np.clip(float(self.max_capacity) / float(self.max_iterations) * (self.logger.step+1), 0, self.max_capacity)
        reconst_loss = self.recon_loss(recon_x, x)
        gamma_kld = self.gamma * (total_kld - self.capacity).abs()
        total_loss = reconst_loss + gamma_kld
        if self.logger != None:
            #self.logger.add_scalar("{}/Capacity".format(self.log_name), self.capacity, self.logger.step)
            self.logger.add_scalars("{}/BVAE2-Loss".format(self.log_name), {
                "total": total_loss.item(),
                "reconst": reconst_loss.item(),
                "kld": total_kld.item(),
                "capacity": self.capacity,
                "max_capacity": self.max_capacity,
                "gamma_kld": gamma_kld.item()}, self.logger.step)
        return total_loss

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))

def normal_log_density(sample, mu, logsigma):
    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    c = np.log(2 * np.pi)
    return -0.5 * (tmp * tmp + 2 * logsigma + c)

class LossTCVAE(LossFunction):
    
    def __init__(self, logger: Logger=None, recon_loss_function=F.mse_loss, alpha=1.0, beta=6.0, gamma=1.0):
        super(LossTCVAE, self).__init__(logger, recon_loss_function)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.include_mutinfo = True
        self.mss = False
        self.lamb = 0
        self.tcvae = True

    def bernoulli_log_density(self, sample, params):
        #presigm_ps = self._check_inputs(sample.size(), params).type_as(sample)
        
        presigm_ps = params.expand(sample.size()).type_as(sample)

        eps = 1e-8
        p = (F.sigmoid(presigm_ps) + eps) * (1 - 2 * eps)
        logp = sample * torch.log(p + eps) + (1 - sample) * torch.log(1 - p + eps)
        return logp

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def compute_loss(self, recon_x, x, mu, logsigma, x_params, z, recon_joints=None, joints=None):

        # log p(x|z) + log p(z) - log q(z|x)

        batch_size = x.shape[0]
        num_latents = mu.shape[1]
        
        prior_mu = torch.zeros_like(mu)
        prior_logsigma = torch.zeros_like(logsigma)

        if self.output_dist == 'normal' or self.output_dist == 'fake_normal':
            (x_mu, x_logsigma) = x_params
            logpx = normal_log_density(x, x_mu, x_logsigma).view(batch_size, -1).sum(1)
        elif self.output_dist == 'bernoulli':
            (x_params, _) = x_params
            x_params = x_params.view(batch_size, x.shape[1], x.shape[2])
            logpx = self.bernoulli_log_density(x, x_params).view(batch_size, -1).sum(1)
        else:
            logpx = self.recon_loss(recon_x, x).mul(-1)
        
        logpz = normal_log_density(z, prior_mu, prior_logsigma).view(batch_size, -1).sum(1)
        logqz_condx = normal_log_density(z, mu, logsigma).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        if self.dataset_size <= 0 or (self.beta == 1 and self.include_mutinfo and self.lamb == 0):
            return elbo.mean().mul(-1)

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = normal_log_density(z.view(batch_size, 1, num_latents), mu, logsigma)

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * self.dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * self.dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, self.dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:

                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        mean_modified_elbo = modified_elbo.mean().mul(-1)
        
        if self.logger != None:

            mean_logpx = logpx.mean().mul(-1)
            mean_logqz_condx = logqz_condx.mean().mul(-1)
            mean_logqz = logqz.mean().mul(-1)
            mean_logqz_prodmarginals = logqz_prodmarginals.mean().mul(-1)
            mean_logpz = logpz.mean().mul(-1)
            mean_elbo = elbo.mean().mul(-1)

            self.logger.add_scalars("{}/TCVAE-Loss".format(self.log_name), {
                'logpx': mean_logpx,
                'logqz_condx': mean_logqz_condx,
                'logqz': mean_logqz,
                'logqz_prodmarginals': mean_logqz_prodmarginals,
                'logpz': mean_logpz,
                'elbo': mean_elbo,
                'modified_elbo': mean_modified_elbo}, self.logger.step)
        
        return mean_modified_elbo
