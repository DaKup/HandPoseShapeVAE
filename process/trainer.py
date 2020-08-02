import time

import numpy as np
import torchvision
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from process.loss import LossBVAE2, LossFunction, LossJoints
from log.logger import Logger

from postprocess.postprocess import postprocess_batch


class Trainer():
    
    def __init__(self, loss_function: LossFunction, device: str, log_interval: int):
        
        self.loss_function = loss_function
        self.device = device
        self.log_interval = log_interval

        self.postprocess = False
        self.accumulate_grad = False
        self.batch_multiplier = 10


    def train(self, logger: Logger, model: nn.Module, optimizer: Optimizer, scheduler: _LRScheduler, dataloader: DataLoader, validation_dataloader: DataLoader, epochs: int, log_name="Train", joint_model=None):

        for epoch in range(epochs):

            model.train()
            # before = list(model.parameters())
            self.loss_function.set_log_name(log_name)
            self.batch_process(logger, train=True, model=model, optimizer=optimizer, scheduler=scheduler, dataloader=dataloader, epochs=1, log_name=log_name, joint_model=joint_model)
            # after = list(model.parameters())
            # for i in range(len(before)):
            #     print(torch.equal(before[i].data, after[i].data))

            if validation_dataloader != None:
                self.loss_function.set_log_name("Validation")
                self.test(logger, model, validation_dataloader, "Validation", joint_model=joint_model)

        model.eval()


    def test(self, logger: Logger, model: nn.Module, dataloader: DataLoader, log_name="Test", joint_model=None):

        model.eval()
        self.loss_function.set_log_name(log_name)
        with torch.no_grad():
            self.batch_process(logger, train=False, model=model, optimizer=None, scheduler=None, dataloader=dataloader, epochs=1, log_name=log_name, joint_model=joint_model)


    def batch_process(self, logger: Logger, train: bool, model: nn.Module, optimizer: Optimizer, scheduler: _LRScheduler, dataloader: DataLoader, epochs: int, log_name: str, joint_model = None):
        
        self.loss_function.dataset_size = len(dataloader.dataset)
        model.to(self.device)

        joint_loss = LossJoints(logger)
        joint_loss.set_log_name(self.loss_function.log_name)

        if self.accumulate_grad:
            loss_value = 0
            count = 0

        for epoch in range(epochs):

            start = time.time()
            loss_sum = 0.0

            for batch_idx, batch_input in enumerate(dataloader):

                joints = None
                if len(batch_input) > 1:
                    joints = batch_input[1].to(self.device)

                batch_input = batch_input[0].to(self.device) # batch_input[1] contains the target data (if this wasn't an autoencoder...)

                if batch_input.dim() == 4: # if rendered dataset; msra native has dim() == 3
                    batch_input = batch_input.squeeze(1)

                if train:
                    if self.accumulate_grad:
                        if count == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            count = self.batch_multiplier
                    else:
                        optimizer.zero_grad()

                if joint_model is not None:
                    joint_model.to(self.device)
                    with torch.no_grad():
                        model.eval()
                        # (z_mu, z_logsigma) = model.encode(batch_input)
                        # z_mu = z_mu.view(batch_input.size(0), model.num_latents)
                        # z_logsigma = z_logsigma.view(batch_input.size(0), model.num_latents)
                        # z = model.reparameterize(z_mu, z_logsigma)

                        batch_output, x_params, z, z_params, _ = model(batch_input)
                        (mu, logsigma) = z_params

                    _, _, _, _, recon_joints = joint_model(z)
                        # (z_mu, z_logsigma) = z_params
                else:
                    batch_output, x_params, z, z_params, recon_joints = model(batch_input)
                    (mu, logsigma) = z_params

                # plot joint-model:
                plot_joint_model = True
                (tmp1, tmp2) = x_params
                if plot_joint_model and tmp1 is not None and z is not None:
                    batch_size = batch_input.size(0)
                    (x_mu, x_logsigma) = model.decode(z)
                    x_mu = x_mu.view(-1, model.input_size * model.input_size)
                    x_logsigma = x_logsigma.view(batch_size, model.input_size * model.input_size)

                    if model.output_dist == 'normal':
                        x = model.reparameterize(x_mu, x_logsigma)
                        x_mu = x_mu.view(batch_size, model.input_size, model.input_size)
                        x_logsigma = x_logsigma.view(batch_size, model.input_size, model.input_size)
                    # elif model.output_dist == 'bernoulli':
                        # x = model.reparameterize_bernoulli(x_mu)
                    elif model.output_dist == 'fake_normal':
                        x_logsigma = torch.zeros_like(x_logsigma)
                        x = model.reparameterize(x_mu, x_logsigma)
                        x_mu = x_mu.view(batch_size, model.input_size, model.input_size)
                        x_logsigma = x_logsigma.view(batch_size, model.input_size, model.input_size)
                    # else:
                        #x = x_params
                        # x = x_mu

                    # x = x.view(batch_size, model.input_size, model.input_size)
                    x_params = (x_mu, x_logsigma)
                    # return x, (x_mu, x_logsigma), z, (z_mu, z_logsigma), None
                
                if recon_joints is not None:
                    loss_value = joint_loss.compute_loss(batch_output, batch_input, mu, logsigma, x_params, z, recon_joints, joints)
                    _ = self.loss_function.loss(batch_output, batch_input, mu, logsigma, x_params, z, recon_joints, joints)
                else:
                    loss_value = self.loss_function.loss(batch_output, batch_input, mu, logsigma, x_params, z, recon_joints, joints)
                if self.accumulate_grad:
                    loss_value /= self.batch_multiplier

                lass_value_item = loss_value.item()
                loss_sum += lass_value_item

                if train:
                    loss_value.backward()
                    if self.accumulate_grad:
                        count -= 1
                    else:
                        optimizer.step()

                if logger:
                    self.log_batch_step(logger, train, batch_idx, log_name, batch_input, batch_output, dataloader, lass_value_item, x_params)

                if train and scheduler:
                    scheduler.step()

            loss_avg = loss_sum / len(dataloader.dataset) # TODO: inconsistency in tutorial code? len(dataloader) or len(dataloader.dataset) => number of batches vs number of frames

            if train:
                if scheduler:
                    #scheduler.step(loss_avg)
                    # scheduler.step()
                    pass

                if logger != None:
                    logger.next_epoch()

            end = time.time()
            print('====> Average {} loss: {:.8f}; Time: {}'.format(log_name, loss_avg, end - start))


    def log_batch_step(self, logger: Logger, train: bool, batch_idx: int, log_name: str, batch_input, batch_output, dataloader, lass_value_item, x_params):
        
        log_step = logger.epoch
        if train:
            log_step = logger.step
            if batch_idx % self.log_interval == 0:
                print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    log_name,
                    logger.epoch, batch_idx * len(batch_input), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), lass_value_item / len(batch_input)))

        if logger.step == 0:
            tmp_input = batch_input[:20].unsqueeze(1) # color channel
            grid_input = torchvision.utils.make_grid(tmp_input, nrow=10, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
            logger.add_image( "{}/Ground-Truth".format(log_name), grid_input.cpu().data.numpy(), log_step) # var.detach().numpy()

        # if batch_idx == 0:
        if batch_idx % self.log_interval == 0:
            # tmp_input = batch_input[:20].unsqueeze(1) # color channel
            
            # plot resampled:
            #tmp_output = batch_output[:20].unsqueeze(1) # color channel

            # plot mu
            (tmp_output, _) = x_params
            if tmp_output is not None:
                tmp_output = tmp_output.view(-1, 128, 128)
                tmp_output = tmp_output[:20].unsqueeze(1) # color channel
                
                # todo: tanh to model
                tanh = nn.Hardtanh(inplace=False)
                tmp_output = tanh(tmp_output)

                # grid_input = torchvision.utils.make_grid(tmp_input, nrow=10, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
                grid_output = torchvision.utils.make_grid(tmp_output, nrow=10, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)

                # logger.add_image( "{}/Ground-Truth".format(log_name), grid_input.cpu().data.numpy(), log_step) # var.detach().numpy()
                logger.add_image( "{}/Reconstructed".format(log_name), grid_output.cpu().data.numpy(), log_step)

                if self.postprocess == True:
                    postprocessed_output = np.expand_dims(postprocess_batch(batch_output[:20].cpu().data.numpy()), 1)
                    grid_postprocessed = torchvision.utils.make_grid(torch.from_numpy(postprocessed_output), nrow=10, padding=2, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=0)
                    logger.add_image( "{}/Postprocessed".format(log_name), grid_postprocessed, log_step)

        if train:
            logger.next_step()
