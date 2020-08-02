import os
from typing import Any
from pathlib import Path

import torch
from torch import save, load
from torch import nn
from torch import optim

from process.trainer import Trainer
from data.dataloader import create_dataloader
from log.logger import Logger
from models.models import VAE


class Solver():

    def __init__(
        self,
        model: VAE,
        loss_function: Any,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: str,
        logger: Logger,
        joint_model = None
        ):

        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainer = Trainer(loss_function, device, logger.log_interval)

        self.joint_model = joint_model

        self.model.eval()

    def save_checkpoint(self, filename: Path):
        print("saving checkpoint: {}".format(filename))
        state = {
            'epoch': self.logger.epoch,
            'step': self.logger.step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
        save(state, filename)


    def load_checkpoint(self, filename: Path):
        print("Loading checkpoint: {}".format(filename))
        if not os.path.isfile(filename):
            print("Cannot load {}".format(filename))
            return
        checkpoint = load(filename)
        self.logger.epoch = checkpoint['epoch']
        self.logger.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()


    def train(self, train_dir: Path, validation_dir: Path, batch_size: int, epochs: int, checkpoint_interval: int, model_dir: Path, overwrite: bool, normalize_input=True, max_persons=-1, shuffle=True, validation_batch_size=None):
        
        print("Training: {}".format(train_dir))
        print("Validation: {}".format(validation_dir))
        print("batch size: {}".format(batch_size))
        print("epochs: {}".format(epochs))

        if validation_batch_size is None:
            validation_batch_size = batch_size
        
        dataloader = create_dataloader(train_dir, batch_size, normalize_input=normalize_input, max_persons=max_persons, shuffle=shuffle)
        
        validation_dataloader = None
        if validation_dir is not None and os.path.exists(validation_dir):
            validation_dataloader = create_dataloader(validation_dir, batch_size, normalize_input=normalize_input, max_persons=max_persons, shuffle=shuffle)
        
        if checkpoint_interval > 0:
            epoch_counter = self.logger.epoch
            while epoch_counter < epochs:
                epoch_counter += checkpoint_interval
                self.trainer.train(self.logger, self.model, self.optimizer, self.scheduler, dataloader, validation_dataloader, checkpoint_interval, joint_model=self.joint_model)
                
                filename = "{}_{}.pt".format(os.path.join(model_dir, self.logger.basename), self.logger.epoch)
                if overwrite or not os.path.isfile(filename):
                    self.save_checkpoint(filename)
        else:
            self.trainer.train(self.logger, self.model, self.optimizer, self.scheduler, dataloader, validation_dataloader, epochs, joint_model=self.joint_model)


    def test(self, test_dir: Path, batch_size: int, normalize_input=True, shuffle=True):
        
        print("Testing: {}".format(test_dir))
        print("batch size: {}".format(batch_size))

        dataloader = create_dataloader(test_dir, batch_size, normalize_input=normalize_input, shuffle=shuffle)
        
        self.trainer.test(self.logger, self.model, dataloader, joint_model=self.joint_model)
