'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

training of tactip pose estimation nueral network
'''
import multiprocessing
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

from tqdm import tqdm
import multiprocessing
from data import dataloader
import utils

import matplotlib.pyplot as plt

class trainer():
    def __init__(self, dataset_train,
                       dataset_val,
                       model,
                       batch_size=16,
                       lr=1e-4,
                       input_size=(128,128),
                       decay=1e-6,
                       epochs=100,
                       l2_reg=0.001,
                       save_dir=None):
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.input_size = input_size
        self.decay = decay
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.save_dir = save_dir
        # misc inits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cores = multiprocessing.cpu_count()
        # get data loader
        self.get_data_loader(prefetch_factor=1)
        self.get_saver()

    def get_data_loader(self, prefetch_factor=1):
        cores = int(self.cores/2)
        self.torch_dataloader_train = DataLoader(self.dataset_train,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=cores,
                                     prefetch_factor=prefetch_factor)
        self.torch_dataloader_val = DataLoader(self.dataset_val,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=max(1, int(cores/4)),
                                     prefetch_factor=prefetch_factor)

    def get_saver(self):
        self.saver = utils.train_saver(self.save_dir,
                                 self.model,
                                 self.lr,
                                 self.batch_size,
                                 self.dataset_train.data_task,  # eg. (edge_2d, tap)
                                 self.dataset_train.data_type,  # eg. sim_ or real_ or nathan_
                                 self.dataset_train.labels_range) # normalisation of outputs to [-1,1]

    def setup(self):
        # optimser
        self.optimiser = optim.Adam(self.model.parameters(), self.lr, weight_decay=self.l2_reg)
        lambda_lr = lambda iterations: (1. / (1. + self.decay * iterations))
        self.scheduler = LambdaLR(self.optimiser, lr_lambda=lambda_lr)
        # self.scheduler = ExponentialLR(self.optimiser, gamma=self.lr_decay)
        # loss criterion for training signal
        self.loss = nn.MSELoss()
        # set up model for training
        self.model = self.model.to(self.device)
        self.model.train()
        self.running_loss = [1]
        self.current_training_loss = 1


    def start(self, val_every=1):
        self.setup()
        # self.get_saver()
        self.val_every = val_every
        self.val_all(0)
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            self.epoch = epoch
            self.running_loss = []
            for step, sample in enumerate(tqdm(self.torch_dataloader_train, desc="Train Steps", leave=False)):
                # if step == 2:
                #     break
                self.train_step(sample)
                self.scheduler.step()
            if self.epoch%self.val_every == 0:
                self.val_all(self.epoch+1)
        # # training finished
        self.saver.save_model(self.model, 'final_model')


    def train_step(self, sample):
        # get training batch sample
        im = sample['image'].to(device=self.device, dtype=torch.float)
        label = sample['label'].to(device=self.device, dtype=torch.float)
        # zero the parameter gradients
        self.optimiser.zero_grad()
        # forward
        pred = self.model(im)
        # loss
        loss = self.loss(pred, label)
        # print('pred:', pred)
        # print('label:', label)
        # print('loss:', loss)
        # backward pass
        loss.backward()
        self.optimiser.step()
        t_loss = loss.cpu().detach().numpy()
        # print(t_loss)
        self.running_loss.append(loss.cpu().detach().numpy()) # save the loss stats


    def val_all(self, epoch):
        self.previous_training_loss = self.current_training_loss
        self.current_training_loss = np.mean(self.running_loss)
        self.model.eval()
        MAEs = []
        for step, sample in enumerate(tqdm(self.torch_dataloader_val, desc="Val Steps", leave=False)):
            # if step == 2:
            #     break
            # get val batch sample
            im = sample['image'].to(device=self.device, dtype=torch.float)
            label = sample['label'].to(device=self.device, dtype=torch.float)
            pred = self.model(im)
            mae = torch.abs(pred - label).mean()
            MAEs.append(mae.cpu().detach().numpy())
        self.model.train()
        self.MAE = sum(MAEs) / len(MAEs)
        print('\nEpoch ', str(epoch), ' training loss: ', np.mean(self.running_loss))
        print('Epoch ', str(epoch), ' validation MAE:', self.MAE)

        stats = {'epoch': [epoch],
                 'mean training loss': [np.mean(self.running_loss)],
                 'val MAE': [self.MAE]}
        self.saver.log_training_stats(stats)
        self.maybe_save_model()

    def maybe_save_model(self):
        if self.previous_training_loss > self.current_training_loss:
            print('saving model')
            self.saver.save_model(self.model, 'best_model')


if __name__ == '__main__':
    import networks.tacnet as t_net
    import networks.model_128 as m_128
    from argparse import ArgumentParser
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--dir", default='dev-data/tactip-127', type=str, help='folder where data is located')
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs to train for')
    parser.add_argument("--task", type=str, nargs='+', default=['edge_2d', 'shear', 'real'], help='dataset to train on')
    parser.add_argument("--batch_size",type=int,  default=16, help='batch size to load and train on')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    ARGS = parser.parse_args()
    training_data = dataloader.get_data(ARGS.dir, ARGS.task, store_ram=ARGS.ram)
    validation_data = dataloader.get_data(ARGS.dir, ARGS.task, store_ram=ARGS.ram, val=True, labels_range=training_data.labels_range)

    # model = t_net.network((128, 128))
    model = m_128.network(final_size=int(ARGS.task[0][-2]), task=ARGS.task[0])
    model.apply(t_net.weights_init_normal)

    t = trainer(training_data,
                validation_data,
                model,
                batch_size=ARGS.batch_size,
                epochs=ARGS.epochs,
                save_dir=os.path.join(ARGS.dir, 'models', 'pose_estimation'))
    t.start()
