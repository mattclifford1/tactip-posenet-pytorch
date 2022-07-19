'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

training of nueral network
'''
import multiprocessing
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

from tqdm import tqdm
import multiprocessing
import dataloader

import matplotlib.pyplot as plt

class trainer():
    def __init__(self, dataset_train,
                       model,
                       batch_size=16,
                       lr=1e-4,
                       input_size=(128,128),
                       decay=1e-6,
                       epochs=100,
                       l2_reg=0.001):
        self.dataset_train = dataset_train
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.input_size = input_size
        self.decay = decay
        self.epochs = epochs
        self.l2_reg = l2_reg
        # misc inits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cores = multiprocessing.cpu_count()
        # get data loader
        self.get_data_loader(prefetch_factor=1)

    def get_data_loader(self, prefetch_factor=1):
        cores = int(self.cores/2)
        self.torch_dataloader_train = DataLoader(self.dataset_train,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=cores,
                                     prefetch_factor=prefetch_factor)

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
        self.running_loss = [0]


    def start(self, val_every=1):
        self.setup()
        # self.get_saver()
        self.val_every = val_every
        self.val_all(0)
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            self.epoch = epoch
            self.running_loss = []
            for step, sample in enumerate(tqdm(self.torch_dataloader_train, desc="Train Steps", leave=False)):
                self.train_step(sample)
                self.scheduler.step()
            if self.epoch%self.val_every == 0:
                self.val_all(self.epoch+1)


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
        print('Epoch ', str(epoch), ': ', np.mean(self.running_loss))


if __name__ == '__main__':
    import networks.tacnet as t_net
    import networks.model_128 as m_128
    from argparse import ArgumentParser
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--dir", default='dev-data/tactip-127/model_surface2d', type=str, help='folder where data is located')
    parser.add_argument("--batch_size",type=int,  default=64, help='batch size to load and train on')
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs to train for')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    ARGS = parser.parse_args()
    training_data = dataloader.get_data(ARGS.dir)

    # model = t_net.network((128, 128))
    model = m_128.network()
    model.apply(t_net.weights_init_normal)

    t = trainer(training_data, model,
                batch_size=ARGS.batch_size,
                epochs=ARGS.epochs)
    t.start()
