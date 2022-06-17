'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

training of nueral network
'''
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
import multiprocessing
import dataloader

class trainer():
    def __init__(self, dataset, model,
                       batch_size=64,
                       lr=1e-4,
                       input_size=(128,128),
                       decay=1e-6,
                       epochs=100):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.input_size = input_size
        self.decay = decay
        self.epochs = epochs
        # misc inits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cores = multiprocessing.cpu_count()
        # get data loader
        self.get_data_loader(prefetch_factor=2)

    def get_data_loader(self, prefetch_factor=1):
        cores = int(self.cores/2)
        self.torch_dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=cores,
                                     prefetch_factor=prefetch_factor)

    def setup(self):
        # optimser
        self.optimiser = optim.Adam(self.model.parameters(), self.lr)
        # self.scheduler = ExponentialLR(self.optimiser, gamma=self.lr_decay)
        # loss criterion for training signal
        self.loss = nn.MSELoss()
        # set up model for training
        self.model = self.model.to(self.device)
        self.model.train()


    def start(self):
        self.setup()
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            for step, sample in enumerate(tqdm(self.torch_dataloader, desc="Train Steps", leave=False)):
                continue


    def train_step(self, sample):
        # get training batch sample
        im = sample['image'].to(device=self.device, dtype=torch.float)
        # zero the parameter gradients
        self.optimiser.zero_grad()
        # forward
        pred = self.model(im)
        # loss
        loss = self.loss(im, sample['label'])
        # backward pass
        loss.backward()
        self.optimiser.step()
        # self.running_loss.append(loss.cpu().detach().numpy()) # save the loss stats




if __name__ == '__main__':
    import networks.tacnet as t_net
    from argparse import ArgumentParser
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--csv", default='dev-data/tactip-127/model_surface2d/targets.csv', type=str, help='targets.csv file')
    parser.add_argument("--image_dir", default='dev-data/tactip-127/model_surface2d/frames_bw', type=str, help='folder where images are located')
    ARGS = parser.parse_args()
    training_data = dataloader.get_data(ARGS.csv,
                                        ARGS.image_dir)

    model = t_net.network((128, 128))
    print(model.parameters())
    t = trainer(training_data, model, epochs=1)
    t.start()
