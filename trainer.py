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
    def __init__(self, dataset, batch_size=64):
        self.dataset = dataset
        self.batch_size = batch_size
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

    def start(self):

        for step, sample in enumerate(tqdm(self.torch_dataloader, desc="Train Steps", leave=False)):
            continue




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--csv", default='dev-data/tactip-127/model_surface2d/targets.csv', type=str, help='targets.csv file')
    parser.add_argument("--image_dir", default='dev-data/tactip-127/model_surface2d/frames_bw', type=str, help='folder where images are located')
    ARGS = parser.parse_args()
    training_data = dataloader.get_data(ARGS.csv,
                                        ARGS.image_dir)
    t = trainer(training_data)
    t.start()
