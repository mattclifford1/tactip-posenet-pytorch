'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

training of tactip pose estimation nueral network
'''
import multiprocessing
import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import sys; sys.path.append('..'); sys.path.append('.')
from data import dataloader


if __name__ == '__main__':
    import networks.tacnet as t_net
    import networks.model_128 as m_128
    from argparse import ArgumentParser
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--dir", default='dev-data/tactip-127', type=str, help='folder where data is located')
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs to train for')
    parser.add_argument("--task", type=str, nargs='+', default=['edge_2d', 'shear', 'real'], help='dataset to train on')
    parser.add_argument("--batch_size",type=int,  default=16, help='batch size to load and train on')
    parser.add_argument("--cores",type=int,  default=multiprocessing.cpu_count(), help='number of cpu cores to use')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    ARGS = parser.parse_args()
    training_data = dataloader.get_data(ARGS.dir, ARGS.task, store_ram=ARGS.ram)
    torch_dataloader_train = DataLoader(training_data,
                                 batch_size=ARGS.batch_size,
                                 shuffle=False,
                                 num_workers=ARGS.cores,
                                 prefetch_factor=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for step, sample in enumerate(tqdm(torch_dataloader_train, desc="Train Steps", leave=True)):
        im = sample['image'].to(device=device, dtype=torch.float)
        label = sample['label'].to(device=device, dtype=torch.float)
