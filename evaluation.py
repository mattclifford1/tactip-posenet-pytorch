'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

evaluation of tactip pose estimation nueral network
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
import json
import utils


def load_weights(model, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(weights_path):
        # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path)
        raise ValueError("Couldn't find network weights path: "+str(weights_path)+"\nMaybe you need to train first?")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))


class evaller:
    def __init__(self, dir='..',
                       task=('surface_3d', 'shear'),
                       train_name='sim_LR:0.0001_BS:16',
                       run=0,
                       model='best_model.pth',
                       store_ram=False,
                       batch_size=4):
        self.dir = dir    # base dir where all models are held
        self.task = task  # task to load eg. ['surface_3d', 'shear']
        self.train_name = train_name
        self.run = 'run_' + str(run)
        self.model_dir = os.path.join(self.dir, 'models', 'pose_estimation', self.task[0], self.task[1], self.train_name, self.run, 'checkpoints')
        self.model_name = model
        self.store_ram = store_ram
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_pretrained_model()
        self.get_val_data()

    def get_val_data(self):
        validation_data = dataloader.get_data(self.dir, self.task, store_ram=self.store_ram, val=True, labels_range=self.normalisation)
        self.data_loader = DataLoader(validation_data, self.batch_size)


    def get_pretrained_model(self):
        self.model = m_128.network(final_size=int(self.task[0][-2]))
        model_file = os.path.join(self.model_dir, self.model_name)
        load_weights(self.model, model_file)
        normalisation_file = os.path.join(self.model_dir, 'output_normilisation.json')
        with open(normalisation_file) as f:
            self.normalisation = json.load(f)

    def predict(self, ims):
        # ims should be a torch tensor on the correct device
        preds = self.model(ims)

    def _get_MAE_batch(self, ims, labels):
        preds = self.model(ims)
        mae = torch.abs(preds - labels).mean()
        mae = mae.cpu().detach().numpy()
        return mae

    def get_MAE(self, real2sim_model=None):
        MAEs = []
        for step, sample in enumerate(tqdm(self.data_loader, desc="Downstream Val Steps", leave=False)):
            if step == 1:
                break
            # get val batch sample
            im = sample['image'].to(device=self.device, dtype=torch.float)
            label = sample['label'].to(device=self.device, dtype=torch.float)
            if real2sim_model is not None:
                # shift domain from real to simulation
                im = real2sim_model(im)
            MAEs.append(self._get_MAE_batch(im, label))
        return sum(MAEs) / len(MAEs)




if __name__ == '__main__':
    import networks.tacnet as t_net
    import networks.model_128 as m_128
    from argparse import ArgumentParser
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--dir", default='..', type=str, help='folder where data is located')
    parser.add_argument("--task", type=str, nargs='+', default=['surface_3d', 'shear', 'sim'], help='dataset to train on')
    parser.add_argument("--batch_size",type=int,  default=16, help='batch size to load and train on')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    ARGS = parser.parse_args()

    e = evaller(ARGS.dir, task=ARGS.task)

    mae =e.get_MAE()
    print(mae)
