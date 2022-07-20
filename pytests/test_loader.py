import unittest
import os
import torch

from os.path import split, realpath, join
import sys
# hacky way of importing from repo
base_dir = split(split(realpath(__file__))[0])[0]
sys.path.append(base_dir)

import dataloader



class test_dataloader(unittest.TestCase):
    def setUp(self):
        self.dir = 'dev-data/tactip-127'
        self.task = ['edge_2d', 'shear', 'real']
        self.training_data = dataloader.get_data(self.dir, self.task, store_ram=False)
        self.validation_data = dataloader.get_data(self.dir, self.task, store_ram=False, val=True, labels_range=self.training_data.labels_range)

    def test_loaded_iamge_tensor(self):
        sample = self.training_data[0]
        assert torch.is_tensor(sample['image'])

    def test_loaded_label(self):
        sample = self.training_data[0]
        assert torch.is_tensor(sample['label'])

    def test_can_load_into_ram(self):
        data = dataloader.get_data(self.dir, self.task, store_ram=True)
        data = dataloader.get_data(self.dir, self.task, store_ram=True, val=True, labels_range=self.training_data.labels_range)






if __name__ == '__main__':
    unittest.main()
