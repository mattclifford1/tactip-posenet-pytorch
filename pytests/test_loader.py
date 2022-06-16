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
        self.csv = join(base_dir, 'dev-data/tactip-127/model_surface2d/targets.csv')
        self.image_dir = join(base_dir, 'dev-data/tactip-127/model_surface2d/frames_bw')
        self.dataloader = dataloader.get_data(self.csv, self.image_dir)

    # def test_dataloader_bad_args(self):
    #     csv = 'bad.xx'
    #     image_dir = 'bad.yy'
    #     self.failUnlessRaises(ValueError, dataloader.get_data, csv, image_dir)

    def test_loaded_iamge_tensor(self):
        sample = self.dataloader[0]
        assert torch.is_tensor(sample['image'])

    def test_loaded_label(self):
        sample = self.dataloader[0]
        assert torch.is_tensor(sample['label'])





if __name__ == '__main__':
    unittest.main()
