import unittest
import os
import torch

from os.path import split, realpath, join
import sys
# hacky way of importing from repo
base_dir = split(split(realpath(__file__))[0])[0]
sys.path.append(base_dir)

import trainer, dataloader
import networks.tacnet as t_net

# tests
# ====================================================


class test_trainer(unittest.TestCase):
    def setUp(self):
        self.csv = join(base_dir, 'dev-data/tactip-127/model_surface2d/targets.csv')
        self.image_dir = join(base_dir, 'dev-data/tactip-127/model_surface2d/frames_bw')
        self.dataloader = dataloader.get_data(self.csv, self.image_dir)
        self.model = t_net.network((128, 128))

    def test_can_start(self):
        t = trainer.trainer(self.dataloader, self.model, epochs=0)
        t.start()






if __name__ == '__main__':
    unittest.main()
