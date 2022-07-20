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
import networks.model_128 as m_128

# tests
# ====================================================


class test_trainer(unittest.TestCase):
    def setUp(self):
        self.dir = 'dev-data/tactip-127'
        self.task = ['edge_2d', 'shear', 'real']
        self.training_data = dataloader.get_data(self.dir, self.task, store_ram=False)
        self.validation_data = dataloader.get_data(self.dir, self.task, store_ram=False, val=True, labels_range=self.training_data.labels_range)
        self.model = m_128.network(final_size=2)
        self.model.apply(t_net.weights_init_normal)

    def test_can_start(self):
        t = trainer.trainer(self.training_data,
                    self.validation_data,
                    self.model,
                    batch_size=16,
                    epochs=0,
                    save_dir=os.path.join(self.dir, 'models', 'pose_estimation'))
        t.start()






if __name__ == '__main__':
    unittest.main()
