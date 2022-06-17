import unittest
import os
import torch

from os.path import split, realpath, join
import sys
# hacky way of importing from repo
base_dir = split(split(realpath(__file__))[0])[0]
sys.path.append(base_dir)

import networks.tacnet as t_net

# tests
# ====================================================


class test_tacnet(unittest.TestCase):
    def test_diff_size_net_inputs(self):
        self.make_and_run_square_net(128, 128)
        self.make_and_run_square_net(256, 256)
        # self.make_and_run_square_net(512)

    def test_input_size_wrong(self):
        self.make_and_run_square_net(64, 128)
        self.make_and_run_square_net(256, 128)
        self.make_and_run_square_net(1024, 256)

    def make_and_run_square_net(self, size_input, size_net):
        x = torch.zeros(1,1,size_input,size_input)
        net = t_net.network(size_net)
        out = net.forward(x)






if __name__ == '__main__':
    unittest.main()
