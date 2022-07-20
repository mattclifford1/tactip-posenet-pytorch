import unittest
import os
import torch

from os.path import split, realpath, join
import sys
# hacky way of importing from repo
base_dir = split(split(realpath(__file__))[0])[0]
sys.path.append(base_dir)

import networks.tacnet as t_net
import networks.model_128 as m_128

# tests
# ====================================================


class test_tacnet(unittest.TestCase):
    def test_diff_size_net_inputs(self):
        self.make_and_run_square_net(128, (128, 128))
        self.make_and_run_square_net(256, (256, 256))
        # self.make_and_run_square_net(512)

    def test_input_size_wrong(self):
        self.make_and_run_square_net(64, (128, 128))
        self.make_and_run_square_net(256, (128, 128))
        self.make_and_run_square_net(1024, (256, 256))

    def make_and_run_square_net(self, size_input, size_net):
        x = torch.zeros(1,1,size_input,size_input)
        net = t_net.network(size_net)
        out = net(x)

    def test_can_init_weights(self):
        model = m_128.network(final_size=2)
        model.apply(t_net.weights_init_normal)

    def test_can_diff_output_size(self):
        model = m_128.network(final_size=2)
        model = m_128.network(final_size=3)






if __name__ == '__main__':
    unittest.main()
