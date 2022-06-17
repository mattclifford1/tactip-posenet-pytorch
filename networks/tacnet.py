'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

Nathan Lepora's regression network for the tactip pose estimation
'''
import torch
import torch.nn as nn
from torchvision import transforms
import os
import errno


class network(nn.Module):
    def __init__(self, input_size):
        super(network, self).__init__()
        self.input_size = (1, input_size[0], input_size[1])

        self.conv_size = 256
        self.kernel_size = 3
        self.num_conv_layers = 5
        self.fc_layer_nums = [64, 2]
        self.contruct_layers()

    def contruct_layers(self):
        # CONVOLUTIONS
        self.conv_layers = {}
        dim = 1 # input is grey scale
        for layer in range(self.num_conv_layers):
            conv_layer = ConvBlock(dim, self.conv_size,
                                   kernel_size=self.kernel_size,
                                   # activation=nn.PReLU
                                   )
            self.conv_layers['conv_'+str(layer)] = conv_layer
            dim = self.conv_size

        # FULLY CONNECTED
        self.fc_layers = {}
        prev_num = self.get_out_conv_shape()
        for layer in range(len(self.fc_layer_nums)):
            fc_layer = FullyConnectedLayer(prev_num, self.fc_layer_nums[layer])
            self.fc_layers['fc_'+str(layer)] = fc_layer
            prev_num = self.fc_layer_nums[layer]


    def get_out_conv_shape(self):
        '''
        pass a dummy input of the correct size and determine size after all the convs
        '''
        dummy_input = torch.zeros(1, self.input_size[0],
                                     self.input_size[1],
                                     self.input_size[2])
        _shape = self.forward_conv_layers(dummy_input).shape
        num_dims = _shape[1]*_shape[2]*_shape[3]
        return num_dims

    def forward_conv_layers(self, x):
        '''
        separation of conv layer forward pass to be able to calculate size after them
        which is required when constructing the fully connected layers
        '''
        for layer in range(self.num_conv_layers):
            x = self.conv_layers['conv_'+str(layer)](x)
        return x


    def forward(self, x):
        '''
        todo: split into forward convs and fc to be able to deterimine
              the dims after conv layer
              '''
        x = self.check_input_size(x)
        x = self.forward_conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        for layer in range(len(self.fc_layer_nums)):
            x = self.fc_layers['fc_'+str(layer)](x)
        return x

    def check_input_size(self, x):
        if isinstance(self.input_size, int):
            H = self.input_size
            W = self.input_size
        elif len(self.input_size) == 2:
            H = self.input_size[0]
            W = self.input_size[1]
        else:
            H = self.input_size[1]
            W = self.input_size[2]
        if x.shape[2] != H and x.shape[3] != W:
            x = transforms.Resize((H, W))(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size,
                 batch_norm=False,
                 activation=nn.ReLU,
                 dropout=0,  # zero is equivelant to identity (no dropout)
                 **kwargs):
        super(ConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        if self.batch_norm:
            x = self.bn(x)
        return x


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_num, out_num,
                 batch_norm=False,
                 activation=nn.ReLU,
                 dropout=0,  # zero is equivelant to identity (no dropout
                 **kwargs):
        super(FullyConnectedLayer, self).__init__()
        self.batch_norm = batch_norm
        self.fc = nn.Linear(in_num, out_num)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.batch_norm:
            x = self.bn(x)
        return x


def load_weights(model, weights_path):
    if not os.path.isfile(weights_path):
        # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path)
        raise ValueError("Couldn't find network weights path: "+str(weights_path)+"\nMaybe you need to train first?")
    model.load_state_dict(torch.load(weights_path))


if __name__ == '__main__':
    # check network works (dev mode)
    x = torch.zeros(1,1,128,128)
    net = network((128, 128))
    out = net(x)
    print('in shape: ', x.shape)
    print('out shape: ', out.shape)
