import torch
import torch.nn as nn
from torchvision import transforms
import os
import errno


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.input_size = 128

        self.conv_size = 256
        self.kernel_size = 3
        self.num_conv_layers = 5
        self.contruct_net()

    def contruct_net(self):
        self.conv_layers = {}
        dim = 1 # input is grey scale
        for layer in range(self.num_conv_layers):
            conv_layer = nn.Sequential(
                nn.Conv2d(dim, self.conv_size, self.kernel_size),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),
                # nn.Dropout(0.3)
                )
            self.conv_layers['conv_'+str(layer)] = conv_layer
            dim = self.conv_size



        self.fc = nn.Sequential(
            nn.Linear(self.final_conv_dims, self.emb_dim*2),
            # nn.Linear(64*self.conv_final_dim*self.conv_final_dim, self.emb_dim*2),
            nn.PReLU(),
            nn.Linear(self.emb_dim*2, self.emb_dim)
        )

    def forward(self, x):
        x = self.check_input_size(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(-1, self.final_conv_dims)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

    def check_input_size(self, x):
        if isinstance(self.input_size, int):
            H = self.input_size
            W = self.input_size
        else:
            H = self.input_size[0]
            W = self.input_size[1]
        if x.shape[2] != H and x.shape[3] != W:
            x = transforms.resuze(x, (H, W))
        return x

def load_weights(model, weights_path):
    if not os.path.isfile(weights_path):
        # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path)
        raise ValueError("Couldn't find network weights path: "+str(weights_path)+"\nMaybe you need to train first?")
    model.load_state_dict(torch.load(weights_path))


if __name__ == '__main__':
    # check network works (dev)
