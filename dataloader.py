'''
pytorch data loader to read images and csv
'''
import pandas as pd
import os
import numpy as np
from skimage import io
import torch
import random


'''
data object to be used with pytorch's dataloader
'''
class get_data:
    def __init__(self,
                 csv,
                 image_dir,
                 transform=None,
                 x_name='image_name',
                 y_names=['pose_2', 'pose_6']):
        self.csv = csv
        self.image_dir = image_dir
        self.transform = transform
        self.x_name = x_name
        self.y_names = y_names
        self.read_data()

    def read_data(self):
        self.df = pd.read_csv(self.csv)
        self.image_paths = self.df[self.x_name].tolist()
        self.labels = {}
        for label in self.y_names:
            self.labels[label] = self.df[label].tolist()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        # get image
        image_path = self.image_paths[i]
        image = io.imread(os.path.join(self.image_dir, image_path))
        sample = {'image': image}
        # data transforms
        if self.transform:
            sample = self.transform(sample)
        # sample = ToTensor(sample)
        # get labels to sample
        sample['label'] = []
        sample['label_names'] = []
        for label in self.labels.keys():
            sample['label'].append(self.labels[label][i])
            sample['label_names'].append(label)
        return sample


def ToTensor(sample):
    """Convert ndarrays in sample to Tensors."""
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W
    for key in sample.keys():
        sample[key] = torch.from_numpy(sample[key].transpose((2, 0, 1)))
    return sample


if __name__ == '__main__':
    # plot a few of the training examples
    import matplotlib.pyplot as plt
    from argparse import ArgumentParser
    home_dir = os.path.expanduser('~')
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--csv", default=os.path.join(home_dir, 'summer-project/nathans-repos/data/dev-data/tactip-127/model_surface2d/targets.csv'), type=str, help='targets.csv file')
    parser.add_argument("--image_dir", default=os.path.join(home_dir, 'summer-project/nathans-repos/data/dev-data/tactip-127/model_surface2d/frames_bw'), type=str, help='folder where images are located')
    ARGS = parser.parse_args()

    training_data = get_data(ARGS.csv, ARGS.image_dir)
    fig = plt.figure()

    for i in range(len(training_data)):
        sample = training_data[i]
        plt.imshow(sample['image'])
        print(sample['label'])
        print(sample['label_names'])
        plt.show()
        if i == 1:
            break
