'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

pytorch data loader to read images and csv
'''
import pandas as pd
import os
import numpy as np
from skimage import io, transform, color
from torchvision import transforms
import torch
import random

'''
Classes that transform the image data
Designed to be used in a modular fashion
'''
class rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple): Desired output size. Output is
            matched to output_size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        new_h, new_w = self.output_size
        self.new_h, self.new_w = int(new_h), int(new_w)

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = transform.resize(sample[key], (self.new_h, self.new_w))
        return sample

'''
data object to be used with pytorch's dataloader
'''
class get_data:
    def __init__(self,
                 base_dir,
                 transform=rescale((128,128)),
                 val=False,
                 store_ram=False,
                 labels_range={}):
        self.base_dir = base_dir
        self.transform = transform
        self.val = val
        self.split_type = 'csv_val' if val else 'csv_train'
        self.labels_range = labels_range
        self.read_data()
        self.store_ram = store_ram
        if self.store_ram == True:
            self.load_images_to_ram()

    def read_data(self):
        dir = 'test' if self.val else 'train'   # from nathans data repo
        self.csv = os.path.join(self.base_dir, dir, 'targets.csv')
        self.image_dir = os.path.join(self.base_dir, 'frames_bw')
        self.x_name = 'image_name'
        self.y_names = ['pose_2', 'pose_6']
        self.task = 'surface_3d' # change this
        self.data_type = 'nathan_'
        if not os.path.isfile(self.csv):
            dir = 'csv_val' if self.val else 'csv_train'   # from sim2real data
            self.csv = os.path.join(self.base_dir, dir, 'targets.csv')
            self.image_dir = os.path.join(self.base_dir, dir, 'images')
            self.x_name = 'sensor_image'
            self.y_names = ['pose_3', 'pose_4']
            self.data_type = 'sim_'
        self.df = pd.read_csv(self.csv)
        self.image_paths = self.df[self.x_name].tolist()

        self.labels = {}
        for label in self.y_names:
            # get min and max to normalise data
            if self.val == False:   # calc how to normalise labels is using training set
                self.labels_range[label] = [round(self.df[label].min()), round(self.df[label].max())]
            # interp to put into range (-1, 1)
            self.labels[label] = np.interp(self.df[label].tolist(), self.labels_range[label], (-1,1))

    def load_images_to_ram(self):
        print('Loading all images into RAM')
        self.images_in_ram = []
        for image_path in self.image_paths:
            self.images_in_ram.append(io.imread(os.path.join(self.image_dir, image_path)))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        # get image
        if self.store_ram == True:  # get pre loaded from ram
            image = self.images_in_ram[i]
        else:
            image_path = self.image_paths[i]
            image = io.imread(os.path.join(self.image_dir, image_path))
        sample = {'image': image}
        # data transforms
        if self.transform:
            sample = self.transform(sample)
        sample = numpy_image_torch_tensor(sample)
        # get labels to sample
        sample['label'] = []
        sample['label_names'] = []
        for label in self.labels.keys():
            sample['label'].append(self.labels[label][i])
            sample['label_names'].append(label)
        sample['label'] = torch.tensor(sample['label'])
        return sample


def numpy_image_torch_tensor(sample):
    """Convert ndarrays in sample to Tensors."""
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W
    for key in sample.keys():
        # add chanel dim to 2D arrays
        if len(sample[key].shape) == 2:
            sample[key] = np.expand_dims(sample[key], axis=2)
        sample[key] = torch.from_numpy(sample[key].transpose((2, 0, 1)))
    return sample



class grey_scale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple): Desired output size. Output is
            matched to output_size.
    """
    def __init__(self, normalise=True):
        self.normalise = normalise

    def make_grey(self, image):
        if self.normalise:
            image = image/255.
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        elif image.shape[2] == 1:
            return image
        else:
            image = color.rgb2gray(image)
        return image

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = self.make_grey(sample[key])
        return sample


if __name__ == '__main__':
    # plot a few of the training examples
    import matplotlib.pyplot as plt
    from argparse import ArgumentParser
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--csv", default='dev-data/tactip-127/model_surface2d/targets.csv', type=str, help='targets.csv file')
    parser.add_argument("--image_dir", default='dev-data/tactip-127/model_surface2d/frames_bw', type=str, help='folder where images are located')
    ARGS = parser.parse_args()

    # composed = transforms.Compose([Rescale((256,256)),
    #                                RandomCrop(224)])
    training_data = get_data(ARGS.csv,
                             ARGS.image_dir,
                             transform=grey_scale())
    # loop over the data set handler to check it's working correctly
    fig = plt.figure()
    for i in range(len(training_data)):
        sample = training_data[i]
        im_numpy = sample['image'].cpu().detach().numpy()
        im_numpy = np.swapaxes(im_numpy,0,1)
        im_numpy = np.swapaxes(im_numpy,1,2)
        plt.imshow(im_numpy)
        print('Max im value: ', sample['image'].max())
        print('Labels: ', sample['label'])
        print('Label names: ', sample['label_names'])
        plt.show()
        if i == 0:
            break
