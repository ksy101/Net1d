#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from scipy.io import loadmat, savemat
from deconv1d_dataset import Pixel4SER

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, path=[], batch_size= int(1), max_dataset_size = float("inf")):   
        
        dataset_train = Pixel4SER(path, max_dataset_size)   
        self.max_dataset_size = max_dataset_size;
        self.dataset = dataset_train
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size = self.batch_size,
            shuffle = True,
            num_workers=int(10))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
#        print(len(self.dataset))
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
#            print(i)
            if i * self.batch_size >= self.max_dataset_size:
                break
            yield data



def tensor2im(input_image, imtype=np.uint8, image_id = int(0)):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
#            print('alor')
        else:
            return input_image
#        print((image_tensor))
            
        image_numpy = input_image[image_id].cpu().detach().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (0,1))
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

#    return image_numpy.astype(imtype)
    return image_numpy


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
#    image_pil = Image.fromarray(image_numpy)
#    image_pil.save(image_path)
    savemat(image_path+'.mat',{'Mu_l2h': image_numpy})
    


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
