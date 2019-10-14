#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:07:56 2019

@author: xie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:49:01 2019

@author: xie
"""
import time
import os.path
import torchvision.transforms as transforms
from scipy.io import loadmat, savemat
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from deconv1d_dataset import Pixel4SER  
from util.visualizer import Visualizer
from util.visualizer import save_images
from models import create_model
from util.util import CustomDatasetDataLoader



if __name__ == '__main__':    
    batch_size = int(20000)
    net_name = 'encoderdecoderlongfcn1dM'
    visualizer = Visualizer(net_name)
    data_loader = CustomDatasetDataLoader('./data/test/', batch_size)
    dataset = data_loader.load_data()
    dataset = dataset
    dataset_size = len(dataset)    # get the number of images in the dataset.
    
    model = create_model(net_name, 1, 1, 0.001, isTrain = False)
    model.setup(isTrain = False, load_iter = 2820)


    for i, data in enumerate(dataset):          
        model.set_input(data)
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results


        img_path = model.get_image_paths()     # get image paths
#        if i % 1000 == 0:  # save images to an HTML file
        print('processing (%04d)-th image... ' % (i))

        save_images( visuals, net_name, img_path)
    

    

    












