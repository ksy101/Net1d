#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:46:04 2019

@author: xie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:52:40 2019

@author: xie
"""

import os.path
import random
import torchvision.transforms as transforms
#from PIL import Image
from scipy.io import loadmat
import numpy as np
import skimage
from torch.utils.data.dataset import Dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.mat', '.MAT',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]



class Pixel4SER(Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dir_AB, max_dataset_size):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.dir_AB = dir_AB  # get the image directory
#        self.AB_paths = sorted(make_dataset(self.dir_AB))  # get image paths
        self.AB_paths = sorted(make_dataset(self.dir_AB, max_dataset_size))
        self.image_name = [x.strip() for x in self.AB_paths]
        self.input_nc = 11
        self.output_nc = 50
#        self.dir_M = os.path.join(opt.dataroot,'M')
#        self.M_paths = sorted(make_dataset(self.dir_M, opt.max_dataset_size))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
#        AB = Image.open(AB_path).convert('RGB')
        Mu = loadmat(AB_path)
        AB = Mu['Mu']
#        print('type of AB %s' % dir(AB))
#        w,h = AB.shape
#        print('w %d, h %d, l %d' %(w, h, l))
#        h2 = int(h / 2)    
#        print('midstep-h2 %d' % h2)
        A = AB[:,:self.input_nc] # same dimension setting as in MATLAB, i.e. the 3rd dimension is channels: first 5 bins for Mu_l; the others 50 bins for Mu_h
        B = AB[:,self.input_nc:(self.input_nc + self.output_nc)]
        M = AB[:,(self.input_nc + self.output_nc):]
        M[:, 2:] = M[:, 2:] * 1e3
        w, h = M.shape   # 40000, 4
        
        """
        material exits or not
        """
        M_codes = np.zeros([w, h])
        for l in range(w):
            for m in range(h):
                if M[l, m] != 0:
                    if m ==2:
                        M_codes[l, 0]= 1
                        M_codes[l, m]= 1
                    elif m ==3:
                        M_codes[l, 0]= 1
                        M_codes[l, m]= 1     
                    elif m ==1:
                        M_codes[l, m]= 1   


#        M_codes_gt = np.array([0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,1.18])
#        M_codes = np.zeros([h, 20])                     
#                if M[l, m] in  M_codes_gt:
#                    idm = np.where(M_codes_gt == M[l, m])
#                    M_codes[m, idm[0]]= 1
                    
        """Prcess of data:
            normalize M for different materials
            add noise to input image A
        """
#        M = NormalizeM(M)
#        A = NormalizeM(A)
#        B = NormalizeM(B)
        
        
#        M_id = np.zeros(M.shape)
#        M_id[M!=0]=1 
        
        # add noise
#        A = skimage.util.random_noise(A, mode = 'gaussian')
        

        A = A.transpose(0,1) # transpose dimension due to the demand of pythorch <batch, channel, W, H>
        B = B.transpose(0,1)
        M = M.transpose(0,1)
        M_codes = M_codes.transpose(0,1)
#        M_id = M_id.transpose(2,0,1)
        
        
        
#        randomly crop image for train
#        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
#        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
        
        
        
        # split AB image into A and B
#        w, h = AB.size


        
#        A = AB.crop((0, 0, w2, h))
#        B = AB.crop((w2, 0, w, h))


        # apply the same transform to both A and B
#        transform_params = get_params(self.opt, A.shape)
#        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
#        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
#
#        A = A_transform(A)
#        B = B_transform(B)
        
        return {'A': A, 'B': B, 'M': M, 'M_codes': M_codes, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

        


