#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:32:25 2019

@author: xie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:54:54 2019

@author: xie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:12:05 2019

@author: xie
"""
import os
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .base_model import BaseModel
from . import networks



class EncoderDecoderLongFcn1dMModel(BaseModel):
    
    def __init__(self, input_nc, output_nc, lr, isTrain):
        BaseModel.__init__(self)
#        self.gpu_ids = str(0)
#        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = './checkpoints/encoderdecoderlongfcn1dM/'
        self.loss_names = ['L1','L2']
        self.model_names = ['G']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_M']
        

#        self.netG = networks.define_ShortM_EncoderDecoderFcn(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids, use_norm = True)
        self.netG = networks.define_LongM_EncoderDecoderFcn(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids, use_norm = True)
        
        self.lr = lr
        self.beta1 = float(0.5)
        
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.criterionTV = networks.MuTV1dLoss().to(self.device)
        self.criterionDe2 = networks.MuDevia1dLoss().to(self.device)
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        self.criterionL2 = torch.nn.MSELoss().to(self.device)
     
        
    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['M_codes'].to(self.device)
#        print(self.device)
        self.real_M = input['M'].to(self.device)
        self.real_A = self.real_A.float()
        self.real_B = self.real_B.float()
        self.image_paths = input['A_paths']

    def forward(self):

        self.fake_B = self.netG(self.real_A)  

        
    def backward(self):
        self.loss_L1 = self.criterionL1(self.real_B, self.fake_B)
        self.loss_L2 = self.criterionL2(self.real_B, self.fake_B)
        self.loss_1d = self.loss_L1
        self.loss_1d.backward()
        

    def optimize_parameters(self):

        self.forward()                   # compute fake images: G(A)       
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward()                   # calculate graidents for G
        self.optimizer_G.step()          # update D's weights


    


        
        








        
   