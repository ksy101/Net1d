#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:55:31 2019

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
from scipy.io import loadmat, savemat


class PhyEncoder1dModel(BaseModel):
    
    def __init__(self, input_nc, output_nc, lr):
        BaseModel.__init__(self)
        self.gpu_ids = str(0)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU     
        self.save_dir = './checkpoints/phyencoder1d/'
        self.loss_names = ['L2']
        self.model_names = ['PhyEncoder1d_1', 'PhyEncoder1d_2']
        self.visual_names = ['A', 'B', 'pred_B']

        AH = loadmat('AH_3.mat')
        AH = AH['AH_3']
        Phydata = torch.from_numpy(AH)
        Phydata = Phydata.float()
        Phydata = Phydata.transpose(0,1)
        Phydata = Phydata.to(self.device)

        
        self.Phydata = Phydata
        self.netPhyEncoder1d_1 = networks.define_PhyEncoder_part1(input_nc,output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids)        
        self.netPhyEncoder1d_2 = networks.define_PhyEncoder_part2(128, output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids)
         
        
        self.lr = lr
        self.beta1 = float(0.5)
        self.optimizer_1 = torch.optim.Adam(self.netPhyEncoder1d_1.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_2 = torch.optim.Adam(self.netPhyEncoder1d_2.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.criterionTV = networks.MuTV1dLoss().to(self.device)
        self.criterionDe2 = networks.MuDevia1dLoss().to(self.device)
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        self.criterionL2 = torch.nn.MSELoss().to(self.device)
     
        
    def set_input(self, input):                
        self.A = input['A'].to(self.device)
        self.B = input['B'].to(self.device)
        self.A = self.A.float() 
        self.B = self.B.float() 
        self.image_paths = input['A_paths']          

    def forward(self):
        A = self.A
        self.pred_M = self.netPhyEncoder1d_1(A)  
        result = torch.mm(self.pred_M.view(-1, 3), self.Phydata)
        result = result.view(-1, 128, 50)
        self.pred_B1 = result
        self.pred_B = self.netPhyEncoder1d_2(result)  
        
    def backward(self):
        self.loss_L1 = self.criterionL1(self.B, self.pred_B)
        self.loss_L2 = self.criterionL2(self.B, self.pred_B)
        self.loss_tv = self.criterionTV(self.pred_B, TV = int(2))* 1
        self.loss_de = self.criterionDe2(self.pred_B, self.B)* 1
        self.loss_1d = self.loss_L2 
        self.loss_1d.backward()
        

    def optimize_parameters(self):

        self.forward()                   
        self.optimizer_1.zero_grad()      
        self.optimizer_2.zero_grad() 
        
        self.backward()                   
        self.optimizer_1.step()          
        self.optimizer_2.step()         


    


        
        








        
   