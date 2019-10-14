import os
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .base_model import BaseModel
from . import networks


class Pix2Pix1dModel(BaseModel):


    def __init__(self, input_nc, output_nc, lr, isTrain):

        BaseModel.__init__(self)
        self.isTrain = isTrain
#        self.gpu_ids = str(0)
#        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU     
        self.save_dir = './checkpoints/pix2pix1d/'
#        self.model_names = ['pix2pix1d']   
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_M']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
            
#        self.netG = networks.define_EncoderDecoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids)
#        self.netG = networks.encoderdecoder1dM(input_nc, output_nc, ngf = 64,  init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
#        self.netG = networks.define_Deconv1d(input_nc,output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids)
#        self.netG = networks.define_ShortM_EncoderDecoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids)
        self.netG = networks.define_ShortM_EncoderDecoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids, use_norm = True)
        
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(input_nc, output_nc, ndf = 64, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids) 

        if self.isTrain:
            self.lr = lr
            self.beta1 = float(0.5)
            self.criterionGAN = networks.GANLoss(gan_mode = 'vanilla').to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['M_codes'].to(self.device)
        self.real_M = input['M'].to(self.device)
        self.real_A = self.real_A.float()
        self.real_B = self.real_B.float()
        self.image_paths = input['A_paths']        

    def forward(self):
        real_A = self.real_A
        self.fake_B = self.netG(real_A)  

    def backward_D(self):
        fake_BM  = self.fake_B[:,:,2:]
        
        fake_AB = torch.cat((self.real_A, fake_BM), 2)  # we use conditional GANs; we need to feed both input and output to the discriminator
#        print(fake_AB.size())
        pred_fake = self.netD(fake_AB.detach())
        print(pred_fake.size())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_BM  = self.real_B[:,:,2:]
        real_AB = torch.cat((self.real_A, real_BM), 2)
#        print(real_AB.size())
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_BM  = self.fake_B[:,:,2:]
        fake_AB = torch.cat((self.real_A, fake_BM), 2)
        pred_fake = self.netD(fake_AB)
        print(pred_fake.size())
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 1000

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
