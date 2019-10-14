#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:49:01 2019

@author: xie
"""
import time
import os.path
from util.visualizer import Visualizer
from models import create_model
from util.util import CustomDatasetDataLoader




if __name__ == '__main__':    
    
    net_name = 'encoderdecoderlongfcn1dM'
    
    batch_size = int(40000)
    niter = int(2000)
    niter_decay=int(1000)
    
    
    visualizer = Visualizer(net_name = net_name)     
    data_loader = CustomDatasetDataLoader('/tmp/xie/data/train/', batch_size)
    dataset = data_loader.load_data()
    dataset = dataset            
    dataset_size = len(dataset) 
#    print(dataset_size)
    
    model = create_model(net_name, 1, 1, 0.001, isTrain = True) 
    model.setup(isTrain = True, continue_train = False, lr_policy = 'linear', niter = niter,
                niter_decay = niter_decay, load_iter = 1260)
    
    
    total_iters = 0                # the total number of training iterations
    id_visdom = 0
    
    for epoch in range(niter + niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>        
        

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % 100 == 0:
                t_data = iter_start_time - iter_data_time
            
            model.set_input(data)
            model.optimize_parameters()
            total_iters += batch_size
            epoch_iter += batch_size
                     
            if total_iters % 100 == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
#                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)         
#                visualizer.display_current_results(model.get_current_visuals(), epoch, False, id_visdom, image_id = [98,67,164])  # 31
                
                   
            iter_data_time = time.time()
#        if epoch % 100 == 0:

        if epoch % 10 == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)    
    
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, niter + niter_decay, time.time() - epoch_start_time))
#        print('Learning rate before update: %.7f' % opt.lr)
        model.update_learning_rate()  


    












