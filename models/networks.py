import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):

#    if len(gpu_ids) > 0:
    assert(torch.cuda.is_available())
#    device = torch.device('cuda:{}'.format(gpu_ids[0])) 
#    net.to(device)
    net.cuda()
    net = torch.nn.DataParallel(net)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net




def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, lr_policy = 'linear', niter = int(200), niter_decay=int(200)):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + int(1)- niter) / float(niter_decay + 1) # lr_l the function multiplied to learning rate, from 1 to 0
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # give lr_lambda to optimizer
        
#    elif lr_policy == 'step':
#        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler



def define_Deconv1d(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64       
    deconvblock = None
    deconvblock = DeconvolutionBlock(input_nc, ngf,  submodule = None, firstlayer = True, lastlayer = False, use_norm = use_norm)    
    deconvblock = DeconvolutionBlock(ngf, ngf * 2,  submodule = deconvblock, firstlayer = False, lastlayer = False, use_norm = use_norm) 
    deconvblock = DeconvolutionBlock(ngf * 2, ngf * 4,  submodule = deconvblock, firstlayer = False, lastlayer = False, use_norm = use_norm) 
    for i in range(2):
        deconvblock = DeconvolutionBlock(ngf * 4, ngf * 4,  submodule = deconvblock, firstlayer = False, lastlayer = False, use_norm = use_norm)  
    for i in range(6):
        deconvblock = FcnBlock(ngf * 4, ngf * 4,  submodule = deconvblock, firstlayer = False, lastlayer = False, use_norm = use_norm)              
        
    deconvblock = DeconvolutionBlock(ngf * 4, ngf * 2,  submodule = deconvblock, firstlayer = False, lastlayer = False, use_norm = use_norm) 
    deconvblock = DeconvolutionBlock(ngf * 2, ngf,  submodule = deconvblock, firstlayer = False, lastlayer = False, use_norm = use_norm) 
    netDeconv1d  = DeconvolutionBlock(ngf, output_nc,  submodule = deconvblock, firstlayer = False, lastlayer = True, use_norm = use_norm)   
    
    return init_net(netDeconv1d, init_type, init_gain, gpu_ids)    

def define_ShortDeconv1d(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 16       
    net = None
#1st layer
    net = ShortDeconvolutionBlock(input_nc, ngf, submodule = None, firstlayer = True, lastlayer = False)    
#    net = ShortDeconvolutionBlock(ngf, ngf * 2, kernel_size = 3, stride = 2, submodule = net, firstlayer = False, lastlayer = False) 
#    net = ShortDeconvolutionBlock(ngf * 2, ngf * 2, kernel_size = 6, stride = 2, submodule = net, firstlayer = False, lastlayer = False) 
#    net = ShortDeconvolutionBlock(ngf * 2, ngf, kernel_size = 1, stride = 1, submodule = net, firstlayer = False, lastlayer = False) 
#    netDeconv1d = ShortDeconvolutionBlock(ngf, output_nc, submodule = net, firstlayer = False, lastlayer = True) 
    net = ShortDeconvolutionBlock(ngf, ngf * 2, kernel_size = 11, stride = 1, submodule = net, firstlayer = False, lastlayer = False) 
    net = ShortDeconvolutionBlock(ngf * 2, ngf * 2, kernel_size = 10, stride = 2, submodule = net, firstlayer = False, lastlayer = False) 
    net = ShortDeconvolutionBlock(ngf * 2, ngf, kernel_size = 1, stride = 1, submodule = net, firstlayer = False, lastlayer = False) 
    netDeconv1d = ShortDeconvolutionBlock(ngf, output_nc, submodule = net, firstlayer = False, lastlayer = True) 
    
    return init_net(netDeconv1d, init_type, init_gain, gpu_ids)        




def define_DeconvFcn1d(input_nc, output_nc, submodule = None, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    convblock = None    
    convblock = FcnBlock(input_nc, input_nc,  submodule = submodule, firstlayer = True, lastlayer = False, use_norm = use_norm)    
    
    for i in range(6):
        convblock = FcnBlock(input_nc,input_nc,  submodule = convblock, firstlayer = False, lastlayer = False, use_norm = use_norm)  
    convblock = FcnBlock(input_nc,input_nc,  submodule = convblock, firstlayer = False, lastlayer = True, use_norm = use_norm)  
    return init_net(convblock, init_type, init_gain, gpu_ids)    


def define_EncoderDecoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64       
    net = None
    net = FcnBlock(input_nc, ngf, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)
    net = FcnBlock(ngf, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)     
    for i in range(4):       
        net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = True, decoder = False, use_norm = use_norm)  
    net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = True, decoder = False, innermost = True, use_norm = use_norm) 
    
    for i in range(7):
        net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = False, decoder = True, use_norm = use_norm)          
#    net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = False, decoder = True, lastlayer = True)  
    
    net = FcnBlock(ngf * 2, ngf,  submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm) 
    net = FcnBlock(ngf, output_nc,  submodule = net,  firstlayer = False, lastlayer = True, use_norm = use_norm) 
 
    return init_net(net, init_type, init_gain, gpu_ids) 


def define_ShortEncoderDecoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64       
    net = None
   
    net = ShortEncoderDecoderBlock(input_nc, ngf * 4, kernel_size = 2, stride = 1, firstlayer = True, submodule = None, encoder = True, decoder = False)  
    net = ShortEncoderDecoderBlock(ngf * 4, ngf * 8, kernel_size = 10, stride = 1, innermost = True, submodule = net, encoder = True, decoder = False)  
    net = ShortEncoderDecoderBlock(ngf * 8, ngf * 4, kernel_size = 8, stride = 2, submodule = net, encoder = False, decoder = True) 
    net = ShortEncoderDecoderBlock(ngf * 4, ngf * 2, kernel_size = 8, stride = 2,  submodule = net, encoder = False, decoder = True) 
    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 1, kernel_size = 8, stride = 2,  submodule = net, encoder = False, decoder = True) 
    net = ShortEncoderDecoderBlock(ngf * 1, output_nc, kernel_size = 1, stride = 1,  submodule = net, encoder = False, decoder = True, lastlayer = True) 

    return init_net(net, init_type, init_gain, gpu_ids) 

def define_ShortM_EncoderDecoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64       
    net = None
    net = FcnBlock(input_nc, ngf, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)                                       # nc = 11
    net = FcnBlock(ngf, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)                                        # nc = 11
    for i in range(4):       
        net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = True, decoder = False, use_norm = use_norm)                           # nc = 9 7 5 3
    net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = True, decoder = False, innermost = True, use_norm = use_norm)             #nc = 1
    
    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 2, kernel_size = 4, stride = 1,  submodule = net, encoder = False, decoder = True)                  #nc = 4
    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 1, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True)      #nc = 4
    net = ShortEncoderDecoderBlock(ngf * 1, output_nc, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True, lastlayer = True)   #nc = 4

    return init_net(net, init_type, init_gain, gpu_ids) 


def define_ShortM_EncoderDecoderFcn(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64       
    net = None
    
    net = FcnBlock(input_nc, ngf, kernel_size = 1, stride=1,  padding=0, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)  # nc = 11       
#    net = FcnBlock(ngf, ngf * 2, kernel_size = 3, stride=1,  padding=1, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)  # nc = 11      
#    net = FcnBlock(ngf * 2, ngf * 4, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)                                       # nc = 11
#    net = FcnBlock( ngf * 4, ngf * 4, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)                                        # nc = 11
    
    for i in range(4):       
        net = EncoderDecoderBlock(ngf , ngf ,  submodule = net, encoder = True, decoder = False, use_norm = use_norm)                           # nc = 9 7 5 3
    net = EncoderDecoderBlock(ngf , ngf ,  submodule = net, encoder = True, decoder = False, innermost = True, use_norm = use_norm)             #nc = 1
    
    net = ShortEncoderDecoderBlock(ngf * 1, output_nc, kernel_size = 4, stride = 1,  submodule = net, encoder = False, decoder = True, lastlayer = True)                  #nc = 4
#    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 1, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True)      #nc = 4
#    net = ShortEncoderDecoderBlock(ngf * 1, output_nc, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True, lastlayer = True)   #nc = 4
    net = FcBlock(4, 4, submodule = net)
#    net = FcBlock(4, 4, submodule = net)
 
    return init_net(net, init_type, init_gain, gpu_ids) 

def define_LongM_EncoderDecoderFcn(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64       
    net = None
    
    net = FcnBlock(input_nc, ngf, kernel_size = 1, stride=1,  padding=0, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)  # nc = 11       
    net = FcnBlock(ngf, ngf * 2, kernel_size = 3, stride=1,  padding=1, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)  # nc = 11      
    net = FcnBlock(ngf * 2, ngf * 4, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)                                       # nc = 11
    net = FcnBlock( ngf * 4, ngf * 4, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)                                        # nc = 11
    
    for i in range(4):       
        net = EncoderDecoderBlock(ngf * 4, ngf * 4,  submodule = net, encoder = True, decoder = False, use_norm = use_norm)                           # nc = 9 7 5 3
    net = EncoderDecoderBlock(ngf * 4, ngf * 4,  submodule = net, encoder = True, decoder = False, innermost = True, use_norm = use_norm)             #nc = 1
    
    net = ShortEncoderDecoderBlock(ngf * 4, ngf * 2, kernel_size = 4, stride = 1,  submodule = net, encoder = False, decoder = True)                  #nc = 4
    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 1, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True)      #nc = 4
    net = ShortEncoderDecoderBlock(ngf * 1, output_nc, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True, lastlayer = True)   #nc = 4
    net = FcBlock(4, 4, submodule = net)
    net = FcBlock(4, 4, submodule = net)
 
    return init_net(net, init_type, init_gain, gpu_ids) 

def define_ShortM_Encoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64       
    net = None
    net = FcnBlock(input_nc, ngf, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)  # nc = 11
    net = FcnBlock(ngf, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)   # nc = 11        
    for i in range(3):       
        net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = True, decoder = False, use_norm = use_norm)    # nc = 9 7 5

    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 2,  kernel_size = 4, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = use_norm)  # nc = 4
    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 1,  kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = use_norm)  
    net = ShortEncoderDecoderBlock(ngf * 1, output_nc,  kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = use_norm)  
    net = FcBlock(4, 4, submodule = net)
    net = FcBlock(4, 4, submodule = net)
    
    return init_net(net, init_type, init_gain, gpu_ids) 



def define_ShortM_Encoder_v2(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    """
    materials with detailed concentrations
    """
    ngf = 64       
    net = None
    net = FcnBlock(input_nc, ngf, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)  # nc = 11
    net = FcnBlock(ngf, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)   # nc = 11        
    for i in range(7):       
        net = FcnBlock(ngf * 2, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm, use_dropout = True)   # nc = 11   
    net = FcnBlock(ngf * 2, ngf, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)   # nc = 11 
    net = FcnBlock(ngf, output_nc, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)   # nc = 11 
    
    net = FcBlock(11, 15, submodule = net)
    net = FcBlock(15, 20, submodule = net)

    return init_net(net, init_type, init_gain, gpu_ids) 

def define_ShortM_Encoder_v3(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    """
    materials exit or not
    """
    ngf = 64       
    net = None
    net = FcnBlock(input_nc, ngf, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)  # nc = 11
    net = FcnBlock(ngf, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)   # nc = 11        
    for i in range(3):       
        net = FcnBlock(ngf * 2, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm, use_dropout = True)   # nc = 11   
    net = FcnBlock(ngf * 2, ngf, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)   # nc = 11 
    net = FcnBlock(ngf, output_nc, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)   # nc = 11 
    
    net = FcBlock(11, 4, submodule = net)


#    for i in range(3):       
#        net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = True, decoder = False, use_norm = use_norm)    # nv = 9 7 5
#    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 2,  kernel_size = 4, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = use_norm)  # nc = 4
#    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 1,  kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = use_norm)  
#    net = ShortEncoderDecoderBlock(ngf * 1, output_nc,  kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = use_norm)  
#    
#    net = FcBlock(4, 4, submodule = net)

    return init_net(net, init_type, init_gain, gpu_ids) 



def define_M_Encoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64       
    net = None
    net = FcnBlock(input_nc, ngf, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)
    net = FcnBlock(ngf, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)     
    net = FcnBlock(ngf * 2, ngf * 4, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm) 
    for i in range(3):       
        net = EncoderDecoderBlock(ngf * 4, ngf * 4,  submodule = net, encoder = True, decoder = False, use_norm = use_norm)  
    net = ShortEncoderDecoderBlock(ngf * 4, ngf * 4,  kernel_size = 4, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = use_norm)  
    net = ShortEncoderDecoderBlock(ngf * 4, ngf * 2,  kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = use_norm)  
    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 1,  kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = use_norm) 
    net = ShortEncoderDecoderBlock(ngf * 1, output_nc,  kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = True, decoder = False, use_norm = False)  
    return init_net(net, init_type, init_gain, gpu_ids) 

def define_M_EncoderDecoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64       
    net = None
    net = FcnBlock(input_nc, ngf, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)
    net = FcnBlock(ngf, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)   
    net = FcnBlock(ngf * 2, ngf * 4, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm) 
    net = FcnBlock(ngf * 4, ngf * 8, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm) 
    
    for i in range(4):       
        net = EncoderDecoderBlock(ngf * 8, ngf * 8,  submodule = net, encoder = True, decoder = False, use_norm = use_norm)  
    net = EncoderDecoderBlock(ngf * 8, ngf * 8,  submodule = net, encoder = True, decoder = False, innermost = True, use_norm = use_norm) 
    
    net = ShortEncoderDecoderBlock(ngf * 8, ngf * 8, kernel_size = 4, stride = 1,  submodule = net, encoder = False, decoder = True)     
    net = ShortEncoderDecoderBlock(ngf * 8, ngf * 4, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True)   
    net = ShortEncoderDecoderBlock(ngf * 4, ngf * 2, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True)  
    net = ShortEncoderDecoderBlock(ngf * 2, ngf * 1, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True)      
    net = ShortEncoderDecoderBlock(ngf * 1, output_nc, kernel_size = 3, stride = 1, padding = 1, submodule = net, encoder = False, decoder = True, lastlayer = True)   
 
    return init_net(net, init_type, init_gain, gpu_ids) 



def define_PhyEncoder_part1(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64           
    net = FcnBlock(input_nc, ngf, submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm)
    net = FcnBlock(ngf, ngf * 2, submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm)     
    for i in range(4):       
        net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = True, decoder = False, use_norm = use_norm)      
#    net = EncoderDecoderBlock(ngf * 2, ngf * 2,  submodule = net, encoder = True, decoder = False, innermost = True)   
    return init_net(net, init_type, init_gain, gpu_ids) 


def define_PhyEncoder_part2(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids=[], use_norm = True):
    ngf = 64               
    net = FcnBlock(ngf * 2, ngf,  submodule = None,  firstlayer = True, lastlayer = False, use_norm = use_norm) 
    net = FcnBlock(ngf, int(ngf * 0.5),  submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm) 
    net = FcnBlock(int(ngf * 0.5), int(ngf * 0.25),  submodule = net,  firstlayer = False, lastlayer = False, use_norm = use_norm) 
    net = FcnBlock(int(ngf * 0.25), output_nc,  submodule = net,  firstlayer = False, lastlayer = True, use_norm = use_norm) 
    
    return init_net(net, init_type, init_gain, gpu_ids) 


class MuTV1dLoss(nn.Module):

    def __init__(self):
        super(MuTV1dLoss, self).__init__()
        self.loss = nn.MSELoss()

    def get_target_tensor(self, vector):
        vector_new = torch.zeros_like(vector)
        vector_label = torch.zeros_like(vector)
        vector_new[:,0,0] = vector[:,0,0] 
        vector_new[:,0,1:] = vector[:,0,0:-1]            
        vector_tv = vector - vector_new            
        vector_new2 = torch.zeros_like(vector_tv)
        vector_new2[:,0,0] = vector_tv[:,0,0]
        vector_new2[:,0,1:] = vector_tv[:,0,0:-1]       
        vector_tv2 = vector_tv - vector_new2      
        vector_tv2[:,0,1] = 0
        vector_tv2[:,0,4:6] = 0
        vector_tv2[:,0,21:23] = 0
        vector_tv[:,0,4] = 0
        vector_tv[:,0,21] = 0        
        return vector_tv, vector_tv2, vector_label

    def __call__(self, vector, TV = int(1)):
        vector_tv, vector_tv2, vector_label = self.get_target_tensor(vector)
        if TV == int(1):        
            loss = self.loss(vector_tv, vector_label)
        elif TV == int(2):
            loss = self.loss(vector_tv2, vector_label)
        else:
            return NotImplementedError('loss_TV [%d] is not implemented', TV)
        
        return loss

class MuDevia1dLoss(nn.Module):

    def __init__(self):
        super(MuDevia1dLoss, self).__init__()
        self.loss = nn.MSELoss()

    def get_target_tensor(self, vector, label):
        vector_new = torch.zeros_like(vector)
        vector_new[:,0,0] = vector[:,0,0] 
        vector_new[:,0,1:] = vector[:,0,0:-1]            
        vector_tv = vector - vector_new             
        vector_new2 = torch.zeros_like(vector_tv)
        vector_new2[:,0,0] = vector_tv[:,0,0]
        vector_new2[:,0,1:] = vector_tv[:,0,0:-1]       
        vector_tv2 = vector_tv - vector_new2    
        
        label_new = torch.zeros_like(label)
        label_new[:,0,0] = label[:,0,0] 
        label_new[:,0,1:] = label[:,0,0:-1]            
        label_tv = label - label_new             
        label_new2 = torch.zeros_like(label_tv)
        label_new2[:,0,0] = label_tv[:,0,0]
        label_new2[:,0,1:] = label_tv[:,0,0:-1]       
        label_tv2 = label_tv - label_new2    
#        
#        label_tv2[:,0,4:6] = 0
#        label_tv2[:,0,21:23] = 0
        return vector_tv2, label_tv2

    def __call__(self, vector, label):
        vector_tv2, label_tv2 = self.get_target_tensor(vector, label)
        loss = self.loss(vector_tv2, label_tv2)      
        return loss

class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss



class DeconvolutionBlock(nn.Module):
    
    def __init__(self, front_nc, back_nc, submodule = None, firstlayer = False, lastlayer = False, use_norm = True):
        super(DeconvolutionBlock, self).__init__()
        self.firstlayer = firstlayer
        self.lastlayer = lastlayer

        norm_layer=nn.BatchNorm1d
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(back_nc)

        if firstlayer:
            deconv = nn.ConvTranspose1d(front_nc, back_nc,                                    
                           kernel_size=5, stride=1,
                           padding=0,output_padding = 0)
            if use_norm:
                model = [deconv, upnorm]
            else:
                model = [deconv]
            
        elif lastlayer:
            deconv = nn.ConvTranspose1d(front_nc, back_nc,                                    
                                       kernel_size=6, stride=1,
                                       padding=0,output_padding = 0)            
            model = [submodule, uprelu, deconv]
        else:            
            deconv = nn.ConvTranspose1d(front_nc, back_nc,                                    
                                       kernel_size=6, stride=1,
                                       padding=0,output_padding = 0)                      
            
            if use_norm:
                model = [submodule, uprelu,deconv,upnorm]
            else:
                model = [submodule, uprelu,deconv]
            
        self.model = nn.Sequential(*model)

       
    def forward(self, input):
        """Standard forward"""
#        print(input.size(), self.model(input).size())
        return self.model(input)

class ShortDeconvolutionBlock(nn.Module):
    
    def __init__(self, front_nc, back_nc, kernel_size = 1, stride=1, submodule = None, firstlayer = False, lastlayer = False, use_norm = True):
        super(ShortDeconvolutionBlock, self).__init__()
        self.firstlayer = firstlayer
        self.lastlayer = lastlayer

        norm_layer=nn.BatchNorm1d
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(back_nc)

        if firstlayer:
            deconv = nn.ConvTranspose1d(front_nc, back_nc,                                    
                           kernel_size=1, stride=1,
                           padding=0,output_padding = 0)
            if use_norm:
                model = [deconv, upnorm]
            else:
                model = [deconv]
            
        elif lastlayer:
            deconv = nn.ConvTranspose1d(front_nc, back_nc,                                    
                                       kernel_size=1, stride=1,
                                       padding=0,output_padding = 0)            
            model = [submodule, uprelu, deconv]
        else:            
            deconv = nn.ConvTranspose1d(front_nc, back_nc,                                    
                                       kernel_size = kernel_size, stride = stride,
                                       padding=0,output_padding = 0)                      
            
            if use_norm:
                model = [submodule, uprelu,deconv,upnorm]
            else:
                model = [submodule, uprelu,deconv]
            
        self.model = nn.Sequential(*model)
    def forward(self, input):
        """Standard forward"""
#        print(input.size(), self.model(input).size())
        return self.model(input)
    
class FcnBlock(nn.Module):
    
    def __init__(self, inner_nc, outer_nc, kernel_size = 5, stride=1,  padding=2, submodule = None,  firstlayer = False, lastlayer = False, use_norm = True, use_dropout = False):
        super(FcnBlock, self).__init__()
        self.firstlayer = firstlayer

        norm_layer=nn.BatchNorm1d
#        uprelu = nn.Sigmoid()
        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc)

        if firstlayer:
            
            conv = nn.Conv1d(inner_nc, outer_nc,                                    
                           kernel_size=kernel_size, stride=stride,
                           padding=padding)
            if submodule is None:
                if use_norm:
                    model = [conv, upnorm]
                else:
                    model = [conv]                               
            else:
                if use_norm:
                    model = [submodule, conv, upnorm]
                else:
                    model = [submodule, conv]                                               
        elif lastlayer:
            conv = nn.Conv1d(inner_nc, outer_nc,                                    
                           kernel_size=kernel_size, stride=stride,
                           padding=padding)        
            model = [submodule, uprelu, conv]        
        else:            
            conv = nn.Conv1d(inner_nc, outer_nc,                                    
                           kernel_size=kernel_size, stride=stride,
                           padding=padding)                      
            
#            if use_norm:
#                model = [submodule, uprelu,conv,upnorm]
#            else:
#                model = [submodule, uprelu,conv]                   
            if use_dropout:
                model = [submodule, uprelu,conv,upnorm, nn.Dropout(0.5)]            
            else:
                model = [submodule, uprelu,conv,upnorm]    
        self.model = nn.Sequential(*model)
       
    def forward(self, input):
        """Standard forward"""
#        print(input.size(), self.model(input).size())
        return self.model(input)
        

class FcBlock(nn.Module):
    
    def __init__(self, inner_nc, outer_nc, submodule = None):
        super(FcBlock, self).__init__()
        model = nn.Linear(inner_nc, outer_nc)
        model = [submodule, model]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        """Standard forward"""
#        print(input.size(), self.model(input).size())
        return self.model(input)    
    

class EncoderPhyBlock(nn.Module):
    def __init__(self, front_nc, back_nc, ngf = 64, submodule = None, encoder = False, lastlayer = False, innermost = False, use_norm = True):
        super(EncoderPhyBlock, self).__init__()

        norm_layer=nn.BatchNorm1d
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(back_nc)

        if encoder:
            enc = nn.Conv1d(front_nc, back_nc,                                    
                           kernel_size=5, stride=1,
                           padding=1)
            if innermost:
                model = [submodule, uprelu, enc]
            else:
                if use_norm:
                    model = [submodule, uprelu, enc, upnorm]
                else:
                    model = [submodule, uprelu, enc]                 
                
        else:      
            return NotImplementedError('EncoderDecoder is not implemented')

        self.model = nn.Sequential(*model)
       
    def forward(self, input):
        """Standard forward"""
#        print(input.size(), self.model(input).size())
        return self.model(input)
    
    

class EncoderDecoderBlock(nn.Module):
    
    def __init__(self, front_nc, back_nc, ngf = 64, submodule = None, encoder = False, decoder = False, lastlayer = False, innermost = False, use_norm = True):
        super(EncoderDecoderBlock, self).__init__()

        norm_layer=nn.BatchNorm1d
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(back_nc)

        if encoder:
            enc = nn.Conv1d(front_nc, back_nc,                                    
                           kernel_size=5, stride=1,
                           padding=1)
            if innermost:
                model = [submodule, downrelu, enc]
            else:
                if use_norm:
                    model = [submodule, downrelu, enc, upnorm]
                else:
                    model = [submodule, downrelu, enc]
                                    
        elif decoder:
            deconv = nn.ConvTranspose1d(front_nc, back_nc,                                    
                                       kernel_size=8, stride=1,
                                       padding=0, output_padding = 0)         
            if lastlayer:               
                model = [submodule, uprelu, deconv]
            else:
                if use_norm:
                    model = [submodule, uprelu, deconv, upnorm]
                else:
                    model = [submodule, uprelu, deconv]               
                
        else:      
            return NotImplementedError('EncoderDecoder is not implemented')

        self.model = nn.Sequential(*model)
       
    def forward(self, input):
        """Standard forward"""
#        print(input.size(), self.model(input).size())
        return self.model(input)
    
class ShortEncoderDecoderBlock(nn.Module):
    def __init__(self, front_nc, back_nc,kernel_size = 10, stride = 1, padding = 0, ngf = 64, submodule = None, encoder = False, decoder = False,
                 firstlayer = False, lastlayer = False, innermost = False, use_norm = True):
        super(ShortEncoderDecoderBlock, self).__init__()

        norm_layer=nn.BatchNorm1d
        downrelu = nn.LeakyReLU(0.2, True)
#        downrelu = nn.Sigmoid()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(back_nc)

        if encoder:
            enc = nn.Conv1d(front_nc, back_nc,                                    
                           kernel_size=kernel_size, stride = stride,
                           padding = padding)
            if innermost:
                model = [submodule, downrelu, enc]
            elif firstlayer:
                model = [enc, upnorm]
                    
            else:
                if use_norm:
                    model = [submodule, downrelu, enc, upnorm]
                else:
                    model = [submodule, downrelu, enc]
                                    
        elif decoder:
            deconv = nn.ConvTranspose1d(front_nc, back_nc,                                    
                                       kernel_size = kernel_size, stride = stride,
                                       padding = padding, output_padding = 0)         
            if lastlayer:               
                deconv = nn.ConvTranspose1d(front_nc, back_nc,                                    
                           kernel_size = kernel_size, stride = stride,
                           padding = padding, output_padding = 0)    
                model = [submodule, uprelu, deconv]
            else:
                if use_norm:
                    model = [submodule, uprelu, deconv, upnorm]
                else:
                    model = [submodule, uprelu, deconv]                              
        else:      
            return NotImplementedError('EncoderDecoder is not implemented')
        self.model = nn.Sequential(*model)       
    def forward(self, input):
        """Standard forward"""
#        print(input.size(), self.model(input).size())
        return self.model(input)
    


def define_Unet1d(input_nc, output_nc, ngf = 64,   init_type='normal', init_gain=0.02,norm_layer=nn.BatchNorm1d, gpu_ids=[], use_dropout=False):
    net = None
    
    unet_block = UnetSkipConnection1dBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
    for i in range(2):          # add intermediate layers with ngf * 8 filters
        unet_block = UnetSkipConnection1dBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)   
    unet_block = UnetSkipConnection1dBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = UnetSkipConnection1dBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = UnetSkipConnection1dBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    net = UnetSkipConnection1dBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost laye
        
    return init_net(net, init_type, init_gain, gpu_ids)      

    




class UnetSkipConnection1dBlock(nn.Module):


    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm1d, use_dropout=False, use_norm = True):
 
        super(UnetSkipConnection1dBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d
        if input_nc is None:
            input_nc = outer_nc
        downconv =  nn.Conv1d(outer_nc, inner_nc,                                    
                           kernel_size=10, stride=1,
                           padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        decnorm = norm_layer(input_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:           
            deconv =  nn.ConvTranspose1d(input_nc, input_nc,                                    
                                       kernel_size=4, stride=2,
                                       padding=0)
            upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc,
                                        kernel_size=10, stride=1,
                                        padding=1)
            
            down = [deconv, decnorm, downrelu, deconv, decnorm, downrelu, downconv]
#            up = [uprelu, upconv, nn.Tanh()]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose1d(inner_nc, outer_nc,
                                        kernel_size=10, stride=1,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc,
                                        kernel_size=10, stride=1,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
    
def define_D(input_nc, output_c, ndf = 64, n_layers_D=3,  init_type='normal', init_gain=0.02, gpu_ids=[]):


#    net = NLayerDiscriminator1d(input_nc, ndf, n_layers=3, norm_layer=nn.BatchNorm1d)
    net = ShortNLayerDiscriminator1d(input_nc, output_c, ndf, norm_layer=nn.BatchNorm1d)

    return init_net(net, init_type, init_gain, gpu_ids)    
    
class NLayerDiscriminator1d(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm1d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator1d, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm1d
        else:
            use_bias = norm_layer != nn.BatchNorm1d

#        kw = 4
        padw = 1
        
        
        sequence = [nn.Conv1d(input_nc, ndf, kernel_size=9, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        sequence += [norm_layer(ndf), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=8, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult  # 4 for n_layer=3
        nf_mult = min(2 ** n_layers, 8) # 8 for n_layer=3
        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=6, stride=1, padding=padw, bias=use_bias),
#            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv1d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence) # * multi inputs

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
    
class ShortNLayerDiscriminator1d(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, output_c, ndf=64, n_layers=3, norm_layer=nn.BatchNorm1d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ShortNLayerDiscriminator1d, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm1d
        else:
            use_bias = norm_layer != nn.BatchNorm1d

#        kw = 4
        padw = 0
        
        
        sequence = [nn.Conv1d(input_nc, ndf, kernel_size = 1, stride = 1, padding=padw), norm_layer(ndf), nn.LeakyReLU(0.2, True)]       # nc = 11
        

        sequence += [
            nn.Conv1d(ndf, ndf, kernel_size = 3, stride = 2, padding=padw, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ]                                                                                                               # nc = 6

        sequence += [                                                                                       
            nn.Conv1d(ndf, ndf, kernel_size = 3, stride = 1, padding=padw, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ]                                                                                                               # nc = 4

        sequence += [                                                                                       
            nn.Conv1d(ndf, ndf, kernel_size = 3, stride = 1, padding=padw, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ]                                                                                                               # nc = 2
        
        sequence += [                                                                                       
            nn.Conv1d(ndf, ndf, kernel_size = 2, stride = 1, padding=padw, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ]                                                                                                               # nc = 1        
        
        
        
        sequence += [nn.Conv1d(ndf, output_c, kernel_size = 1, stride = 1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence) # * multi inputs

    def forward(self, input):
        """Standard forward."""
        return self.model(input)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    