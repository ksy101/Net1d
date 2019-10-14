import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
#from scipy.misc import imresize
from scipy.io import savemat
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(visuals, net_name, image_path):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
#    image_dir = webpage.get_image_dir()
    
    num_img = len(image_path)
#    print(visuals)
    for i in range(num_img):
        short_path = ntpath.basename(image_path[i])
        name = os.path.splitext(short_path)[0]
    #    print(short_path, name)
    #    webpage.add_header(name)
        ims, txts, links = [], [], []
    
        for label, im_data in visuals.items():
            
            im = util.tensor2im(im_data, image_id = i)        
            image_name = '%s_%s' % (name, label)        
            path_parent = './result/' + net_name + '/test/'
            if not os.path.exists(path_parent):                
                os.makedirs(path_parent)
                print('create save dir: %s' %path_parent)
            save_path = os.path.join(path_parent, image_name)       
            util.save_image(im, save_path)

#        ims.append(image_name)
#        txts.append(label)
#        links.append(image_name)
#    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, net_name = 'deonv1d'):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """

        display_ncols = int(5)
        display_server = 'http://localhost'
        display_env = 'main'        
        checkpoints_dir = './checkpoints'
        name = net_name
        
        
#        self.opt = opt  # cache the option
        self.display_id = int(1)
        self.use_html = True
        self.win_size = int(256)
        self.name = name
        self.port = int(8097)
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = display_ncols
            self.vis = visdom.Visdom(server=display_server, port=self.port, env=display_env)
#            self.vis = visdom.Visdom()
            if not self.vis.check_connection():
                print('f 1')
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(checkpoints_dir, name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(checkpoints_dir, name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
#        cmd = sys.executable + ' -m visdom.server'
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)



    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()



    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.7f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.7f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


    def display_current_results(self, visuals, epoch, save_result, id_visdom, image_id = int(0)):
        
        
        
        if self.display_id > 0:  # show images in the browser using visdom
                       
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                
#                print(visuals)
#                print(next(iter(visuals.values())).shape)
                n, d, l = next(iter(visuals.values())).shape
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (d, l)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
#                    print(label)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([0, 1])) # due to the first dimension for visdom.image is channels
#                    print(image_numpy.transpose([2, 0, 1]).shape)
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''

                        
                white_image = np.ones_like(image_numpy.transpose([0, 1])) * 255
#                print(images)
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:    
                   
                    im_real_A = images[0]
                    im_fake_B = images[1]
                    im_real_B = images[2]
                                         
                    w,h = im_real_B.shape    
                    out_nc = h

                    if out_nc == 50:
                        self.vis.line(
                                Y= im_real_A[0],
                                opts={
                                    'title': 'pix_0' + '_A',
    #                                'legend': 'real_A',
                                    'xlabel': 'epoch',
                                    'ylabel': 'Mu'},
                                win=id_visdom + 2)
                        
                        self.vis.line(
                                X = np.column_stack((np.arange(1, out_nc + 1), np.arange(1, out_nc + 1))),
                                Y = np.column_stack((im_real_B[0], im_fake_B[0])),
                                
                                opts={
                                    'title': 'pix_0' + '_B',
                                    'legend': ['real_B', 'fake_B'],
                                    'xlabel': 'epoch',  
                                    'ylabel': 'Mu'},
                                    win = id_visdom + 3,
                                    name = '1',
                                    update = 'append'
                                    )
                        
                        if image_id != 0:
                            for label, image in visuals.items():
                                image_numpy = util.tensor2im(image, image_id = image_id)
                                label_html_row += '<td>%s</td>' % label
                                images.append(image_numpy.transpose([0, 1])) # due to the first dimension for visdom.image is channels
                                
                            im_real_A_2 = images[3]
                            im_fake_B_2 = images[4]
                            im_real_B_2 = images[5]
                            
                            self.vis.line(
                                    Y= im_real_A_2[0],
                                    opts={
                                        'title': 'pix_' + str(image_id) + '_A',
        #                                'legend': 'real_A',
                                        'xlabel': 'epoch',
                                        'ylabel': 'Mu'},
                                    win=id_visdom + 4)
        
                            self.vis.line(
                                    X = np.column_stack((np.arange(1, out_nc + 1), np.arange(1, out_nc + 1))),
                                    Y = np.column_stack((im_real_B_2[0], im_fake_B_2[0])),
                                    
                                    opts={
                                        'title': 'pix_' + str(image_id) + '_B',
                                        'legend': ['real_B', 'fake_B'],
                                        'xlabel': 'epoch',  
                                        'ylabel': 'Mu'},
                                        win = id_visdom + 5,
                                        name = '1',
                                        update = 'append'
                                        )
                    else:
                        if image_id != 0:
#                            num_win = len(image_id)
                            for i in range(len(image_id)):
                                for label, image in visuals.items():                                
                                    image_numpy = util.tensor2im(image, image_id = image_id[i])
                                    label_html_row += '<td>%s</td>' % label
                                    images.append(image_numpy.transpose([0, 1])) # due to the first dimension for visdom.image is channels
#                                print(images)
#                            print(len(image_id))
                            for i in range(len(image_id)):  
                                im_real_A_2 = images[4 + i*4]
                                im_fake_B_2 = images[5 + i*4]
                                im_real_B_2 = images[6 + i*4]
#                                print(im_real_B_2.shape)
#                            line_real_B = im_real_B_2[0]
#                            line_fake_B = im_fake_B_2[0]
#                            line_real_B[2:] = line_real_B[2:]*1e3
#                            line_fake_B[2:] = line_fake_B[2:]*1e3

                                self.vis.line(
                                        Y= im_real_A_2[0],
                                        opts={
                                            'title': 'pix_' + str(image_id[i]) + '_A',
            #                                'legend': 'real_A',
                                            'xlabel': 'epoch',
                                            'ylabel': 'Mu'},
                                        win=id_visdom + 4 + i*2)                               

                                self.vis.line(
                                        X = np.column_stack((np.arange(1, out_nc + 1), np.arange(1, out_nc + 1))),
                                        Y = np.column_stack((im_real_B_2[0], im_fake_B_2[0])),
                                        
                                        opts={
                                            'title': 'pix_' + str(image_id[i]) + '_B_IorGd',
                                            'legend': ['real_B', 'fake_B'],
                                            'xlabel': 'epoch',  
                                            'ylabel': 'Mu'},
                                            win = id_visdom + 5 + i*2
#                                            name = '1',
#                                            update = 'append'
                                            )    
                           
#                                self.vis.line(
#                                        X = np.column_stack((np.arange(1, out_nc + 1), np.arange(1, out_nc + 1))),
#                                        Y = np.column_stack((im_real_B_2[0], im_fake_B_2[0])),
#                                        
#                                        opts={
#                                            'title': 'pix_' + str(image_id[i]) + '_B_Gd',
#                                            'legend': ['real_B', 'fake_B'],
#                                            'xlabel': 'epoch',  
#                                            'ylabel': 'Mu'},
#                                            win = id_visdom + 6 + i*2
##                                            name = '1',
##                                            update = 'append'
#                                            )                                        
#                
                                self.vis.line(
                                        Y= im_real_A[0],
                                        opts={
                                            'title': 'pix_0' + '_A',
            #                                'legend': 'real_A',
                                            'xlabel': 'epoch',
                                            'ylabel': 'Mu'},
                                        win=id_visdom + 2)
                               
                                self.vis.line(
                                        X = np.column_stack((np.arange(1, out_nc + 1), np.arange(1, out_nc + 1))),
                                        Y = np.column_stack((im_real_B[0], im_fake_B[0])),
                                        
                                        opts={
                                            'title': 'pix_0' + '_B_PMMA',
                                            'legend': ['real_B', 'fake_B'],
                                            'xlabel': 'epoch',  
                                            'ylabel': 'Mu'},
                                            win = id_visdom + 3,
                                            name = '1',
                                            update = 'append'
                                            )    


                except VisdomExceptionBase:
                    self.create_visdom_connections()
