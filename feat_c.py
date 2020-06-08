import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
from tifffile import imsave

class save():
    def __init__(self, type):
        self.save_type=type
    def save_data(self, E_R, E_T, path, image_name):
        if self.save_type=='npz':
            if not os.path.exists(path):
                os.makedirs(path)
            np.savez(os.path.join(path, image_name.replace('png','npz')), rgb=E_R, thr=E_T) 
        elif self.save_type=='tif':
            if not os.path.exists(os.path.join(path.replace('result', 'result/images'), 'rgb')):
                os.makedirs(os.path.join(path.replace('result', 'result/images'), 'rgb'))
                os.makedirs(os.path.join(path.replace('result', 'result/images'), 'thr'))
            imsave(os.path.join(path.replace('result', 'result/images'), 'rgb', image_name.replace('png','tif')), E_R)
            imsave(os.path.join(path.replace('result', 'result/images'), 'thr', image_name.replace('png','tif')), E_T)
            
if __name__ == '__main__':
    result_path='./result'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling  comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip  comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display  the test code saves the results to a HTML file.
    saver=save(opt.save_type)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks  create schedulers
    
    if opt.eval:
        model.eval()
    from tqdm import tqdm
    for i, data in tqdm(enumerate(dataset)):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results

        E_R  = visuals['R_A'].cpu().numpy()[0].transpose((1,2,0))  # MDII (RGB based)
        E_T  = visuals['R_B'].cpu().numpy()[0].transpose((1,2,0))  # MDII (Thermal based)

        img_path = model.get_image_paths()     # get image paths
        img_name = os.path.basename( img_path[0] )  
        svg_path = '/'.join([img_path[0].split('/')[-3]]+img_path[0].split('/')[:-3]+[opt.name]+[opt.epoch+'epoch']+[opt.data_type])
        path=os.path.join(result_path, svg_path)
        saver.save_data(E_R, E_T, path, img_name)
        
