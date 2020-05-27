import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling  comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip  comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display  the test code saves the results to a HTML file.
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
        
        svg_path = '/'.join(img_path[0].split('/')[:-3]+[opt.epoch]+[opt.name]+[opt.phase]+[img_path[0].split('/')[-3]]) 
        svg_path = svg_path.replace('datasets','results') 

        if not os.path.exists(svg_path):
           os.makedirs( svg_path ) 
        np.savez( os.path.join( svg_path, img_name.replace('jpg','npz') ), rgb=E_R, thr=E_T) 

