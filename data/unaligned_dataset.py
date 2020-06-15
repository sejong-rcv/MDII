import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags  needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.data_type + 'A')  # create a path '/path/to/data/trainA' : RGB
        self.dir_B = os.path.join(opt.dataroot, opt.data_type + 'B')  # create a path '/path/to/data/trainB': Thermal
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        if 'M' in opt.loss_type:
           if opt.data_type=='train':
              gps_name = 'AM09_gps.txt' 
           else: 
              gps_name = 'AM05_gps.txt' 
           with open( os.path.join( './datasets', gps_name ), 'r') as fp:
             gps=[[float(x.rstrip().split()[0]),float(x.rstrip().split()[1])] for x in fp.readlines()] 
           gps = np.array(gps) 
           gps = np.float32(gps) 
           import sklearn.metrics as skm
           Distance = skm.pairwise_distances( gps, gps ) 
           D = (Distance > opt.mt_neg) 
           self.DM = D  
           self.DV= Distance 
        else:
           self.DM = None 

    def __select_negative(self, index, mode='A'):
        DM = np.where( self.DM[index] == 1)[0] 
        if DM.shape[0] != 0:
           idx = torch.randperm( DM.shape[0]) 
        else:
           idx = torch.randperm( self.DM.shape[0] ) 
        C = DM[idx[0].item()] 
        if mode == 'A':
           n_idx = self.A_paths[C] 
        else:
           n_idx = self.B_paths[C] 
        return n_idx

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        if self.opt.phase == 'train':
           index_A = index % self.A_size 
           index_B = index_A 
           while(index_A == index_B):
             if self.opt.serial_batches:   # make sure index is within then range
                index_B = index % self.B_size
             else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
        else:
          if self.opt.serial_batches:   # make sure index is within then range
              index_B = index % self.B_size
          else:   # randomize the index for domain B to avoid fixed pairs.
              index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        if self.opt.no_color:
           A_img = A_img.convert('L') 
           A_img = A_img.convert('RGB') 

        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
       
        if 'M' in self.opt.loss_type:
           A_neg = self.__select_negative( index  , mode='B') 
           B_neg = self.__select_negative( index_B, mode='A') 
           A_n   = Image.open(A_neg).convert('RGB') 
           B_n   = Image.open(B_neg).convert('RGB') 
           if self.opt.no_color:
              B_n = B_n.convert('L') 
              B_n = B_n.convert('RGB') 
           B_n   = self.transform_A(B_n) 
           A_n   = self.transform_B(A_n) 
        else:
           A_n, B_n = B, A 
        return {'A': A, 'B': B, 'A_n': A_n, 'B_n': B_n, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
