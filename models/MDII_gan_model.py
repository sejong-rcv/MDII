import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class MDIIGANModel(BaseModel):
    """
    This class implements the MDII model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    "
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B  G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B  D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--delta', type=float, default=0.1, help='weight for SSIM loss')
            parser.add_argument('--beta', type=float, default=1.0, help='weight for encoder loss')
            parser.add_argument('--gamma', type=float, default=1.0, help='weight for style loss')
            parser.add_argument('--alpha', type=float, default=0.1, help='weight for metric loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--gamma_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags  needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A','G_A','cycle_A','idt_A','D_B','G_B','cycle_B','idt_B','S_A','S_B','M_A','M_B','E_A','E_B','X_A','X_B','SS_A','SS_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'R_A', 'R_AB']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'R_B', 'R_BA']

        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('X_BB')
            visual_names_B.append('X_AA')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'E']
        else:  # during test time, only load Gs
            #self.model_names = ['G_A', 'G_B', 'E']
            self.model_names= ['E']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.outE_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.outE_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        """ ENCODER """
        self.netE  = networks.define_G(opt.input_nc, opt.outE_nc, opt.nef, opt.netE, opt.normE,
                     not opt.no_dropoutE, opt.init_typeE, opt.init_gainE, self.gpu_ids, keep_size=self.opt.keep_size, downsample=self.opt.downsample, aef=opt.aef)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.X_AA_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.X_BB_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            if self.opt.gan_mode == 'relative':
              self.criterionGAN = networks.PairedGANLoss().to(self.device) 
            else:
              self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSTL = torch.nn.MSELoss()
            self.criterionMT  = networks.TripletLoss(margin=self.opt.mt_margin, p=self.opt.mt_p) 
            self.criterionENC = torch.nn.L1Loss() 
            self.criterionSSIM= networks.SSIMLoss() 

            # initialize optimizers  schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),self.netE.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
        if 'M' in self.opt.loss_type:
           self.real_A_n = input['A_n' if AtoB else 'B_n'].to(self.device)
           self.real_B_n = input['B_n' if AtoB else 'A_n'].to(self.device)
        

    def forward(self):
        """Run forward pass  called by both functions <optimize_parameters> and <test>."""
        """
        Generators: G_A: A -> B  G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B  D_B: G_B(B) vs. A.
        """
        self.R_A ,self.F_A  = self.netE(self.real_A)  
        self.X_AB           = self.netG_A(self.R_A)  
        self.fake_B = self.X_AB 
        self.X_AA           = self.netG_B(self.R_A) 
        self.R_AB,self.F_AB = self.netE(self.X_AB) 
        self.R_AA,self.F_AA = self.netE(self.X_AA) 
        self.X_ABA          = self.netG_B(self.R_AB)  
        self.rec_A = self.X_ABA 

        self.R_B ,self.F_B  = self.netE(self.real_B)  
        self.X_BA           = self.netG_B(self.R_B) 
        self.fake_A = self.X_BA 
        self.X_BB           = self.netG_A(self.R_B) 
        self.R_BA,self.F_BA = self.netE(self.X_BA) 
        self.R_BB,self.F_BB = self.netE(self.X_BB) 
        self.X_BAB          = self.netG_A(self.R_BA)  
        self.rec_B = self.X_BAB 

    def backward_D_basic(self, netD, real, fake_list):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        pred_real = netD(real)
        loss_D_fake = 0 
 
        if self.opt.gan_mode == 'relative':
          for fake in fake_list:
              pred_fake = netD(fake.detach())
              loss_D_fake += self.criterionGAN(pred_real, pred_fake, True) 
          loss_D = loss_D_fake * (1./len(fake_list)) 
        else:
          loss_D_real = self.criterionGAN(pred_real, True)
          for fake in fake_list:
              pred_fake = netD(fake.detach())
              loss_D_fake += self.criterionGAN(pred_fake, False) 
          # Combined loss and calculate gradients
          loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """ Calculate GAN loss for discriminator D_A """
        """ Discriminators: D_A: G_A(A) vs. B  D_B: G_B(B) vs. A. """
        fake_B = self.fake_B_pool.query(self.fake_B)
        X_BB   = self.X_BB_pool.query(self.X_BB) 
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, [fake_B,X_BB])  

    def backward_D_B(self):
        """ Calculate GAN loss for discriminator D_B """
        """ Discriminators: D_A: G_A(A) vs. B  D_B: G_B(B) vs. A. """
        fake_A = self.fake_A_pool.query(self.fake_A)
        X_AA   = self.X_AA_pool.query(self.X_AA) 
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, [fake_A,X_AA])  
    
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||  
            self.loss_idt_A = self.criterionIdt(self.X_BB, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.loss_idt_B = self.criterionIdt(self.X_AA, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        if self.opt.gan_mode == 'relative':
           self.loss_G_A=self.criterionGAN(self.netD_A(self.real_B),self.netD_A(self.fake_B),False) 
           self.loss_G_B=self.criterionGAN(self.netD_B(self.real_A),self.netD_B(self.fake_A),False) 
        else:
           # GAN loss D_A(G_A(A))
           self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
           # GAN loss D_B(G_B(B))
           self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        
        #--self.loss_names = ['D_A','G_A','cycle_A','idt_A','D_B','G_B','cycle_B','idt_B','S_A','S_B','M_A','M_B','E_A','E_B','X_A','X_B']

        """Calculate the loss for encoder"""
        gamma_identity = self.opt.gamma_identity 
        beta  = self.opt.beta 
        gamma = self.opt.gamma * gamma_identity 
        alpha = self.opt.alpha 
        delta = self.opt.delta 

        if self.opt.gan_mode == 'relative':
          self.loss_X_A=self.criterionGAN(self.netD_A(self.real_B),self.netD_A(self.X_BB),False)
          self.loss_X_B=self.criterionGAN(self.netD_B(self.real_A),self.netD_B(self.X_AA),False)
        else:
          # GAN loss D_A(X_BB), D_B(X_AA)
          self.loss_X_A = self.criterionGAN(self.netD_A(self.X_BB), True)  
          self.loss_X_B = self.criterionGAN(self.netD_B(self.X_AA), True)  

        if self.opt.no_AA_BB:
           A_enc, A_feat = self.R_A, self.F_A 
           B_enc, B_feat = self.R_B, self.F_B 
        else:
           A_enc, A_feat = self.R_AA, self.F_AA 
           B_enc, B_feat = self.R_BB, self.F_BB 
        
        ## Encoder Loss
        if 'En' in self.opt.loss_type:
            self.loss_E_A = self.criterionENC( A_enc, self.R_AB ) * beta 
            self.loss_E_B = self.criterionENC( B_enc, self.R_BA ) * beta 
        else:
            self.loss_E_A, self.loss_E_B = 0, 0 

        # Combinatorial loss [ME, MF, ME+E, MF+E]
        ## intermidiated feature metric learning
        if not 'M' in self.opt.loss_type:
           self.loss_M_A, self.loss_M_B = 0, 0 
           if 'SF' in self.opt.loss_type:
              self.loss_M_A += self.criterionMT( A_feat, self.F_AB, B_feat ) * alpha 
              self.loss_M_B += self.criterionMT( B_feat, self.F_BA, A_feat ) * alpha 
           if 'SE' in self.opt.loss_type:
              self.loss_M_A += self.criterionMT( A_enc, self.R_AB, B_enc ) * alpha 
              self.loss_M_B += self.criterionMT( B_enc, self.R_BA, A_enc ) * alpha 

        else:
           self.R_AN, self.F_AN  = self.netE( self.real_A_n ) 
           self.R_BN, self.F_BN  = self.netE( self.real_B_n ) 
           self.loss_M_A, self.loss_M_B = 0, 0 
           if 'MF' in self.opt.loss_type: # metric learning for 'F'eature
              self.loss_M_A += self.criterionMT( A_feat, self.F_AB, self.F_AN ) * alpha 
              self.loss_M_B += self.criterionMT( B_feat, self.F_BA, self.F_BN ) * alpha 
           if 'ME' in self.opt.loss_type:
              self.loss_M_A += self.criterionMT( A_enc, self.R_AB, self.R_AN ) * alpha 
              self.loss_M_B += self.criterionMT( B_enc, self.R_BA, self.R_BN ) * alpha 

        if delta > 0:
           if self.opt.toGray:
              if self.opt.aef == 'tanh':
                 A_ref = (A_enc + 1.) / 2. 
                 B_ref = (B_enc + 1.) / 2. 
                 A_tar = (self.R_AB + 1.) / 2. 
                 B_tar = (self.R_BA + 1.) / 2. 
              else:
                 A_ref, B_ref, A_tar, B_tar = A_enc, B_enc, self.R_AB, self.R_BA 
           else:
              A_ref, B_ref, A_tar, B_tar = A_enc, B_enc, self.R_AB, self.R_BA 

           self.loss_SS_A = self.criterionSSIM( A_ref, A_tar ) * delta 
           self.loss_SS_B = self.criterionSSIM( B_ref, B_tar ) * delta 
        else:
           self.loss_SS_A, self.loss_SS_B = 0, 0 

        if gamma_identity > 0:
           if self.opt.style_change:
              A_feat, B_feat = B_feat, A_feat
           self.loss_S_A = self.criterionSTL( self.gram_matrix(self.F_AB),
                                              self.gram_matrix(B_feat) ) * gamma 
           self.loss_S_B = self.criterionSTL( self.gram_matrix(self.F_BA),
                                              self.gram_matrix(A_feat) ) * gamma 
        else:
           self.loss_S_A, self.loss_S_B = 0, 0 

        self.loss_E = 0.5*(self.loss_X_B + self.loss_X_A + self.loss_E_A + self.loss_E_B + self.loss_M_A + self.loss_M_B + self.loss_S_A + self.loss_S_B) 

        self.loss_G += self.loss_E 
        self.loss_G.backward() 

    def gram_matrix(self, input):
        B, C, H, W = input.size()  # a=batch size(=1)
        
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(B * C, H * W)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(B *C *H * W)


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights  called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
