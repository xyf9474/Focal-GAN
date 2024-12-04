import os
import numpy as np
import torch
import torch.nn as nn
import itertools
from PIL import Image
import matplotlib.pyplot as plt
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.loss_functions import BinaryDiceLoss

class SegCycleGANModel(BaseModel):
    """
    This class implements the SegCycleGAN model, for learning image-to-image translation and segmentation without paired data.

    The model training requires '--dataset_mode UnalignedAndSeg' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
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
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        Segmentation loss : Dice Loss
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the SegCycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.end_epochs = self.opt.n_epochs + self.opt.n_epochs_decay
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'Seg_A_real','Seg_A_fake', 'D_B', 'G_B', 'cycle_B', 'idt_B','Seg_B_real','Seg_B_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'seg_A' ,'real_A_seg', 'fake_B_seg', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'seg_B' ,'real_B_seg', 'fake_A_seg', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'Seg_A', 'Seg_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netSeg_A = networks.define_G(input_nc=1,output_nc=1,ngf=64,netG='unet_256',norm='instance',
                                          use_dropout=True,init_type='normal',init_gain=0.02, gpu_ids=self.gpu_ids,
                                          activation_function=nn.Sigmoid())
        self.netSeg_B = networks.define_G(input_nc=1,output_nc=1,ngf=64,netG='unet_256',norm='instance',
                                          use_dropout=True,init_type='normal',init_gain=0.02, gpu_ids=self.gpu_ids,
                                          activation_function=nn.Sigmoid())

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
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSeg_dice = BinaryDiceLoss()
            self.criterionSeg_bce = torch.nn.BCELoss() # model have already use sigmoid

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Seg = torch.optim.Adam(itertools.chain(self.netSeg_A.parameters(), self.netSeg_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_Seg)

    def set_input(self, input, epoch):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
            epoch

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.this_epoch = epoch
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.seg_A = input['A_seg' if AtoB else 'B_seg'].to(self.device)
        # image = Image.fromarray(np.uint8(torch.squeeze(self.real_A_seg.cpu()).numpy()))
        # plt.figure("Image")  # 图像窗口名称
        # plt.imshow(image)
        # plt.axis('on')  # 关掉坐标轴为 off
        # plt.title('image')  # 图像题目
        #
        # # 必须有这个，要不然无法显示
        # plt.show()
        self.seg_B = input['B_seg' if AtoB else 'A_seg'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.seg_paths = input['A_seg_paths' if AtoB else 'B_seg_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_B_seg = self.netSeg_A(self.fake_B) # Seg_A(G_A(A))
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.fake_A_seg = self.netSeg_B(self.fake_A) # Seg_B(G_B(B))
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_Seg_basic(self,netSeg, AorB):
        # seg_fake = netSeg(fake_img.detach())

        if AorB == 'A':
            seg_real = netSeg(self.real_B)
            real_label = self.seg_B
            # fake_label = self.seg_A
            lambda_dice = self.opt.lambda_dice_seg_A # 0.4

        elif AorB == 'B':
            seg_real = netSeg(self.real_A)
            real_label = self.seg_A
            # fake_label = self.seg_B
            lambda_dice = self.opt.lambda_dice_seg_B # 0.7

        loss_S_bce_real = self.criterionSeg_bce(seg_real,real_label)
        # loss_S_bce_fake = self.criterionSeg_bce(seg_fake,fake_label)
        loss_S_dice_real = self.criterionSeg_dice(seg_real,real_label)
        # loss_S_dice_fake = self.criterionSeg_dice(seg_fake,fake_label)

        # loss_S_fake = loss_S_bce_fake*(1.0-lambda_dice) + loss_S_dice_fake * lambda_dice
        loss_S_real = loss_S_bce_real*(1.0-lambda_dice) + loss_S_dice_real * lambda_dice
        # loss_S = (loss_S_fake * min(self.this_epoch / self.opt.n_epochs,1.0) + loss_S_real) * 0.5
        loss_S_real.backward()

        if AorB == 'A':
            # self.fake_B_seg = torch.where(seg_fake>0.5,torch.ones_like(seg_fake),torch.zeros_like(seg_fake))
            self.real_B_seg = torch.where(seg_real>0.5,torch.ones_like(seg_real),torch.zeros_like(seg_real))
        elif AorB == 'B':
            # self.fake_A_seg = torch.where(seg_fake>0.5,torch.ones_like(seg_fake),torch.zeros_like(seg_fake))
            self.real_A_seg = torch.where(seg_real>0.5,torch.ones_like(seg_real),torch.zeros_like(seg_real))

        return loss_S_real

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_and_Seg_A(self):
        """Calculate GAN loss for discriminator D_A and S_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_Seg_A_real = self.backward_Seg_basic(self.netSeg_A, AorB='A')

    def backward_D_and_Seg_B(self):
        """Calculate GAN loss for discriminator D_B and S_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_Seg_B_real = self.backward_Seg_basic(self.netSeg_B, AorB='B')

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # if self.this_epoch >= 100:
        #     lambda_seg = 0.5
        # else:
        #     lambda_seg = 0
        lambda_seg = (self.opt.lambda_seg_end-self.opt.lambda_seg_start)*min((self.this_epoch-self.opt.epoch_count)/self.opt.n_epochs,1.0)+self.opt.lambda_seg_start
        # print(lambda_seg)

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_Seg_A_fake = (self.criterionSeg_bce(self.fake_B_seg,self.seg_A) * (1-self.opt.lambda_dice_seg_A) +
                         self.criterionSeg_dice(self.fake_B_seg,self.seg_A) * self.opt.lambda_dice_seg_A) * lambda_seg
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_Seg_B_fake = (self.criterionSeg_bce(self.fake_A_seg,self.seg_B) * (1-self.opt.lambda_dice_seg_B) +
                         self.criterionSeg_dice(self.fake_A_seg, self.seg_B) * self.opt.lambda_dice_seg_B) * lambda_seg
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + \
                      self.loss_Seg_A_fake + self.loss_Seg_B_fake
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B,self.netSeg_A,self.netSeg_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A,Seg_A and D_B,Seg_B
        self.set_requires_grad([self.netD_A, self.netD_B,self.netSeg_A,self.netSeg_B], True)

        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.optimizer_Seg.zero_grad()   # set S_A and S_B's gradients to zero

        self.backward_D_and_Seg_A()      # calculate gradients for D_A,Seg_A
        self.backward_D_and_Seg_B()      # calculate graidents for D_B,Seg_B

        self.optimizer_D.step()  # update D_A and D_B's weights
        self.optimizer_Seg.step()  # update D_A and D_B's weights

