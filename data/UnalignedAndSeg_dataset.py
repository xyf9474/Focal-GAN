import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torchvision.transforms as transforms
from options.train_options import TrainOptions
from data import create_dataset
from data.base_dataset import BaseDataset, get_transform_img_and_seg
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedAndSegdataset(BaseDataset):
    """
    This dataset class can load unpaired datasets with segmentation.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A','img')  # create a path '/path/to/data/trainA/img'
        self.dir_A_seg = os.path.join(opt.dataroot, opt.phase + 'A','seg')  # create a path '/path/to/data/trainA/seg'

        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B','img')  # create a path '/path/to/data/trainB/img'
        self.dir_B_seg = os.path.join(opt.dataroot, opt.phase + 'B','seg')  # create a path '/path/to/data/trainB/seg'

        self.A_img_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA/img'
        # print(self.A_img_paths)
        self.A_seg_paths = sorted(make_dataset(self.dir_A_seg, opt.max_dataset_size))   # load images from '/path/to/data/trainA/seg'
        self.B_img_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB/img'
        self.B_seg_paths = sorted(make_dataset(self.dir_B_seg, opt.max_dataset_size))    # load images from '/path/to/data/trainB/seg'

        self.A_size = len(self.A_img_paths)  # get the size of dataset A
        self.B_size = len(self.B_img_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform_img_and_seg(self.opt, grayscale=True)
        self.transform_B = get_transform_img_and_seg(self.opt, grayscale=True)

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
        A_path = self.A_img_paths[index % self.A_size]  # make sure index is within then range
        A_seg_path = self.A_seg_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_img_paths[index_B]
        B_seg_path = self.B_seg_paths[index_B]

        A_img,A_img_seg = Image.open(A_path).convert('RGB'),Image.open(A_seg_path).convert('1')
        B_img,B_img_seg = Image.open(B_path).convert('RGB'),Image.open(B_seg_path).convert('1')

        A,A_img_seg_tensor = self.transform_A(A_img,A_img_seg)
        B,B_img_seg_tensor = self.transform_B(B_img,B_img_seg)



        if A.shape[1:] != A_img_seg_tensor.shape[1:]:
            print(A_path,A.shape,'img shape not match',A_seg_path,A_img_seg_tensor.shape)

        if A_path.split('/')[-1] != A_seg_path.split('/')[-1]:
            print(A_path,'path not match',A_seg_path)

        if B.shape[1:] != B_img_seg_tensor.shape[1:]:
            print(B_path,B.shape,'img shape not match',B_seg_path,B_img_seg_tensor.shape)

        if B_path.split('/')[-1] != B_seg_path.split('/')[-1]:
            print(B_path,'not match',B_seg_path)

        return {'A': A, 'B': B, 'A_seg': A_img_seg_tensor ,'B_seg' : B_img_seg_tensor,'A_paths': A_path, 'B_paths': B_path,'A_seg_paths':A_seg_path,'B_seg_paths':B_seg_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def normalization_label(self,origin_tensor):
        zero = torch.zeros_like(origin_tensor)
        one = torch.ones_like(origin_tensor)
        _tensor = torch.where(origin_tensor < 0, zero, origin_tensor)
        _tensor = torch.where(_tensor > 0, one, _tensor)
        return _tensor

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    for i, data in enumerate(dataset):
        print(i)
        # print(data)
        # A = torch.squeeze(data['A_paths'])
        # B = torch.squeeze(data['A_seg_paths'])
        # plt.imshow(A)
        # plt.show()
        # plt.imshow(B)
        # plt.show()
        # print(data)
        break



