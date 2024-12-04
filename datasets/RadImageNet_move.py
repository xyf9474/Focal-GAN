import os
import csv
import shutil
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torchvision import datasets,transforms



def data_move(root_path,target_path,division_num = 1):
    total_num = 0
    root_sub_1 = os.listdir(root_path)
    for sub_1 in root_sub_1:
        if 'abd' in sub_1:
            sub_1_path = os.path.join(root_path,sub_1)
            root_sub_2 = os.listdir(sub_1_path)
            for sub_2 in root_sub_2:
                if sub_2 == 'normal':
                    sub_2_path = os.path.join(sub_1_path,sub_2)
                    pics_list = os.listdir(sub_2_path)
                    pics_list_random = random.sample(pics_list, len(pics_list)//division_num)
                    print('current dir:',sub_2_path)
                    for item in tqdm(pics_list_random):
                        item_path = os.path.join(sub_2_path,item)
                        shutil.copy(item_path, os.path.join(target_path, item))
                        total_num += 1
    print('total pics num:',total_num)

def get_csv(csv_path):
    csv_list = []
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            csv_list.append(row)  # 选择某一列加入到data数组中
    return csv_list,header

def find_non_axis(tensor,num_y_slices,max):
    indicator = 0
    _tensor = torch.squeeze(tensor, dim=0)
    if torch.max(_tensor[:num_y_slices]) < max and torch.max(_tensor[-num_y_slices:]) < max:
        pass

    else:
        indicator = 1
    return indicator


    # max_0 = torch.max(_tensor, 0, keepdim=False)
    # print(max_0)



if __name__ == '__main__':
    ## move to trainA and trainB
    # RadImageNet_path = '/home/ailab404/data/Copy_of_rin2d/radiology_ai'
    # ct_path = os.path.join(RadImageNet_path,'CT')
    # mr_path = os.path.join(RadImageNet_path,'MR')
    # target_ct_path = '/home/ailab404/data/Copy_of_rin2d/radiology_re_abd_only/trainA'
    # target_mr_path = '/home/ailab404/data/Copy_of_rin2d/radiology_re_abd_only/trainB'
    #
    # data_move(ct_path,target_ct_path,1) # ct-abd-normal:69238
    # data_move(mr_path,target_mr_path,1) # mr-abd-normal:72054

    ## use csv to split train and val pics from abd and mriabd respectively and record
    # train_csv_path = '/home/ailab404/data/Copy_of_rin2d/RadiologyAI_train.csv'
    # val_csv_path = '/home/ailab404/data/Copy_of_rin2d/RadiologyAI_val.csv'
    # test_csv_path = '/home/ailab404/data/Copy_of_rin2d/RadiologyAI_test.csv'
    #
    # train_csv_list = []
    # train_csv_list_abd = [['filename','label']]
    # with open(train_csv_path) as csvfile:
    #     csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    #     header = next(csv_reader)        # 读取第一行每一列的标题
    #     for row in csv_reader:  # 将csv 文件中的数据保存到data中
    #         train_csv_list.append(row)  # 选择某一列加入到data数组中
    #
    # for item in tqdm(train_csv_list):
    #     position = item[-1].split('-')[0]
    #     if position == 'abd':
    #         train_csv_list_abd.append(item)

    # with open('/home/ailab404/data/Copy_of_rin2d/radiology_cycleGAN/abd_train.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in train_csv_list_abd:
    #         writer.writerow(row)

    ## use csv to copy abd and mriabd to train,val,test
    # csv_path = '/home/ailab404/data/Copy_of_rin2d/radiology_cycleGAN/mriabd_test.csv'
    # src_path = '/home/ailab404/data/Copy_of_rin2d'
    # target_path = '/home/ailab404/data/Copy_of_rin2d/radiology_cycleGAN/mriabd_test'
    # csv_list,head = get_csv(csv_path)
    # for item in tqdm(csv_list):
    #     item_path = item[0]
    #     src_dir = os.path.join(src_path,item_path)
    #     item_path_sub = item_path.split('/')[3]
    #     target_dir = os.path.join(target_path,item_path_sub)
    #     if not os.path.exists(target_dir):
    #         os.makedirs(target_dir)
    #     shutil.copy(src_dir,target_dir)

    ## Delete non-axis image
    path = '/home/ailab404/data/Copy_of_rin2d/radiology_cycleGAN/mriabd_val'
    dirs = os.listdir(path)
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor()
                                    ])
    for dir in tqdm(dirs):
        dir_path = os.path.join(path,dir)
        items = os.listdir(dir_path)
        for item in items:
            item_path = os.path.join(dir_path,item)
            item_img = Image.open(item_path)
            img_process = transform(item_img)
            indicator = find_non_axis(img_process,10,0.2)
            if indicator == 1:
                os.remove(item_path)
            else:
                pass





