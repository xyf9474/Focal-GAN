# ImageFolder 函数是一个用来加载数据集的函数, 该数据集可以为自定义的数据集
# 自定义的数据集中一共有多少个类, 就应创建多少个文件夹, 其中各个类的文件夹的名字应为类名, 将对应类的图片保存在对应的文件夹中


from torchvision import datasets,transforms
import torch

transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize([256, 256]),
                                transforms.ToTensor()
                                ])



root_train = '/home/ailab404/data/Copy_of_rin2d/radiology_cycleGAN/abd_train'
root_val = '/home/ailab404/data/Copy_of_rin2d/radiology_cycleGAN/abd_val'
dataset = datasets.ImageFolder(root_train,transform=transform)
print(dataset)
# print(dataset.classes)
# print(dataset.class_to_idx)
# # dataset[0] 表示取第一个训练样本，即(path， class_index)。
# print(dataset[0][0].shape)
# data = dataset[0][0][0]
# print(torch.max(data))
# print(torch.min(data))


