import torch
from torch.backends import cudnn

# pytorch
print(torch.__version__)

# cuda
print(torch.version.cuda)
torch.zeros(1).cuda() # 这个不报错很关键
print(torch.cuda.is_available())

# cudnn
print(cudnn.version())
print(cudnn.is_available())
