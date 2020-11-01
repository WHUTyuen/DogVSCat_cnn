import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import transforms
class MyDataset(Dataset):

    def __init__(self,path):
        self.path = path
        self.dataset = os.listdir(self.path)
        # self.dataset.sort(key=lambda s: (int(s[0]), int(s[s.index(".") + 1: s.index(".", s.index(".") + 1)])))
        # self.dataset.sort()
        self.dataset.sort(key=lambda x : int(x.split(".")[1]))

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        datapath = self.dataset[index]
        img = Image.open(os.path.join(self.path, datapath))
        data = np.array(img)
        # C H W
        data = torch.Tensor(data).permute(2,0,1) / 255 # 获取数据并归一化
        data = self.normal(data) # 均值方差归一化
        target = torch.Tensor([int(datapath[0:1])])   # 获取标签
        return data, target

    def normal(self, tensor):
        trans = transforms.Normalize((0.4876, 0.4542, 0.4166), (0.2624, 0.2558, 0.2580))
        return trans(tensor)

if __name__ == '__main__':

    mydata = MyDataset(r"D:\images_cat_dog\img")
    # print(mydata[0][0].shape)
    print(mydata.dataset)

    # print(len(mydata))
    # print(mydata[6000])
    #
    # x = mydata[0][0]
    # y = mydata[0][1]
    # print(x,y)
    # x = x.numpy()
    # x = (x + 0.5)*255
    # x = np.array(x, dtype=np.int8)
    # # print(x)
    #
    # #从numpy数组转回图片
    # img = Image.fromarray(x, "RGB")
    # img.show()