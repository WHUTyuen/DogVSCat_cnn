from torch.utils.data import DataLoader
import torch
from MyDataset import MyDataset
from MyCNN import CNN_Net
import torch.nn as nn
import numpy as np

def train():
    dataset = MyDataset(r"D:\images_cat_dog\img")
    trainloader = DataLoader(dataset, batch_size=300, shuffle=True)
    net = CNN_Net().cuda()
    loss_func = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters())
    for i in range(30):
        print("epochs:{}".format(i))
        for j, (data, target) in enumerate(trainloader):
            data = data.cuda()
            output = net(data)
            target = torch.zeros(target.size(0), 2).scatter(1, target.long(), 1).cuda()
            loss = loss_func(output, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if j % 5 == 0:
                print("{}/{},loss:{}".format(j, len(trainloader), loss.float()))
                predict = torch.argmax(output, dim=1)
                target = target.argmax(dim=1)
                print("正确率:", str(((predict == target).float().mean()*100).item())+"%")

    torch.save(net, "models/cnn_net2.pth")

train()