
import PIL.Image as im
import torch
import numpy as np
import os
from tkinter import *
from torchvision.transforms import transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def test():

    net = torch.load("models/net2.pth")
    img = im.open("testimg/cat2.jpeg")
    # resize成100*100的尺寸
    img = img.resize((100, 100))

    # 图片转成
    img = np.array(img)
    print(img.shape)

    img = torch.tensor(img, dtype=torch.float32)
    img = img/255 - 0.5
    print(img)
    # img = img.unsqueeze(0)
    print(img.shape)
    out = net(img)
    print(out)
    max = out.detach().numpy().max()
    print(max)

    res = "猫" if out.argmax(dim=1).item() == 0 else "狗"

    top = Tk()
    width = 300
    height = 50
    screenwidth = top.winfo_screenwidth()
    screenheight = top.winfo_screenheight()

    alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    top.geometry(alignstr)
    top.title("AI预测结果:")
    w = Label(top, text="这是一只{},\n概率为:{}".format(res,str(round(max*100,2))+"%"))
    w.pack()

    # 设置窗口大小不可改变
    # top.resizable(width=False, height=False)
    top.mainloop()

def normal(tensor):
    trans = transforms.Normalize((0.4876,  0.4542, 0.4166), (0.2624, 0.2558, 0.2580))
    return trans(tensor)

def test_cnn():
    net = torch.load("models/cnn_net.pth")
    img = im.open("bg_pic/cat/pic11.jpg")
    # resize成100*100的尺寸
    img = img.resize((100, 100))
    # 图片转成
    img = np.array(img)
    # print(img)
    img = torch.Tensor(img).permute(2,0,1) / 255
    img = normal(img)
    img = img.unsqueeze(0)
    out = net(img.cuda())
    # print(out)
    max = out.detach().cpu().numpy().max()
    # print(max)

    res = "猫" if out.argmax(dim=1).item() == 0 else "狗"
    print(res,"精度:", round(float(max)*100,2))
    # top = Tk()
    # width = 300
    # height = 50
    # screenwidth = top.winfo_screenwidth()
    # screenheight = top.winfo_screenheight()
    #
    # alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    # top.geometry(alignstr)
    # top.title("AI预测结果:")
    # w = Label(top, text="这是一只{},\n概率为:{}".format(res, str(round(max * 100, 2)) + "%"))
    # w.pack()
    # # 设置窗口大小不可改变
    # # top.resizable(width=False, height=False)
    # top.mainloop()

if __name__ == '__main__':
    test_cnn()
    # t.train()

