#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-1-4
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
#下载数据，标准化并转化
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])
                                ])
mnist = datasets.MNIST(root='/media/yu/Document/data',
                       train=True,
                       transform=transform,
                       )
data_loader = torch.utils.data.DataLoader(mnist,
                                          batch_size=128,
                                          shuffle=True)

#定义模型
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        #编码器网络定义 28*28->128->64->12->3
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        #解码器网络定义 3->12->64->128->28*28
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh() #图片transforms之后标准化在-1~1之间，tanh函数满足这个定义
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

net = autoencoder()
# x = Variable(torch.randn(1, 28*28)) #batch=1
# code ,_ = net(x)
# print(code.shape) #输出为三维的

loss_fun = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def to_image(x):
    '''
    此函数将最后结果转化为图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x

for e in range(1000):
    for img, _ in data_loader:
        img = img.view(img.shape[0], -1)
        img = Variable(img)
        #前向传播 反向传播 优化
        _,output = net(img)
        loss = loss_fun(output, img) / img.shape[0] #平均
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 20 == 0:  # 每 20 次，将生成的图片保存一下
        print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data[0]))
        pic = to_image(output.cpu().data)
        if not os.path.exists('./png_autoencoder'):
            os.mkdir('./png_autoencoder')
        save_image(pic, './png_autoencoder/image_{}.png'.format(e + 1))