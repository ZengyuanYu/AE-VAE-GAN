#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-1-17
import torch.nn as nn

class NetG(nn.Module):
    """
    定义生成器G
    """
    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = opt.ngf #生成器feature map数

        self.main = nn.Sequential(
            #输出一个nz维度的噪声，我们定义它为1*1*nz的feature map
            nn.ConvTranspose2d(opt.nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            #此时输出的形状为（ngf*8）*4*4

            nn.ConvTranspose2d(ngf*8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 此时输出的形状为（ngf*4）*8*8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 此时输出的形状为（ngf*2）*16*16

            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # 此时输出的形状为（ngf*8）*32*32

            nn.ConvTranspose2d(ngf * 1, 3, 5, 3, 1, bias=False),
            nn.Tanh() #控制输出范围在-1~1
            # 此时输出形状为 3*96*96
        )
    def forward(self, input):
        return self.main(input)

class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            #输入3*96*96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #输出 ndf*32*32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #输出 ndf*2*16*16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #输出 ndf*4*8*8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 ndf*8*4*4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() #输出0~1之间的概率值
        )
    def forward(self, input):
        return self.main(input).view(-1)