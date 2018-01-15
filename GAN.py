#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-1-14
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0,1)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

#图片预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),
                         std=(0.5,0.5,0.5))
])

#MNIST数据集
mnist = datasets.MNIST(root='./data',
                       train=True,
                       transform=transform,
                       download=True)
#数据加载
data_loder = torch.utils.data.DataLoader(dataset=mnist,
                                         batch_size=100,
                                         shuffle=True)

#Discrimination
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

#Generator
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784),
    nn.Tanh()
)

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

#损失函数和优化
loss_fun = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

#traning
for epoch in range(200):
    for i, (images, _) in enumerate(data_loder):
        #建立最小批数据
        batch_size = images.size(0)
        images = to_var(images.view(batch_size, -1))

        #创建labels在后面计算loss的时候使用
        real_labels = to_var(torch.ones(batch_size))
        fake_labels = to_var(torch.zeros(batch_size))

        #==================训练Discrimination=================#
        #用真是图片计算loss
        #当real_labels=1时loss总是0
        outputs = D(images)
        d_loss_real = loss_fun(outputs, real_labels)
        real_score = outputs

        #用假的图片来计算loss
        #
        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = loss_fun(outputs, fake_labels)
        fake_score = outputs

        #BP+优化
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        #================训练Generator=========================#
        #计算假图片的loss
        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)

        #训练G头最大化log(D(G(z)))代替最小化log(1-D(G(z)))
        g_loss = loss_fun(outputs, real_labels)

        #BP+优化
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if(i+1) % 300 == 0:
            print('Epoch [%d/%d], step[%d/%d], d_loss:%.4f,'
                  'g_loss:%.4f, D(x):%.2f, D(G(z)):%.2f'
                  %(epoch, 200, i+1, 600, d_loss.data[0], g_loss.data[0],
                    real_score.data.mean(), fake_score.data.mean()))
    #存储真实图片
    if(epoch+1) == 1:
        images = images.view(fake_images.size(0), 1, 28, 28)
        save_image(denorm(images.data), './data/GAN/real_images.png')

    #存储生成的图片
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), './data/GAN/fake_images-%d.png'%(epoch+1))
#存储训练好的参数
torch.save(G.state_dict(), './data/GAN/generator.pkl')
torch.save(D.state_dict(), './data/GAN/discriminator.pkl')