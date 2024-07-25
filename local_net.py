#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：torch_code 
@File    ：local_net.py
@IDE     ：PyCharm 
@Author  ：Polaris
@Date    ：2024/6/14 22:40 
@Descripe:网络结构，输入是地图、坐标周围的地图信息，包含了全局信息和局部信息
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=100, is_training=False, dropout_keep_prob=0.5, spatial_squeeze=True):
        super(Net, self).__init__()
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) # BCHW
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#[B, 64, 256, 256]

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#[B, 128, 128, 128]

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#[B, 256, 64, 64]

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#[B, 512, 32, 32]

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#[B, 512, 16, 16]

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv62 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#[B, 512, 8, 8]

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv72 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#[B, 512, 4, 4]

        self.fc8 = nn.Conv2d(512, 2000, kernel_size=4, stride=1, padding=0)#[B, 2000, 1, 1]
        self.fc9 = nn.Conv2d(2000, 2000, kernel_size=1, stride=1, padding=0)#[B, 2000, 1, 1]
        self.fc10 = nn.Conv2d(2000, 2000, kernel_size=1, stride=1, padding=0) #[B, 2000, 1, 1]

        #local_elemenet卷积提取专用
        self.conv8 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) # BCHW[B, 64, 8, 8]
        self.conv82 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)# [B, 128, 8, 8]
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)# [B, 128, 4, 4]

        self.conv9 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)# [B, 256, 4, 4]
        self.conv92 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)# [B, 512, 4, 4]
        self.pool9 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)# [B, 512, 2, 2]

        self.conv10 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)# [B, 1024, 1, 1]
        self.conv10_2 = nn.Conv2d(1024, 2000, kernel_size=1, stride=1, padding=0)# [B, 2000, 1, 1]

        self.conv11 = nn.Conv2d(2000, 2000, kernel_size=1, stride=1, padding=0) #[B, 2000, 1, 1]

        #定义全连接层
        self.fc11 = nn.Linear(4000, 1000)
        self.fc12 = nn.Linear(1000, 32)
        self.fc13 = nn.Linear(32, 32)
        self.fc14 = nn.Linear(32, num_classes)


    def forward(self, x, loc):
        loc = loc.int()
        #卷积网络提取地图信息
        raw_x = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv12(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv22(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv42(x))
        # x = F.relu(self.conv43(x))
        x = self.pool4(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv52(x))
        # x = F.relu(self.conv53(x))
        x = self.pool5(x)

        x = F.relu(self.conv6(x))
        x = F.relu(self.conv62(x))
        x = self.pool6(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv72(x))
        x = self.pool7(x)

        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))

        if self.spatial_squeeze:
            x = x.squeeze(dim=3).squeeze(dim=2)
        #导入正则化后的位置坐标

        inteval = 3 #local_element尺寸是[batch_size, 1, 2*inteval, 2*inteval]
        local_element = raw_x[:, :, loc[0][0]-inteval:loc[0][0]+inteval, loc[0][1]-inteval:loc[0][1]+inteval]


        loc = F.relu(self.conv8(local_element))
        loc = F.relu(self.conv82(loc))
        loc = self.pool8(loc)

        loc = F.relu(self.conv9(loc))
        loc = F.relu(self.conv92(loc))
        loc = self.pool9(loc)
        # print(loc.size())
        loc = F.relu(self.conv10(loc))
        loc = F.relu(self.conv10_2(loc))
        loc = F.relu(self.conv11(loc))
        if self.spatial_squeeze:
            loc = loc.squeeze(dim=3).squeeze(dim=2)
        # print(x.size())
        # print(loc.size())
        #将地图信息和位置坐标关联在一起
        x = torch.cat((x, loc), dim=1)

        x = F.relu(self.fc11(x))
        # x = F.dropout(x, p=self.dropout_keep_prob, training=self.is_training)
        x = F.relu(self.fc12(x))
        x = F.dropout(x, p=self.dropout_keep_prob, training=self.is_training)
        x = F.relu(self.fc13(x))
        x = F.dropout(x, p=self.dropout_keep_prob, training=self.is_training)
        y = self.fc14(x)

        return y
