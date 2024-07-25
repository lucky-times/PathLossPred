import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=1, dropout_keep_prob=0.5, spatial_squeeze=True):
        super(Net, self).__init__()
        # self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv62 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv72 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc8 = nn.Conv2d(512, 2000, kernel_size=4, stride=1, padding=0)
        self.fc9 = nn.Conv2d(2000, 2000, kernel_size=1, stride=1, padding=0)
        self.fc10 = nn.Conv2d(2000, 2000, kernel_size=1, stride=1, padding=0)

        self.fc11 = nn.Linear(4000, 1000)
        self.fc12 = nn.Linear(1000, 32)
        self.fc13 = nn.Linear(32, 32)
        self.fc14 = nn.Linear(32, num_classes)

        self.fc21 = nn.Linear(2, 1000)
        self.fc22 = nn.Linear(1000, 2000)
        self.fc23 = nn.Linear(2000, 2000)

        self.drop_out1 = nn.Dropout(p=self.dropout_keep_prob)
        self.drop_out2 = nn.Dropout(p=self.dropout_keep_prob)


    def forward(self, x, loc):
        #卷积网络提取地图信息
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
        loc = F.relu(self.fc21(loc))
        loc = F.relu(self.fc22(loc))
        loc = F.relu(self.fc23(loc))

        #将地图信息和位置坐标关联在一起
        x = torch.cat((x, loc), dim=1)

        x = F.relu(self.fc11(x))

        # x = F.dropout(x, p=self.dropout_keep_prob, training=self.training, inplace=False)
        x = F.relu(self.fc12(x))
        # x = self.drop_out1(x)
        # x = F.dropout(x, p=self.dropout_keep_prob, training=self.training, inplace=False)
        x = F.relu(self.fc13(x))
        x = F.dropout(x, p=self.dropout_keep_prob, training=self.training, inplace=False)
        # x = self.drop_out2(x)
        y = self.fc14(x)

        return y
