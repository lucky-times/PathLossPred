import sys

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np



class DATA_OBJECT(Dataset):
    def __init__(self, data_file, arg):
        self.images, self.locations, self.labels = self.load_data(data_file, arg)
               

    def load_data(self, data_file, arg):
        imgs_file = os.path.join(data_file, 'maps')
        locs_file = os.path.join(data_file, 'locations')
        labels_file = os.path.join(data_file, 'labels')

        num = len(os.listdir(imgs_file))
        if os.path.exists(os.path.join(labels_file, 'ignore.txt')):
            exclude_indices = np.loadtxt(os.path.join(labels_file, 'ignore.txt'))
            include_indices = [i for i in range(1, num+1) if i not in exclude_indices]
        else:
            include_indices = [i for i in range(1, num+1)]
        valid_file_num = len(include_indices)
        # num = num // 5
        dataset_name = os.path.split('/')[-2]
        print(f'数据集{dataset_name}包含文件数目：{valid_file_num}')

        images = torch.empty([valid_file_num, 1, arg.map_size, arg.map_size]) #BCHW
        labels = torch.empty([valid_file_num, arg.output_numclass])
        #加载归一化的位置坐标
        loc = []
        for i in range(0, valid_file_num):
            p = "%04d" % (include_indices[i])
            filen = os.path.join(locs_file, str(p) + '.txt')
            pos = np.loadtxt(filen)
            loc.append(pos)
        loc = np.array(loc)
        loc = loc.reshape(-1, 2)
        # locations = torch.Tensor(loc)
        locations = self.normalize_columns(torch.Tensor(loc))              #正则化loc坐标


        for i in range(0, valid_file_num):
            #加载地图信息
            p = "%04d" % include_indices[i]
            filen = os.path.join(imgs_file, str(p) + '.txt')
            img = np.loadtxt(filen)
            img = self.normalize(torch.tensor(img))
            images[i, 0, :, :] = img.reshape(arg.map_size, arg.map_size)
            #加载标签值
            filen = os.path.join(labels_file, str(p) + '.txt')
            line = np.loadtxt(filen)
            labels[i, :] = torch.Tensor(line)                                             #导入标签值
        return images, locations, labels
    #按最大最小值 归一化地图信息
    def normalize(self, matrix):
        min_value = torch.min(matrix)
        max_value = torch.max(matrix)
        normalized_matrix = (matrix - min_value) / (max_value - min_value)
        return normalized_matrix
    
    #归一化位置坐标范围，按照列，横纵坐标
    def normalize_columns(self, matrix):
        min_values = torch.min(matrix, dim=0)[0]
        max_values = torch.max(matrix, dim=0)[0]
        normalized_matrix = (matrix - min_values) / (max_values - min_values)
        return normalized_matrix

    def __len__(self):
        # print('形状是：%d' % self.images.shape[0])
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        location = self.locations[idx]
        label = self.labels[idx]
        return image, location, label



def provide_data(arg):
    if arg.do_train:
        train_dir = arg.train_file
        test_dir = arg.test_file
        # 两个训练集
        train = DATA_OBJECT(train_dir, arg)
        # train2 = DATA_OBJECT(train_dir+'2', arg)
        validation = DATA_OBJECT(test_dir, arg)
        return train, validation
    if arg.do_test:
        test_dir = arg.test_file
        test_dir = DATA_OBJECT(test_dir, arg)
        return test_dir

    return 0

def get_dataLoader(args, dataset, batch_size=None, shuffle=False):
    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle,
        drop_last=True
    )