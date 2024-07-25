from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import time
import numpy as np
# from pykalman import KalmanFilter

import net
import local_net
import input
from arg import parse_args

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")


# 自定义损失函数
class customLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        mse = torch.mean(torch.square(y_true - y_pred))
        smoothing = torch.mean(torch.square(y_pred[1:, :] - y_pred[:-1, :]))
        return mse + smoothing  # 0.1为正则化权重，可以调节


def train_loop(args, dataloader, model, criterion, optimizer):# dataloader2,
    progress_bar = tqdm(range(len(dataloader)), unit='batch', leave=True) #  + len(dataloader2)
    progress_bar.set_description('Progress bar:')
    total_loss = 0.
    model.train()
    step = 0
    for step, (X, Y, Z) in enumerate(dataloader, start=1):
        img, loc, label = X.to(args.device), Y.to(args.device), Z.to(args.device)
        pred = model(img, loc)
        # MSE
        S_loss = criterion(pred, label)
        # RMSE
        loss = torch.sqrt(S_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{total_loss / step :>7f}dB'})
        progress_bar.update(1)
    # for step2, (X, Y, Z) in enumerate(dataloader2, start=step+1):
    #     img, loc, label = X.to(args.device), Y.to(args.device), Z.to(args.device)
    #     pred = model(img, loc)
    #     # MSE
    #     S_loss = criterion(pred, label)
    #     # RMSE
    #     loss = torch.sqrt(S_loss)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     total_loss += loss.item()
    #     progress_bar.set_postfix({'loss': f'{total_loss / step :>7f}dB'})
    #     progress_bar.update(1)

    progress_bar.close()
    return total_loss / (len(dataloader)) # +len(dataloader2)


def test_loop(args, dataloader, model, criterion, mode='Test'):
    assert mode in ['Valid', 'Test']
    total_loss = 0
    diff = []
    model.eval()
    prediction = []
    labels = []
    with torch.no_grad():
        for (X, Y, Z) in dataloader:
            img, loc, label = X.to(args.device), Y.to(args.device), Z.to(args.device)
            pred = model(img, loc)
            a = pred.view(-1, 1)
            a = a.to('cpu')
            a = a.tolist()
            prediction.append(a)
            b = label.view(-1, 1)
            b = b.to('cpu')
            b = b.tolist()
            labels.append(b)

            # labels.append(label.cpu())
            diff.extend((pred - label).cpu())
            # MSE
            # S_loss = torch.mean(torch.square(pred-label))
            S_loss = criterion(pred, label)
            # RMSE
            loss = torch.sqrt(S_loss)
            total_loss += loss
    # diff = np.array(diff).reshape(-1)
    total_loss /= len(dataloader)

    if args.do_test:
        # # 初始化卡尔曼滤波器
        # predictions = [element for sublist in prediction for element in sublist]
        # labels = [element for sublist in labels for element in sublist]
        #
        # # predictions = [[pre] for pre in prediction]
        # initial_state_mean = predictions[0]
        # initial_state_covariance = 1.0
        # transition_matrix = 1.0
        # observation_matrix = 1.0
        # transition_covariance = 0.1
        # observation_covariance = 0.1
        #
        # kf = KalmanFilter(
        #     initial_state_mean=initial_state_mean,
        #     initial_state_covariance=initial_state_covariance,
        #     transition_matrices=transition_matrix,
        #     observation_matrices=observation_matrix,
        #     transition_covariance=transition_covariance,
        #     observation_covariance=observation_covariance
        # )
        #
        # # 使用卡尔曼滤波器校正预测结果
        # state_means, state_covariances = kf.filter(predictions)
        #
        # plt.figure(1)
        # plt.plot(range(1, len(diff) + 1), diff)
        # plt.xlabel('num')
        # plt.ylabel('difference/dB')
        # plt.title('the difference of predictions and labels')
        # plt.show()

        plt.figure(2)
        abs_diff = [abs(x) for x in diff]
        sorted_data = np.sort(abs_diff)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        print(f'50%点为{sorted_data[round(len(sorted_data) * 0.5)].item():.2f}dB')
        print(f'80%点为{sorted_data[round(len(sorted_data) * 0.8)].item():.2f}dB')
        print(f'95%点为{sorted_data[round(len(sorted_data) * 0.95)].item():.2f}dB')
        # 绘制CDF曲线
        plt.plot(sorted_data, yvals, )
        plt.xlabel('dB')
        plt.ylabel('the CDF of difference')
        plt.show()

        # plt.figure(3)
        # plt.plot(range(1, len(state_means)+1), state_means-labels)
        # plt.xlabel('num')
        # plt.ylabel('difference/dB')
        # plt.title('the difference after kalman filter')
        # plt.show()

    return total_loss


def train(args, train_dataset, dev_dataset, model):# train2_dataset
    """ Train the model """
    train_dataloader = input.get_dataLoader(args, train_dataset, shuffle=False)
    # train2_dataloader = input.get_dataLoader(args, train2_dataset, shuffle=False)
    dev_dataloader = input.get_dataLoader(args, dev_dataset, shuffle=False)
    t_total = (len(train_dataloader)) * args.num_train_epochs # +len(train2_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 定义学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=args.learning_rate_decay_factor)
    # criterion = nn.MSELoss()
    criterion = customLoss()
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataloader)}") # +len(train2_dataloader)
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    best_loss = 10.
    train_loss_per_epoch = []
    test_loss_per_epoch = []

    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}\n-------------------------------")
        model.train()# 更改self.training为True
        train_loss = train_loop(args, train_dataloader, model, criterion, optimizer) # , train2_dataloader
        if (epoch + 1) % 2 == 0:
            learning_rate *= args.learning_rate_decay_factor
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_loss_per_epoch.append(train_loss)

        model.eval()# 更改self.training为False
        test_loss = test_loop(args, dev_dataloader, model, criterion, mode='Valid')
        a = test_loss.cpu()
        test_loss_per_epoch.append(a.detach().numpy())
        logger.info(f"Dev loss: {test_loss :>0.2f}dB")
        if test_loss < best_loss:
            best_loss = test_loss
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch + 1}_dev_loss_{best_loss :0.1f}dB_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    filename = time.strftime('time_%Y-%m-%d_%H:%M:%S', time.localtime()) + f'_dev_loss_{test_loss :0.1f}dB_weights.bin'
    torch.save(model.state_dict(), os.path.join(args.output_dir, filename))
    plt.figure(1)
    plt.plot(range(1, args.num_train_epochs + 1), train_loss_per_epoch, color='b', label='train')
    plt.plot(range(1, args.num_train_epochs + 1), test_loss_per_epoch, '--', color='g', label='test')
    plt.xlabel('epoch')
    plt.ylabel('the RMSE of PL/dB')
    plt.legend()
    plt.show()
    logger.info("train Done!")
    # return save_weight


def test(args, test_dataset, model, save_weights):
    test_dataloader = input.get_dataLoader(args, test_dataset, shuffle=False)
    # save_weight = save_weights[-1]
    save_weight = save_weights
    logger.info('***** Running testing *****')
    logger.info(f'loading weights: {save_weight}...')
    model.load_state_dict(torch.load(save_weight))
    criterion = nn.MSELoss()
    test_loss = test_loop(args, test_dataloader, model, criterion, mode="Test")
    logger.info(f"Test Loss: {(test_loss):>0.2f}dB")


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print(f'Using {args.device} device, n_gpu: {args.n_gpu}')

    # model = local_net.Net(num_classes=1, is_training=False, dropout_keep_prob=0.5, spatial_squeeze=True) #全局地图和该点附近的局部地图信息
    model = net.Net(num_classes=1, dropout_keep_prob=0.00, spatial_squeeze=True)  # 全局地图和该点归一化后的坐标
    model.to(args.device)

    if args.do_train:
        # train_dataset, train2_dataset, valid_dataset = input.provide_data(args)
        train_dataset, valid_dataset = input.provide_data(args)
        train(args, train_dataset, valid_dataset, model)

    if args.do_test:
        model.eval()
        test_dataset = input.provide_data(args)

        # 加载最新创建的模型，用于评估测试集
        testdir = './result/'
        # 列出目录下所有的文件
        list = os.listdir(testdir)
        # 对文件修改时间进行升序排列
        list.sort(key=lambda fn: os.path.getmtime(testdir + fn))
        # 获取最新修改时间的文件
        filetime = datetime.fromtimestamp(os.path.getmtime(testdir + list[-1]))
        # 获取文件所在目录
        filepath = os.path.join(testdir, list[-1])
        print('加载的模型文件名为%s' % filepath)
        save_weights = filepath
        test(args, test_dataset, model, save_weights)

# #超参数设置
# batch_size = 16
# shuffle = True
# num_epochs = 20
# learning_rate = 0.001
# learning_rate_decay_factor = 0.95
# num_epochs_per_decay = 2
# is_training = False
# fine_tuning = False
# online_test = True
# allow_soft_placement = True
# log_device_placement = False

# datapath = '/home/root888/LC_low_attitude/data_0602'
# # datapath = 'D:\\北邮项目\\研究生毕设相关\\bishe_codes\\Generate_datas\\data_0602'
# if not os.path.isabs(train_dir):
#     raise ValueError('You must assign absolute path for --train_dir')

# train, validation, test = input.provide_data(datapath)
# dimensionality_train = train.images.shape
# num_train_samples = dimensionality_train[0]
# num_channels = dimensionality_train[1]
# height = dimensionality_train[2]
# width = dimensionality_train[3]
# # train, validation, test = torch.Tensor(train), torch.Tensor(validation), torch.Tensor(test)

# train_Dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
# validation_Dataloader = DataLoader(validation, batch_size=batch_size, shuffle=shuffle)
# test_Dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # model = net.Net(num_classes=1, is_training=False, dropout_keep_prob=0.5, spatial_squeeze=True)
# model = local_net.Net(num_classes=1, is_training=False, dropout_keep_prob=0.5, spatial_squeeze=True)
# model.to(device)
# print(model)
# summary(model, inputsize=(batch_size, 1, 512, 512))
# criterion = nn.MSELoss()

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#     # with tqdm(train_Dataloader, unit='batch', leave=True, desc=f'Epoch:{epoch+1}/{num_epochs}') as loop:
#     loop = tqdm(train_Dataloader, unit='batch', leave=True, desc=f'Epoch:{epoch + 1}/{num_epochs}')
#     l = 0
#     for images, locs, labels in loop:
#         images = images.to(device)
#         locs = locs.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images, locs)
#         S_loss = criterion(outputs, labels)
#         loss = torch.sqrt(S_loss)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         l += 1
#         loop.set_postfix({'loss':'{0:1.5f}'.format(train_loss/l)})
#     loop.close()
#     time.sleep(1)
#     model.eval()
#     test_loss = 0.0
#     with torch.no_grad():
#         for images, locs, labels in test_Dataloader:
#             images = images.to(device)
#             locs = locs.to(device)
#             labels = labels.to(device)
#             outputs = model(images, locs)
#             loss = torch.sqrt(criterion(outputs, labels))
#             test_loss += loss.item()
#     print('Test Loss: {:.4f}'.format(test_loss / len(test_Dataloader)))
#     #学习率衰减
#     if epoch % num_epochs_per_decay == 0:
#         learning_rate *= learning_rate_decay_factor
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# result = []
# device2 = torch.device('cpu')
# test2_Dataloader = DataLoader(test, batch_size=1, shuffle=shuffle)
# for images, locs, labels in test2_Dataloader:
#     images = images.to(device)
#     locs = locs.to(device)
#     labels = labels.to(device)
#     outputs = model(images, locs)
#     diff = outputs - labels
#     # diff = diff.cpu()
#     result.extend(diff.tolist())
# plt.plot(range(len(result)), result)
# plt.show()


# torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pth'))
# print('模型已保存')
