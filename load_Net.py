import torch
import matplotlib as plt
from torch.utils.data import DataLoader

import input
import net

#导入训练好的模型
model_dir = './checkpoints/model.pth'
pre_model = net.Net()
pre_model.load_state_dict(torch.load(model_dir))


batch_size = 16
shuffle = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# datapath = 'D:\\北邮项目\\研究生毕设相关\\bishe_codes\\Generate_datas\\data_0602'
datapath = '/home/root888/LC_low_attitude/data_0602'
train, validation, test = input.provide_data(datapath)
test_Dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
diff = []
for images, locs, labels in test_Dataloader:
    images = images.to(device)
    locs = locs.to(device)
    labels = labels.to(device)
    outputs = pre_model(images, locs)
    diff.append(outputs-labels)

plt.plot(range(len(diff)), diff)
plt.show()

