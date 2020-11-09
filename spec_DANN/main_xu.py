import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.model_selection import train_test_split
import collections
from functions import training, validation, testing
from model_spec import DANNSpect


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

torch.cuda.set_device(0)
seed = 0
torch.manual_seed(seed)

train_dataset = np.load('formatted_datasets/saved_total_users_train_dataset_xu_7.npy',
                  encoding='bytes', allow_pickle=True)
test_dataset = np.load('formatted_datasets/saved_total_users_test_dataset_xu_7.npy',
                  encoding='bytes', allow_pickle=True)

total_train_data, total_train_labels = train_dataset  # 二者都为(11, )，其中total_data 内部元素为<np.ndarry> 而total_labels内部元素为<list>
total_test_data, total_test_labels = test_dataset

source_user_index = [x for x in range(len(total_train_data) - 1)]
target_user_index = len(total_train_data) - 1

source_train_data = []
source_train_label = []
target_train_data = []
target_train_label = []

source_valid_test_data = []
source_valid_test_label = []
target_valid_test_data = []
target_valid_test_label = []

source_valid_data = []
source_valid_label = []
source_test_data = []
source_test_label = []
target_valid_data = []
target_valid_label = []
target_test_data = []
target_test_label = []

# 按index作leave-one-subject-out 划分
for index in range(11):     # index: NO.1-10 subjects 作为source subject
    if index in source_user_index:
        source_train_data.extend(total_train_data[index])           # <list, np.ndarray> (55805, (4, 8, 14))
        source_train_label.extend(total_train_labels[index])        # <list, int> (55805, (1, ))
        source_valid_test_data.extend(total_test_data[index])       # <list, np.ndarray> (342263, (4, 8, 14))
        source_valid_test_label.extend(total_test_labels[index])    # <list, int> (342263, (1, ))

    elif index == target_user_index:        # NO.11 Subject 作为 target subject
        target_train_data.extend(total_train_data[index])           # <list, np.ndarray> (4454, (4, 8, 14))
        target_train_label.extend(total_train_labels[index])        # <list, int> (4454, (1, ))
        target_valid_test_data.extend(total_test_data[index])       # <list, np.ndarray> (25149, (4, 8, 14))
        target_valid_test_label.extend(total_test_labels[index])    # <list, int> (25149, (1, ))

# 各个子数据集中样本分布情况
c_strain = collections.Counter(source_train_label)
print(c_strain)
c_stest = collections.Counter(source_valid_test_label)
print(c_stest)
c_ttrain = collections.Counter(target_train_label)
print(c_ttrain)
c_ttest = collections.Counter(target_valid_test_label)
print(c_ttest)

# list to array
source_train_data = np.array(source_train_data, dtype=np.float32)                 # (55805, 4, 8, 14)
source_train_label = np.array(source_train_label, dtype=np.int64)               # (55805, )
source_valid_test_data = np.array(source_valid_test_data, dtype=np.float32)       # (342263, 4, 8, 14)
source_valid_test_label = np.array(source_valid_test_label, dtype=np.int64)     # (342263, )

# 扩展 target_train 数据， 使其与source_train 处于同一量级
target_train_data = np.array(target_train_data, dtype=np.float32)                 # (4454, 4, 8, 14)
target_train_concat = target_train_data
for i in range(12):
    target_train_concat = np.concatenate((target_train_concat, target_train_data), axis=0)
target_train_data = target_train_concat
# print(source_train_data.shape)
# print(target_train_data.shape)
# print(target_train_concat.shape)
# time.sleep(100)

# 扩展 target_train 数据， 使其与source_train 处于同一量级
target_train_label = np.array(target_train_label, dtype=np.int64)               # (4454, )
target_train_concat = target_train_label
for i in range(12):
    target_train_concat = np.concatenate((target_train_concat, target_train_label), axis=0)
target_train_label = target_train_concat

target_valid_test_data = np.array(target_valid_test_data, dtype=np.float32)       # (25149, 4, 8, 14)
target_valid_test_label = np.array(target_valid_test_label, dtype=np.int64)     # (25149, )

# split valid/test set -> 10 samples for validation and 20 samples for testing
source_valid_data,  source_test_data, source_valid_label, source_test_label \
    = train_test_split(source_valid_test_data, source_valid_test_label, test_size=0.66, random_state=0)
target_valid_data,  target_test_data, target_valid_label, target_test_label \
    = train_test_split(target_valid_test_data, target_valid_test_label, test_size=0.66, random_state=0)

# numpy to tensor
source_train = TensorDataset(torch.from_numpy(source_train_data), torch.from_numpy(source_train_label))
source_valid = TensorDataset(torch.from_numpy(source_valid_data), torch.from_numpy(source_valid_label))
source_test = TensorDataset(torch.from_numpy(source_test_data), torch.from_numpy(source_test_label))

target_train = TensorDataset(torch.from_numpy(target_train_data), torch.from_numpy(target_train_label))
target_valid = TensorDataset(torch.from_numpy(target_valid_data), torch.from_numpy(target_valid_label))
target_test = TensorDataset(torch.from_numpy(target_test_data), torch.from_numpy(target_test_label))

# form DataLoader
source_train_dataloader = DataLoader(source_train, batch_size=4096, shuffle=True, drop_last=True)
source_valid_dataloader = DataLoader(source_valid, batch_size=1024, shuffle=True, drop_last=True)
source_test_dataloader = DataLoader(source_test, batch_size=4096, shuffle=True, drop_last=True)

target_train_dataloader = DataLoader(target_train, batch_size=4096, shuffle=True, drop_last=True)
target_valid_dataloader = DataLoader(target_valid, batch_size=1024, shuffle=True, drop_last=True)
target_test_dataloader = DataLoader(target_test, batch_size=4096, shuffle=True, drop_last=True)


net = DANNSpect(number_of_class=7).cuda()
for p in net.parameters():
    p.requires_grad = True

precision = 1e-8
optimizer = optim.Adam(net.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5,
                                                 verbose=True, eps=precision)

epoch_num = 200     # 120
patience = 16   # 12
patience_increase = 16  # 12
best_acc = 0
best_loss = float('inf')

for epoch in range(epoch_num):
    epoch_start_time = time.time()

    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

    train_loss, train_acc, alpha, net = training(source_train_dataloader, target_train_dataloader, net, optimizer, epoch_num, epoch=epoch)

    src_valid_D_loss, src_valid_C_loss, src_valid_acc = validation(source_valid_dataloader, net=net, alpha=alpha)
    tag_valid_D_loss, tag_valid_C_loss, tag_valid_acc = validation(target_valid_dataloader, net=net, alpha=alpha)
    tag_test_D_loss, tag_test_C_loss, tag_test_acc = testing(target_test_dataloader, net=net, alpha=alpha)

    print('Training: Loss:{:.4f}  src_Acc:{:.4f}'.format(train_loss, train_acc))
    print('Valid Source:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(src_valid_D_loss, src_valid_C_loss, src_valid_acc))
    print('Valid Target:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(tag_valid_D_loss, tag_valid_C_loss, tag_valid_acc))
    print('Test_t:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(tag_test_D_loss, tag_test_C_loss, tag_test_acc))

    print('epoch time usage: {:.2f}s'.format(time.time() - epoch_start_time))

    val_loss = tag_valid_D_loss + src_valid_C_loss + src_valid_D_loss

    scheduler.step(val_loss)

    if val_loss + precision < best_loss:
        print('New Best Validation Loss: {:.4f}'.format(val_loss))
        best_loss = val_loss
        best_weights = copy.deepcopy(net.state_dict())
        patience = patience_increase + epoch
        print('So Far Patience: ', patience)

    print()


#
# target_data = []
# target_label = []
#
# # 按index作leave-one-subject-out 划分
# for index in range(11):
#     if index in source_user_index:
#         source_data.extend(total_data[index])       # <list> (263382, )
#         source_label.extend(total_labels[index])    # <list> (263382, )
#     elif index == target_user_index:
#         target_data.extend((total_data[index]))     # <list> (20807, )
#         target_label.extend(total_labels[index])    # <list> (20807, )
#
# # list array to numpy array
# source_data = np.array(source_data, dtype=np.float32)
# source_label = np.array(source_label, dtype=np.int64)
# target_data = np.array(target_data, dtype=np.float32)
# target_label = np.array(target_label, dtype=np.int64)
#
# # split train/valid/test set
# source_data_train_valid, source_data_test, source_label_train_valid, source_label_test \
#     = train_test_split(source_data, source_label, train_size=)




# TODO:
#       1. 使用分批加载train/test的形式：
#               即加载每个手势的前5个样本作为训练集，再加载后30个作为测试集
#               然后将测试集中划分30% 作为验证集，用于模型调参











