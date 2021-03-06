"""
使用 ANN 作非TL方法的识别性能对照实验
    ANN:
        Input layer: <input_size: 448>
        Hidden layer: <512>
        Output layerL <output_size: 7>


"""

from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import collections
import copy
import data_process

from model import ANN


# --------------------------------- Data Pre-process --------------------------------------- #


train_dataset = np.load('formatted_datasets/saved_total_users_train_dataset_xu_8_subjects.npy',
                  encoding='bytes', allow_pickle=True)

# 二者都为(11, )，其中total_data 内部元素为<np.ndarry> 而total_labels内部元素为<list>
total_train_data, total_train_labels = train_dataset

# 查看数据包含的样本的标签类别和数量
for index in range(len(total_train_labels)):
    print('Sub{}:'.format(index))                       # subject:  0-10
    c = collections.Counter(total_train_labels[index])  # 1/ 2/ 7/ 8 subject 的样本过少 | 2/ 4/ 5 样本过多
    print(c)                                            # 可以选择性删除以上两组


list_total_train_data = []
list_subject_label = []
for subject_index in range(len(total_train_data)):
    list_total_train_data.extend(total_train_data[subject_index])
    list_subject_label.extend([subject_index] * len(total_train_data[subject_index]))

list_total_train_labels = []
for subject_index in range(len(total_train_labels)):
    list_total_train_labels.extend(total_train_labels[subject_index])

# 检查数据包含的样本的标签类别和数量是否与合并前保持一致
c = collections.Counter(list_subject_label)
print(c)

# shuffle the data set
random_vector = np.arange(len(list_subject_label))
np.random.shuffle(random_vector)
new_data = []
new_gesture_label = []
new_subject_label = []
for i in random_vector:
    new_data.append(list_total_train_data[i])
    new_gesture_label.append(list_total_train_labels[i])
    new_subject_label.append(list_subject_label[i])

train_test_ratio = 0.7
sep = int(train_test_ratio * len(random_vector))

# data split
train_data = new_data[:sep]
test_data = new_data[sep:]

train_gesture_labels = new_gesture_label[:sep]
test_gesture_labels = new_gesture_label[sep:]

train_subject_labels = new_subject_label[:sep]
test_subject_labels = new_subject_label[sep:]

# list to numpy
train_data = np.array(train_data, dtype=np.float32)
test_data = np.array(test_data, dtype=np.float32)

train_gesture_labels = np.array(train_gesture_labels, dtype=np.int64)
test_gesture_labels = np.array(test_gesture_labels, dtype=np.int64)

# train_subject_labels = np.array(train_subject_labels, dtype=np.float32)
# test_subject_labels = np.array(test_subject_labels, dtype=np.float32)
train_subject_labels = np.array(train_subject_labels, dtype=np.int64)
test_subject_labels = np.array(test_subject_labels, dtype=np.int64)

# numpy to tensor
train_data = TensorDataset(torch.from_numpy(train_data),
                           torch.from_numpy(train_gesture_labels),
                           torch.from_numpy(train_subject_labels))
test_data = TensorDataset(torch.from_numpy(test_data),
                          torch.from_numpy(test_gesture_labels),
                          torch.from_numpy(test_subject_labels))

# tensor to DataLoader
train_dataloader = DataLoader(train_data, batch_size=1024, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=1024, shuffle=True, drop_last=True)

# ----------------------------------------- Training & Validation ----------------------------------------- #

precision = 1e-8
ann = ANN(input_channel=448, number_of_classes=7).cuda()
class_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(ann.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=6,
                                                 verbose=True, eps=precision)

epoch_num = 120
patience = 12
patience_increase = 12
best_loss = float('inf')

for epoch in range(epoch_num):
    epoch_start_time = time.time()
    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

    running_loss = 0.
    correct_gesture_label, total_num = 0.0, 0.0

    # training
    ann.train()
    for i, (data, gesture_label, subject_label) in enumerate(train_dataloader):
        data = data.cuda()  # torch.Size([1024, 4, 8, 14])
        # print(data.shape)
        gesture_label = gesture_label.cuda()  # torch.Size([1024])     1024

        data = data.view(1024, -1)              # torch.Size([1024, 448]) -> 每个样本变为一维向量 适配ANN的输入
        # print(data.shape)
        # time.sleep(100)

        pred_gesture_label = ann(data)
        loss = class_criterion(pred_gesture_label, gesture_label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        correct_gesture_label += torch.sum(torch.argmax(pred_gesture_label, dim=1) == gesture_label).item()
        total_num += data.shape[0]

    total_acc = correct_gesture_label / total_num
    total_loss = running_loss / (i + 1)
    print('Train:   Loss: {:.4f}   Acc: {:.4f}'.format(total_loss, total_acc))

    # validation
    running_loss = 0.
    correct_gesture_label, total_num = 0.0, 0.0

    ann.eval()
    for i, (data, gesture_label, subject_label) in enumerate(test_dataloader):
        data = data.cuda()  # torch.Size([1024, 4, 8, 14])
        # print(data.shape)
        gesture_label = gesture_label.cuda()  # torch.Size([1024])     1024

        data = data.view(1024, -1)              # torch.Size([1024, 448]) -> 每个样本变为一维向量 适配ANN的输入
        # print(data.shape)
        # time.sleep(100)

        pred_gesture_label = ann(data)
        loss = class_criterion(pred_gesture_label, gesture_label)
        running_loss += loss.item()

        correct_gesture_label += torch.sum(torch.argmax(pred_gesture_label, dim=1) == gesture_label).item()
        total_num += data.shape[0]

    valid_acc = correct_gesture_label / total_num
    valid_loss = running_loss / (i + 1)

    print('Valid:   Loss: {:.4f}   Acc: {:.4f}'.format(valid_loss, valid_acc))
    print('Time usage: {:.2f}s'.format(time.time() - epoch_start_time))
    print()

    scheduler.step(valid_loss)
    if valid_loss + precision < best_loss:
        print('New best validation loss: {:.4f}'.format(valid_loss))
        best_loss = valid_loss
        best_weights = copy.deepcopy(ann.state_dict())
        patience = patience_increase + epoch
        print('So Far Patience: ', patience)


torch.save(best_weights, r'saved_model\ANN_8_subjects.pkl')

