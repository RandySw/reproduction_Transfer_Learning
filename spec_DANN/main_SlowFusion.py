"""


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

from model import SlowFusionModel

# -------------------------------------------- Load Data ---------------------------------------------- #

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

# -------------------------------------- Training Stage ------------------------------------------- #

net = SlowFusionModel().cuda()

for p in net.parameters():
    p.requires_grad = True

precision = 1e-8

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=8,
                                                 verbose=True, eps=precision)

epoch_num = 120   # 120
patience = 15   # 12
patience_increase = 15  # 12
best_acc = 0
best_loss = float('inf')


for epoch in range(epoch_num):

    epoch_start_time = time.time()
    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

    len_dataloader = len(train_dataloader)

    running_loss, total_num, correct_gesture_label = 0.0, 0.0, 0.0

    net.train()

    for i, (data, gesture_label, subject_label) in enumerate(train_dataloader):
        data = data.cuda()                      # torch.Size([1024, 4, 8, 14])
        # print(data.shape)
        gesture_label = gesture_label.cuda()    # torch.Size([1024])     1024

        pred_label = net(data)
        loss = class_criterion(pred_label, gesture_label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        correct_gesture_label += torch.sum(torch.argmax(pred_label, dim=1) == gesture_label).item()
        total_num += data.shape[0]

    gesture_acc = correct_gesture_label / total_num
    epoch_loss = running_loss / (i + 1)
    print('Train:   Accuracy: {:.4f}    Epoch Loss: {:.4f}'.format(gesture_acc, epoch_loss))


# validation
    running_loss, total_num, correct_gesture_label = 0.0, 0.0, 0.0
    net.eval()

    for i, (data, gesture_label, subject_label) in enumerate(test_dataloader):
        data = data.cuda()                      # torch.Size([512, 4, 8, 14])
        # print(data.shape)
        gesture_label = gesture_label.cuda()    # torch.Size([512])

        pred_label = net(data)
        loss = class_criterion(pred_label, gesture_label)
        running_loss += loss.item()

        correct_gesture_label += torch.sum(torch.argmax(pred_label, dim=1) == gesture_label).item()
        total_num += data.shape[0]

    gesture_acc = correct_gesture_label / total_num
    epoch_val_loss = running_loss / (i + 1)
    print('Valid:   Accuracy: {:.4f}    Epoch Loss: {:.4f}'.format(gesture_acc, epoch_val_loss))

    print('Time Usage:  {:.2f}s'.format(time.time() - epoch_start_time))

    scheduler.step(epoch_val_loss)

    if epoch_val_loss + precision < best_loss:
        print('New best validation loss:    {:.4f}'.format(epoch_val_loss))
        best_loss = epoch_val_loss
        best_weights = copy.deepcopy(net.state_dict())
        patience = patience_increase + epoch
        print('So Far Patience: ', patience)

    print()

# save model
torch.save(best_weights, r'saved_model\slow_fusion_cnn.pkl')
print('Model Saved')



