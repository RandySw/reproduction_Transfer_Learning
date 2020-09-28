import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt


import time
import os
import copy

from domain_adversarial_network import DaNNet
from utliz import scramble, load_source_train_data, load_target_train_data, load_target_test_data


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def training(source_dataloader, target_dataloader, net, optim, num_epoch, epoch=1):

    running_loss = 0.0
    source_hit_num, total_num = 0.0, 0.0

    net.train()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    len_dataloader = min(len(source_dataloader), len(target_dataloader))

    optimizer = optim

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        if i > len_dataloader:
            break
        # print('{} batch'.format(i))

        source_data = source_data.cuda()  # [256, 1, 8, 52]
        source_label = source_label.cuda()

        target_data = target_data.cuda()

        s_domain_labels = torch.ones([source_label.shape[0], 1]).cuda()
        t_domain_labels = torch.zeros([target_data.shape[0], 1]).cuda()

        p = float(i + epoch * len_dataloader) / num_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # train with source data
        s_class_output, s_domain_output = net(source_data, alpha=alpha)

        err_s_label = class_criterion(s_class_output, source_label)
        err_s_domain = domain_criterion(s_domain_output, s_domain_labels)

        # train with target data
        _, t_domain_output = net(target_data, alpha=alpha)
        err_t_domain = domain_criterion(t_domain_output, t_domain_labels)

        err = err_t_domain + err_s_domain + err_s_label
        running_loss += err

        net.zero_grad()
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        source_hit_num += torch.sum(torch.argmax(s_class_output, dim=1) == source_label).item()
        total_num += source_data.shape[0]

    s_acc = source_hit_num / total_num
    running_loss = running_loss / (i + 1)

    return running_loss, s_acc, alpha


def validation(dataloader, net, alpha=1, domain='source'):

    running_D_loss, running_C_loss = 0.0, 0.0
    correct_pred_num, total_num = 0.0, 0.0

    net.eval()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()
        if domain == 'source':
            domain_labels = torch.ones([data.shape[0], 1]).cuda()
        elif domain == 'target':
            domain_labels = torch.zeros([data.shape[0], 1]).cuda()

        class_output, domain_output = net(data, alpha=alpha)

        # calculate loss
        err_domain = domain_criterion(domain_output, domain_labels)
        err_class = class_criterion(class_output, labels)
        running_D_loss += err_domain.item()
        running_C_loss += err_class.item()

        # calculate accuracy
        correct_pred_num += torch.sum(torch.argmax(class_output, dim=1) == labels).item()
        total_num += data.shape[0]

    dataloader_D_loss = running_D_loss / (i + 1)
    dataloader_C_loss = running_C_loss / (i + 1)
    dataloader_acc = correct_pred_num / total_num

    return dataloader_D_loss, dataloader_C_loss, dataloader_acc


def testing(dataloader, net, alpha=1):
    net.eval()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    running_D_loss, running_C_loss = 0.0, 0.0
    correct_pred_num, total_num = 0.0, 0.0

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()
        domain_labels = torch.zeros([data.shape[0], 1]).cuda()

        class_output, domain_output = net(data, alpha=alpha)

        # calculate loss
        err_domain = domain_criterion(domain_output, domain_labels)
        err_class = class_criterion(class_output, labels)
        running_D_loss += err_domain.item()
        running_C_loss += err_class.item()

        # calculate accuracy
        correct_pred_num += torch.sum(torch.argmax(class_output, dim=1) == labels).item()
        total_num += data.shape[0]

    dataloader_D_loss = running_D_loss / (i + 1)
    dataloader_C_loss = running_C_loss / (i + 1)
    dataloader_acc = correct_pred_num / total_num

    return dataloader_D_loss, dataloader_C_loss, dataloader_acc


torch.cuda.set_device(0)
seed = 0
torch.manual_seed(seed)

os.environ['KMP_DUPLICATE_LIB_OK']= 'True'


# %% Load raw dataset
source_set_training = np.load("../../PyTorchImplementation/formatted_datasets"
                              "/saved_pre_training_dataset_spectrogram.npy", encoding="bytes", allow_pickle=True)
source_data_training, source_labels_training = source_set_training


target_set_training = np.load('../../PyTorchImplementation/formatted_datasets/saved_evaluation_dataset_training.npy',
                           encoding="bytes", allow_pickle=True)
target_data_training, target_labels_training = target_set_training

target_set_test0 = np.load('../../PyTorchImplementation/formatted_datasets/saved_evaluation_dataset_test0.npy',
                         encoding="bytes", allow_pickle=True)
target_data_test0, target_labels_test0 = target_set_test0

target_set_test1 = np.load('../../PyTorchImplementation/formatted_datasets/saved_evaluation_dataset_test1.npy',
                         encoding="bytes", allow_pickle=True)
target_data_test1, target_labels_test1 = target_set_test1


list_source_train_dataloader, list_source_valid_dataloader \
    = load_source_train_data(source_data_training, source_labels_training)

list_target_train_dataloader, list_target_valid_dataloader \
    = load_target_train_data(target_data_training, target_labels_training)

list_target_test0_dataloader = load_target_test_data(target_data_test0, target_labels_test0)

list_target_test1_dataloader = load_target_test_data(target_data_test1, target_labels_test1)


src_data_train = []
for i in range(len(list_source_train_dataloader)):
    src_data_train.extend(list_source_train_dataloader[i])

src_data_valid = []
for i in range(len(list_source_valid_dataloader)):
    src_data_valid.extend(list_source_valid_dataloader[i])

tag_data_train = []
for i in range(len(list_target_train_dataloader)):
    # i = 0
    tag_data_train.extend(list_target_train_dataloader[i])

tag_data_valid = []
for i in range(len(list_target_valid_dataloader)):
    # i = 0
    tag_data_valid.extend(list_target_valid_dataloader[i])

tag_data_test0 = []
for i in range(len(list_target_test0_dataloader)):
    tag_data_test0.extend(list_target_test0_dataloader[i])
# tag_data_test0.extend(list_target_test0_dataloader[0])

tag_data_test1 = []
for i in range(len(list_target_test1_dataloader)):
    tag_data_test1.extend(list_target_test1_dataloader[i])

train_loss_cur = []
train_acc_cur = []
src_val_D_loss_cur = []
src_val_C_loss_cur = []
src_val_acc_cur = []
tag_val_D_loss_cur = []
tag_val_C_loss_cur = []
tag_val_acc_cur = []
tag_test0_D_loss_cur = []
tag_test0_C_loss_cur = []
tag_test0_acc_cur = []
tag_test1_D_loss_cur = []
tag_test1_C_loss_cur = []
tag_test1_acc_cur = []


Da_net = DaNNet(number_of_classes=7).cuda()

for p in Da_net.parameters():
    p.requires_grad = True

precision = 1e-8

optimizer = optim.Adam(Da_net.parameters(), lr=1.5e-3)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.1, patience=5,
                                                 verbose=True, eps=precision)
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

# training
epoch_num = 120
patience = 12
patience_increase = 12
best_acc = 0

for epoch in range(epoch_num):
    epoch_start_time = time.time()

    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

    len_dataloader = min(len(src_data_train), len(tag_data_train))
    # train model with source data

    train_loss, train_acc, alpha = training(src_data_train, tag_data_train, Da_net, optimizer, epoch_num, epoch=epoch)

    src_valid_D_loss, src_valid_C_loss, src_valid_acc = validation(src_data_valid, net=Da_net, alpha=alpha)

    tag_valid_D_loss, tag_valid_C_loss, tag_valid_acc = validation(tag_data_valid, net=Da_net, alpha=alpha)

    tag_test0_D_loss, tag_test0_C_loss, tag_test0_acc = testing(tag_data_test0, net=Da_net, alpha=alpha)

    tag_test1_D_loss, tag_test1_C_loss, tag_test1_acc = testing(tag_data_test1, net=Da_net, alpha=alpha)

    print('Training: Loss:{:.4f}  src_Acc:{:.4f}'.format(train_loss, train_acc))
    print('Valid Source:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(src_valid_D_loss, src_valid_C_loss, src_valid_acc))
    print('Valid Target:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(tag_valid_D_loss, tag_valid_C_loss, tag_valid_acc))
    print('Test0:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(tag_test0_D_loss, tag_test0_C_loss, tag_test0_acc))
    print('Test1:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(tag_test1_D_loss, tag_test1_C_loss, tag_test1_acc))
    print('epoch time usage: {:.2f}s'.format(time.time() - epoch_start_time))

    scheduler.step(train_loss)

    if tag_valid_acc + precision > best_acc:
        print('New Best Target Validation Accuracy: {:.4f}'.format(tag_valid_acc))
        best_acc = tag_valid_acc
        best_weights = copy.deepcopy(Da_net.state_dict())
        patience = patience_increase + epoch
        print('So Far Patience: ', patience)
    print()

    # collecting results
    train_loss_cur.append(train_loss)
    train_acc_cur.append(train_acc)
    src_val_D_loss_cur.append(src_valid_D_loss)
    src_val_C_loss_cur.append(src_valid_C_loss)
    src_val_acc_cur.append(src_valid_acc)
    tag_val_D_loss_cur.append(tag_valid_D_loss)
    tag_val_C_loss_cur.append(tag_valid_C_loss)
    tag_val_acc_cur.append(tag_valid_acc)
    tag_test0_D_loss_cur.append(tag_test0_D_loss)
    tag_test0_C_loss_cur.append(tag_test0_C_loss)
    tag_test0_acc_cur.append(tag_test0_acc)
    tag_test1_D_loss_cur.append(tag_test1_D_loss)
    tag_test1_C_loss_cur.append(tag_test1_C_loss)
    tag_test1_acc_cur.append(tag_test1_acc)

    if epoch > patience:
        break

print('-' * 20 + '\n' + '-' * 20)
print('Best Best Target Validation Accuracy: {:.4f}'.format(best_acc))

# Loss curve
plt.plot(train_loss_cur)
plt.title('Training Loss - Epoch')
plt.legend(['train_loss'])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()

plt.plot(src_val_D_loss_cur)
plt.plot(tag_val_D_loss_cur)
plt.plot(tag_test0_D_loss_cur)
plt.plot(tag_test1_D_loss_cur)
plt.title('Domain Loss - Epoch')
plt.legend(['src_val_D_loss', 'tag_val_D_loss', 'tag_test0_D_loss', 'tag_test1_D_loss'])
plt.xlabel('Epoch')
plt.ylabel('Domain Loss')
plt.show()

plt.plot(src_val_C_loss_cur)
plt.plot(tag_val_C_loss_cur)
plt.plot(tag_test0_C_loss_cur)
plt.plot(tag_test1_C_loss_cur)
plt.title('Classification Loss - Epoch')
plt.legend(['src_val_C_loss', 'tag_val_C_loss', 'tag_test0_C_loss', 'tag_test1_C_loss'])
plt.xlabel('Epoch')
plt.ylabel('Classification Loss')
plt.show()

plt.plot(train_acc_cur)
plt.plot(src_val_acc_cur)
plt.plot(tag_val_acc_cur)
plt.plot(tag_test0_acc_cur)
plt.plot(tag_test1_acc_cur)
plt.title('Classification Accuracy - Epoch')
plt.legend(['train_acc', 'src_val_acc', 'tag_val_acc', 'tag_test0_acc', 'tag_test1_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()





    # ToDo:
    # dropout rate
    # 收集每次epoch结果数据绘制曲线
    # 尝试单人target




