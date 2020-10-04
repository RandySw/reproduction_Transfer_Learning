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
from functions import training, validation, testing


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

torch.cuda.set_device(0)
seed = 0
torch.manual_seed(seed)

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
    tag_data_train.extend(list_target_train_dataloader[i])

tag_data_valid = []
for i in range(len(list_target_valid_dataloader)):
    tag_data_valid.extend(list_target_valid_dataloader[i])
# tag_data_valid.extend(list_target_valid_dataloader[0])

tag_data_test0 = []
for i in range(len(list_target_test0_dataloader)):
    tag_data_test0.extend(list_target_test0_dataloader[i])
# tag_data_test0.extend(list_target_test0_dataloader[0])

tag_data_test1 = []
for i in range(len(list_target_test1_dataloader)):
    tag_data_test1.extend(list_target_test1_dataloader[i])
# tag_data_test1.extend(list_target_test1_dataloader[0])

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

optimizer = optim.Adam(Da_net.parameters(), lr=2e-4)
# optimizer = optim.SGD(Da_net.parameters(), momentum=0.5, lr=2e-4)
# optimizer = optim.SGD(Da_net.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5,
                                                 verbose=True, eps=precision)
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

# training
epoch_num = 120
patience = 20
patience_increase = 20
best_acc = 0
best_loss = float('inf')

for epoch in range(epoch_num):
    epoch_start_time = time.time()

    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

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

    val_loss = tag_valid_D_loss + src_valid_C_loss + src_valid_D_loss

    scheduler.step(val_loss)

    # if tag_valid_acc + precision > best_acc:
    #     print('New Best Target Validation Accuracy: {:.4f}'.format(tag_valid_acc))
    #     best_acc = tag_valid_acc
    #     best_weights = copy.deepcopy(Da_net.state_dict())
    #     patience = patience_increase + epoch
    #     print('So Far Patience: ', patience)
    # print()

    if val_loss + precision < best_loss:
        print('New Best Validation Loss: {:.4f}'.format(val_loss))
        best_loss = train_loss
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
# print('Best Best Target Validation Accuracy: {:.4f}'.format(best_acc))
print('Best Best Validation Loss: {:.4f}'.format(best_loss))

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



