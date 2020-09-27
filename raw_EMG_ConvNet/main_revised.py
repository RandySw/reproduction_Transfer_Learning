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


def training(source_dataloader, target_dataloader, feature_extractor, domain_classifier, label_predictor,
             optim_F, optim_C, optim_D, lamb=0.1):

    running_D_loss, running_F_loss = 0.0, 0.0
    source_hit_num, total_num = 0.0, 0.0

    feature_extractor.train()
    domain_classifier.train()
    label_predictor.train()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    len_dataloader = min(len(source_dataloader), len(target_dataloader))

    optimizer_F = optim_F
    optimizer_C = optim_C
    optimizer_D = optim_D

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        if i > len_dataloader:
            break
        # print('{} batch'.format(i))

        source_data = source_data.cuda()  # [256, 1, 8, 52]
        source_label = source_label.cuda()
        target_data = target_data.cuda()

        # 混合 source / target data
        mixed_data = torch.cat([source_data, target_data], dim=0)  # [512, 1, 8, 52]

        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # 设定 source data 的 label 是 1
        domain_label[: source_data.shape[0]] = 1

        # train Domain Classifier
        feature = feature_extractor(mixed_data)  # [512, 64, 4, 4]
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()  # 一定要使用.item() 将loss张量转化成float格式存储，不然显存会爆

        loss.backward()
        optimizer_D.step()

        # train Feature Extractor and Domain Classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss 包括 source data 的 label loss 以及 source data 和 target data 的 domain loss
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits,
                                                                                     domain_label)
        running_F_loss += loss.item()

        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        source_hit_num += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        # print(i, end='\r')

    dataloader_D_loss = running_D_loss / (i + 1)
    dataloader_F_loss = running_F_loss / (i + 1)
    dataloader_source_acc = source_hit_num / total_num

    return dataloader_D_loss, dataloader_F_loss, dataloader_source_acc


def validation(dataloader, feature_extractor, domain_classifier, label_predictor, domain='source'):

    running_D_loss, running_F_loss = 0.0, 0.0
    correct_pred_num, total_num = 0.0, 0.0

    label_predictor.eval()
    feature_extractor.eval()
    domain_classifier.eval()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()
        if domain == 'source':
            domain_labels = torch.ones([data.shape[0], 1]).cuda()
        elif domain == 'target':
            domain_labels = torch.zeros([data.shape[0], 1]).cuda()

        feature = feature_extractor(data)

        # get predicted domain and class labels
        domain_logits = domain_classifier(feature)
        class_logits = label_predictor(feature)

        # calculate loss
        loss_domain = domain_criterion(domain_logits, domain_labels)
        loss_class = class_criterion(class_logits, labels)

        # calculate accuracy
        correct_pred_num += torch.sum(torch.argmax(class_logits, dim=1) == labels).item()
        total_num += data.shape[0]
        # print(i, end='\r')

        running_D_loss += loss_domain.item()
        running_F_loss += loss_class.item()    # 仅包括分类损失
        # print('batch_C_loss: ', loss_class.item())

    dataloader_domain_loss = running_D_loss / (i + 1)
    dataloader_class_loss = running_F_loss / (i + 1)
    dataloader_acc = correct_pred_num / total_num
    # print('dataloader_C_loss: ,', dataloader_class_loss)

    return dataloader_domain_loss, dataloader_class_loss, dataloader_acc


def testing(dataloader, feature_extractor, domain_classifier, label_predictor):
    label_predictor.eval()
    feature_extractor.eval()
    domain_classifier.eval()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    running_D_loss, running_F_loss = 0.0, 0.0
    correct_pred_num, total_num = 0.0, 0.0

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()
        domain_labels = torch.zeros([data.shape[0], 1]).cuda()

        feature = feature_extractor(data)

        # get predicted domain and class labels
        domain_logits = domain_classifier(feature)
        class_logits = label_predictor(feature)

        # calculate loss
        loss_domain = domain_criterion(domain_logits, domain_labels)
        loss_class = class_criterion(class_logits, labels)

        # calculate accuracy
        correct_pred_num += torch.sum(torch.argmax(class_logits, dim=1) == labels).item()
        total_num += data.shape[0]

        running_D_loss += loss_domain.item()
        running_F_loss += loss_class.item()    # 仅包括分类损失

    dataloader_domain_loss = running_D_loss / (i + 1)
    dataloader_class_loss = running_F_loss / (i + 1)
    dataloader_acc = correct_pred_num / total_num

    return dataloader_domain_loss, dataloader_class_loss, dataloader_acc


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


net = DaNNet(number_of_classes=7)

for p in net.parameters():
    p.requires_grad = True

optimizer = optim.Adam(net.parameters(), lr=1e-3)
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

# training
epoch_num = 100

for epoch in range(epoch_num):
    epoch_start_time = time.time()

    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

    len_dataloader = min(len(src_data_train), len(tag_data_train))
    data_source_iter = iter(src_data_train)
    data_target_iter = iter(tag_data_train)

    i = 0
    while i < len_dataloader:
        p = float(i + epoch * len_dataloader) / epoch_num / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        data_source = data_source_iter.next()
        s_data, labels = data_source

        net.zero_grad()

        batch_size = len(labels)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        s_data = s_data.cuda()
        labels = labels.cuda()
        domain_label = domain_label.cuda()

        class_output, domain_output = net(s_data, alpha=alpha)
        err_s_label = class_criterion(class_output, labels)
        err_s_domain = domain_criterion(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_data, _ = data_target

        batch_size = len(t_data)

        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        t_data = t_data.cuda()
        domain_label = domain_label.cuda()

        _, domain_output = net(t_data, alpha=alpha)
        err_t_domain = domain_criterion(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label

        err.backward()
        optimizer.step()

        i += 1

        print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))


print('done')




















