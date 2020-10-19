import numpy as np
import torch
import torch.nn as nn


def training(source_dataloader, target_dataloader, net, optim, num_epoch, epoch=1):

    running_loss = 0.0
    source_hit_num, total_num = 0.0, 0.0

    net.train()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    len_dataloader = min(len(source_dataloader), len(target_dataloader))

    optimizer = optim

    for i, ((source_data, source_label), (target_data, target_label)) in enumerate(zip(source_dataloader, target_dataloader)):
        if i > len_dataloader:
            break
        # print('{} batch'.format(i))

        source_data = source_data.cuda()  # [256, 1, 8, 52]
        source_label = source_label.cuda()

        target_data = target_data.cuda()
        target_label = target_label.cuda()

        s_domain_labels = torch.ones([source_label.shape[0], 1]).cuda()
        t_domain_labels = torch.zeros([target_data.shape[0], 1]).cuda()

        p = float(i + epoch * len_dataloader) / num_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # train with source data
        s_class_output, s_domain_output = net(source_data, alpha=alpha)

        err_s_label = class_criterion(s_class_output, source_label)
        err_s_domain = domain_criterion(s_domain_output, s_domain_labels)

        # train with target data
        t_class_output, t_domain_output = net(target_data, alpha=alpha)
        err_t_domain = domain_criterion(t_domain_output, t_domain_labels)
        err_t_label = class_criterion(t_class_output, target_label)

        err = err_t_domain + err_t_label + err_s_domain + err_s_label
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

