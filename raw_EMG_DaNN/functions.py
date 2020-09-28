import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable


def training(source_dataloader, target_dataloader,
                   feature_extractor, domain_classifier, label_predictor, lamb=0.1):

    running_D_loss, running_F_loss = 0.0, 0.0
    source_hit_num, total_num = 0.0, 0.0

    feature_extractor.train()
    domain_classifier.train()
    label_predictor.train()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    len_dataloader = min(len(source_dataloader), len(target_dataloader))

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())

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

        running_D_loss += loss_domain
        running_F_loss += loss_class    # 仅包括分类损失

    dataloader_domain_loss = loss_domain / (i + 1)
    dataloader_class_loss = loss_class / (i + 1)
    dataloader_acc = correct_pred_num / total_num

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

        running_D_loss += loss_domain
        running_F_loss += loss_class    # 仅包括分类损失

    dataloader_domain_loss = loss_domain / (i + 1)
    dataloader_class_loss = loss_class / (i + 1)
    dataloader_acc = correct_pred_num / total_num

    return dataloader_domain_loss, dataloader_class_loss, dataloader_acc








