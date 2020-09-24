import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import time

from domain_adversarial_network import DaNNet, FeatureExtractor, DomainClassifier, LabelPredictor


def scramble(examples, labels):
    """
    :param examples:
    :param labels:
    :return:
            返回顺序打乱的数据集
    """
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])
    return new_examples, new_labels


# %% Load raw dataset

# 原 pre-training0
source_set_training = np.load("../../PyTorchImplementation/formatted_datasets"
                              "/saved_pre_training_dataset_spectrogram.npy", encoding="bytes", allow_pickle=True)
source_data_training, source_labels_training = source_set_training

# 原 training0
target_set_training = np.load('../../PyTorchImplementation/formatted_datasets/saved_evaluation_dataset_training.npy',
                           encoding="bytes", allow_pickle=True)
target_data_training, target_labels_training = target_set_training

# 原 test0
target_set_test0 = np.load('../../PyTorchImplementation/formatted_datasets/saved_evaluation_dataset_test0.npy',
                         encoding="bytes", allow_pickle=True)
target_data_test0, target_labels_test0 = target_set_test0

# 原 test1
target_set_test1 = np.load('../../PyTorchImplementation/formatted_datasets/saved_evaluation_dataset_test1.npy',
                         encoding="bytes", allow_pickle=True)
target_data_test1, target_labels_test1 = target_set_test1

# scalar configurations
number_of_training_cycles = 4
number_of_training_repetitions = 1
learning_rate = 1e-3
precision = 1e-8


# %% Dataset Partition

list_source_train_dataloader = []
list_source_valid_dataloader = []
list_target_train_dataloader = []
list_target_valid_dataloader = []

# source domain dataset -> 19 subjects
print('Start loading Source training data...')
for j in range(19):
    source_examples_personne_training = []
    source_labels_personne_training = []

    source_examples_personne_valid = []
    source_labels_personne_valid = []

    for k in range(len(source_data_training[j])):
        # 前 3个cycle 的数据用作train
        if k < 21:
            source_examples_personne_training.extend(source_data_training[j][k])
            source_labels_personne_training.extend(source_labels_training[j][k])
        # 最后 1个cycle 的数据用作valid
        else:
            source_examples_personne_valid.extend(source_data_training[j][k])
            source_labels_personne_valid.extend(source_labels_training[j][k])

    # data scrambled
    source_examples_personne_training_scrambled, source_labels_personne_training_scrambled \
        = scramble(source_examples_personne_training, source_labels_personne_training)
    source_examples_personne_valid_scrambled, source_labels_personne_valid_scrambled \
        = scramble(source_examples_personne_valid, source_labels_personne_valid)

    # numpy2tensor
    source_train = TensorDataset(
        torch.from_numpy(np.array(source_examples_personne_training_scrambled, dtype=np.float32)),
        torch.from_numpy(np.array(source_labels_personne_training_scrambled, dtype=np.int64)))
    source_valid = TensorDataset(torch.from_numpy(np.array(source_examples_personne_valid_scrambled, dtype=np.float32)),
                                 torch.from_numpy(np.array(source_labels_personne_valid_scrambled, dtype=np.int64)))

    # tensor2dataloader
    source_train_Loader = DataLoader(source_train, batch_size=512, shuffle=True, drop_last=True)
    source_valid_Loader = DataLoader(source_valid, batch_size=256, shuffle=True, drop_last=True)

    # 装载 19名 source subject 的 dataloader
    list_source_train_dataloader.append(source_train_Loader)
    list_source_valid_dataloader.append(source_valid_Loader)
    print('No.{} subject'.format(j+1))
    print('source_examples_personne_training:   ', np.shape(source_examples_personne_training))
    print('source_examples_personne_valid:      ', np.shape(source_examples_personne_valid))
    print('size of source train dataloader:', len(list_source_train_dataloader))
    print('size of source valid dataloader:', len(list_source_valid_dataloader))
    print('-' * 30)

print('Loading Source training data finished.\n\n')

# target domain dataset -> 17 subjects
print('Start loading Target training data...')
for j in range(17):
    target_examples_personne_training = []
    target_labels_personne_training = []

    target_examples_personne_valid = []
    target_labels_personne_valid = []

    for k in range(len(target_data_training[j])):
        if k < 21:
            target_examples_personne_training.extend(target_data_training[j][k])
            target_labels_personne_training.extend(target_labels_training[j][k])
        else:
            target_examples_personne_valid.extend(target_data_training[j][k])
            target_labels_personne_valid.extend(target_labels_training[j][k])

    # dataset scrambled
    target_examples_personne_training_scrambled, target_labels_personne_training_scrambled \
        = scramble(target_examples_personne_training, target_labels_personne_training)
    target_examples_personne_valid_scrambled, target_labels_personne_valid_scrambled \
        = scramble(target_examples_personne_valid, target_labels_personne_valid)

    # numpy2tensor
    target_train = TensorDataset(torch.from_numpy(np.array(target_examples_personne_training_scrambled, dtype=np.float32)),
                                 torch.from_numpy(np.array(target_labels_personne_training_scrambled, dtype=np.int64)))
    target_valid = TensorDataset(torch.from_numpy(np.array(target_examples_personne_valid_scrambled, dtype=np.float32)),
                                 torch.from_numpy(np.array(target_labels_personne_valid_scrambled, dtype=np.int64)))

    # data set -> DataLoader
    target_train_loader = DataLoader(target_train, batch_size=512, shuffle=True, drop_last=True)
    target_valid_loader = DataLoader(target_valid, batch_size=256, shuffle=True, drop_last=True)

    # 装载 17名 target subject 的 dataloader
    list_target_train_dataloader.append(target_train_loader)
    list_target_valid_dataloader.append(target_valid_loader)

    print('No.{} subject'.format(j+1))
    print('source_examples_personne_training:   ', np.shape(target_examples_personne_training))
    print('source_examples_personne_valid:      ', np.shape(target_examples_personne_valid))
    print('size of source train dataloader:', len(list_target_train_dataloader))
    print('size of source valid dataloader:', len(list_target_valid_dataloader))
    print('-' * 30)

print('Loading target training data finished.\n\n')


# %% Network initialization
# net = DaNNet(number_of_classes=7)
best_loss = float('inf')
num_epochs = 300
criterion = nn.CrossEntropyLoss(reduction='sum')
# optimizer = optim.Adam(net.parameters(), lr=0.001)
precision = 1e-8

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor(number_of_classes=7).cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())

lamb = 1

# %% Training stage
source_train_dataloaders = {'train': list_source_train_dataloader, 'valid': list_source_valid_dataloader}
target_train_dataloaders = {'train': list_target_train_dataloader, 'valid': list_target_valid_dataloader}

# Traversal all target subjects
for target_subject_train_loader, target_subject_valid_loader in zip(list_target_train_dataloader, list_target_valid_dataloader):
    patience = 30
    patience_increase = 30
    # print(type(target_subject_train_loader))
    # print(type(list_target_train_dataloader[0]))
    # print(len(list_target_train_dataloader[0]))
    # time.sleep(1000)

    target_subject_num = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        for phase in ['train', 'valid']:
            if phase == 'train':
                feature_extractor.train(True)
                domain_classifier.train(True)
                label_predictor.train(True)
            else:
                feature_extractor.train(False)
                domain_classifier.train(False)
                label_predictor.train(False)

            running_D_loss, running_F_loss = 0.0, 0.0
            total_hit, total_num = 0.0, 0.0
            source_total_hit, target_total_hit = 0.0, 0.0
            source_total_num, target_total_num = 0.0, 0.0

            running_loss = 0.
            running_corrects = 0
            total = 0
            # get a random order for the source training subject
            random_vec = np.arange(len(source_train_dataloaders[phase]))  # len(random_vec) = 19
            np.random.seed(0)
            np.random.shuffle(random_vec)

            # 遍历 source domain 的每个subject dataloader
            for dataset_index in random_vec:
                loss_over_datasets = 0.
                correct_over_datasets = 0.

                # traversal all source training dataloader
                # 这里使用了 enumerate 直接遍历了dataloader中的每个batch
                if phase == 'train':
                    for i, ((source_data, source_label), (target_data, _)) in enumerate(
                            zip(source_train_dataloaders[phase][dataset_index], target_subject_train_loader)):
                        source_data = source_data.cuda()  # [256, 1, 8, 52] (target_data 也是这个尺寸)
                        # print('size of source_data: ', source_data.shape)
                        # print('size of target_data: ', target_data.shape)
                        source_label = source_label.cuda()
                        target_data = target_data.cuda()
                        # 混合 source / target data
                        mixed_data = torch.cat([source_data, target_data], dim=0)  # [512, 1, 8, 52]
                        # print('size of mixed_data:  ', mixed_data.shape)
                        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
                        # 设定 source data 的 label 是 1
                        domain_label[: source_data.shape[0]] = 1

                        # train Domain Classifier
                        feature = feature_extractor(mixed_data)  # [512, 64, 4, 4]
                        # print(feature.shape)
                        # time.sleep(1000)
                        domain_logits = domain_classifier(feature.detach())
                        loss = domain_criterion(domain_logits, domain_label)
                        running_D_loss += loss.item()
                        loss.backward()
                        optimizer_D.step()

                        # train Feature Extractor and Domain Classifier
                        class_logits = label_predictor(feature[:source_data.shape[0]])
                        domain_logits = domain_classifier(feature)
                        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits,
                                                                                                     domain_label)
                        running_F_loss += loss.item()
                        loss.backward()
                        optimizer_F.step()
                        optimizer_C.step()

                        optimizer_D.zero_grad()
                        optimizer_F.zero_grad()
                        optimizer_C.zero_grad()

                        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
                        total_num += source_data.shape[0]
                        train_accuracy = total_hit / total_num

                        print('Training Stage:  Subject:{}  batch:{}    train acc: {:.6f}'
                              .format(dataset_index, i, train_accuracy))

                else:
                    label_predictor.eval()
                    feature_extractor.eval()
                    print('validation stage:\n')
                    for i, ((source_data, source_label), (target_data, target_label)) in enumerate(
                            zip(source_train_dataloaders[phase][dataset_index], target_subject_valid_loader)):
                        source_data = source_data.cuda()  # [256, 1, 8, 52] (target_data 也是这个尺寸)
                        source_label = source_label.cuda()
                        target_data = target_data.cuda()
                        target_label = target_label.cuda()

                        source_domain_label = torch.ones([source_data.shape[0], 1]).cuda()
                        target_domain_label = torch.zeros([target_data.shape[0], 1]).cuda()

                        feature_source = feature_extractor(source_data)
                        feature_target = feature_extractor(target_data)

                        # predict domain labels for both source and target data
                        domain_source_logits = domain_classifier(feature_source)
                        domain_target_logits = domain_classifier(feature_target)

                        # predict class labels for both source and target data
                        class_source_logits = label_predictor(feature_source)
                        class_target_logits = label_predictor(feature_target)

                        # calculate loss
                        loss_domain = domain_criterion(domain_source_logits, source_domain_label) \
                                      + domain_criterion(domain_target_logits, target_domain_label)
                        loss_class = class_criterion(class_source_logits, source_label)
                        loss = loss_class - lamb * loss_domain

                        source_total_hit += torch.sum(torch.argmax(class_source_logits, dim=1) == source_label).item()
                        source_total_num += source_data.shape[0]
                        source_accuracy = source_total_hit / source_total_num

                        target_total_hit += torch.sum(torch.argmax(class_target_logits, dim=1) == target_label).item()
                        target_total_num += target_data.shape[0]
                        target_accuracy = target_total_hit / target_total_num

                        print('{}   Loss:  {:.8f}   s_Acc:  {:.8f}  t_Acc:  {:.8}'
                              .format(phase, loss, source_accuracy, target_accuracy))

                # loss_over_datasets += loss
                #
                # running_loss =
                # loss_over_datasets += loss
                # correct_over_datasets +=

    target_subject_num += 1



                #
                # for i, ((source_data, source_label), (target_data, _)) in enumerate(
                #         zip(source_train_dataloaders[phase][dataset_index], target_subject_train_loader)):
                #
                #     if phase == 'train':
                #         source_data = source_data.cuda()                # [256, 1, 8, 52] (target_data 也是这个尺寸)
                #         # print('size of source_data: ', source_data.shape)
                #         # print('size of target_data: ', target_data.shape)
                #         source_label = source_label.cuda()
                #         target_data = target_data.cuda()
                #         # 混合 source / target data
                #         mixed_data = torch.cat([source_data, target_data], dim=0)   # [512, 1, 8, 52]
                #         # print('size of mixed_data:  ', mixed_data.shape)
                #         domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
                #         # 设定 source data 的 label 是 1
                #         domain_label[: source_data.shape[0]] = 1
                #
                #         # train Domain Classifier
                #         feature = feature_extractor(mixed_data)         # [512, 64, 4, 4]
                #         # print(feature.shape)
                #         # time.sleep(1000)
                #         domain_logits = domain_classifier(feature.detach())
                #         loss = domain_criterion(domain_logits, domain_label)
                #         running_D_loss += loss.item()
                #         loss.backward()
                #         optimizer_D.step()
                #
                #         # train Feature Extractor and Domain Classifier
                #         class_logits = label_predictor(feature[:source_data.shape[0]])
                #         domain_logits = domain_classifier(feature)
                #         loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
                #         running_F_loss += loss.item()
                #         loss.backward()
                #         optimizer_F.step()
                #         optimizer_C.step()
                #
                #         optimizer_D.zero_grad()
                #         optimizer_F.zero_grad()
                #         optimizer_C.zero_grad()
                #
                #         total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
                #         total_num += source_data.shape[0]
                #         train_accuracy = total_hit / total_num
                #
                #         print('Training Stage:  Subject:{}  batch:{}    train acc: {:.6f}'
                #               .format(dataset_index, i, train_accuracy), end='\r')
                #
                #     else:
                #         # evaluation process
                #         label_predictor.eval()
                #         feature_extractor.eval()
                #         print('validation stage:')
                #         time.sleep(1000)












                    # print(type(source_data))
                    # print('source_data: ', source_data)
                    # print(type(source_label))
                    # print('source_label: ', source_label)
                    # print(type(target_data))
                    # print('target_data: ', target_data)
                    # time.sleep(1000)












