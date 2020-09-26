import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable


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


def load_source_train_data(source_data_training, source_labels_training):

    list_source_train_dataloader = []
    list_source_valid_dataloader = []

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
        source_valid = TensorDataset(
            torch.from_numpy(np.array(source_examples_personne_valid_scrambled, dtype=np.float32)),
            torch.from_numpy(np.array(source_labels_personne_valid_scrambled, dtype=np.int64)))

        # tensor2dataloader
        source_train_Loader = DataLoader(source_train, batch_size=512, shuffle=True, drop_last=True)
        source_valid_Loader = DataLoader(source_valid, batch_size=256, shuffle=True, drop_last=True)

        # 装载 19名 source subject 的 dataloader
        list_source_train_dataloader.append(source_train_Loader)
        list_source_valid_dataloader.append(source_valid_Loader)

        print('No.{} subject'.format(j + 1))
        print('source_examples_personne_training:   ', np.shape(source_examples_personne_training))
        print('source_examples_personne_valid:      ', np.shape(source_examples_personne_valid))
        print('size of source train dataloader:', len(list_source_train_dataloader))
        print('size of source valid dataloader:', len(list_source_valid_dataloader))
        print('-' * 30)

    print('Loading Source training/valid data finished.\n\n')

    return list_source_train_dataloader, list_source_valid_dataloader


def load_target_train_data(target_data_training, target_labels_training):

    list_target_train_dataloader = []
    list_target_valid_dataloader = []

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
        target_train = TensorDataset(
            torch.from_numpy(np.array(target_examples_personne_training_scrambled, dtype=np.float32)),
            torch.from_numpy(np.array(target_labels_personne_training_scrambled, dtype=np.int64)))
        target_valid = TensorDataset(
            torch.from_numpy(np.array(target_examples_personne_valid_scrambled, dtype=np.float32)),
            torch.from_numpy(np.array(target_labels_personne_valid_scrambled, dtype=np.int64)))

        # data set -> DataLoader
        target_train_loader = DataLoader(target_train, batch_size=512, shuffle=True, drop_last=True)
        target_valid_loader = DataLoader(target_valid, batch_size=256, shuffle=True, drop_last=True)

        # 装载 17名 target subject 的 dataloader
        list_target_train_dataloader.append(target_train_loader)
        list_target_valid_dataloader.append(target_valid_loader)

        print('No.{} subject'.format(j + 1))
        print('source_examples_personne_training:   ', np.shape(target_examples_personne_training))
        print('source_examples_personne_valid:      ', np.shape(target_examples_personne_valid))
        print('size of source train dataloader:', len(list_target_train_dataloader))
        print('size of source valid dataloader:', len(list_target_valid_dataloader))
        print('-' * 30)

    print('Loading target training/valid data finished.\n\n')

    return list_target_train_dataloader, list_target_valid_dataloader


def load_target_test_data(target_data_testing, target_labels_testing):

    print('Start loading Target test data...')
    list_target_test_loader = []

    for j in range(17):
        target_examples_personne_testing = []
        target_labels_personne_testing = []

        for k in range(len(target_data_testing[j])):
            target_examples_personne_testing.extend(target_data_testing[j][k])
            target_labels_personne_testing.extend(target_labels_testing[j][k])

        # dataset scrambled
        target_examples_personne_testing_scrambled, target_labels_personne_testing_scrambled \
            = scramble(target_examples_personne_testing, target_labels_personne_testing)

        # numpy2tensor
        target_test = TensorDataset(
            torch.from_numpy(np.array(target_examples_personne_testing_scrambled, dtype=np.float32)),
            torch.from_numpy(np.array(target_labels_personne_testing_scrambled, dtype=np.int64)))

        # data set -> DataLoader
        target_test_loader = DataLoader(target_test, batch_size=512, shuffle=True, drop_last=True)

        # 装载 17名 target subject 的 dataloader
        list_target_test_loader.append(target_test_loader)

        print('No.{} subject'.format(j + 1))
        print('target_examples_personne_testing:   ', np.shape(target_examples_personne_testing))
        print('size of target_test dataloader:', len(list_target_test_loader))
        print('-' * 30)

    print('Loading target testing data finished.\n\n')

    return list_target_test_loader

















