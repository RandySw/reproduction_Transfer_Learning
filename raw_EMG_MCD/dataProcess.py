import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def scramble(examples, labels, second_labels=[]):
    """
    :param examples:
    :param labels:
    :param second_labels:
    :return:
            返回顺序打乱的数据集
    """
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    if len(second_labels) == len(labels):
        new_second_labels = []
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
            new_second_labels.append(second_labels[i])
        return new_examples, new_labels, new_second_labels
    else:
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
        return new_examples, new_labels


def sourceDataLoader(examples, labels):
    """
        :param examples:
                        第一维：受试者
                        第二维：单个受试者的27个样本
                        第三维：样本中的数据矩阵
        :param labels:
        :return:
        """
    list_train_dataloader = []
    list_validation_dataloader = []
    human_number = 0

    # 遍历19个受试者
    for j in range(19):
        examples_personne_training = []
        labels_gesture_personne_training = []
        labels_human_personne_training = []

        examples_personne_valid = []
        labels_gesture_personne_valid = []
        labels_human_personne_valid = []

        # k：从 0 遍历到 26 -> 对应单个受试者的 27 个样本
        for k in range(len(examples[j])):
            # 前三个 cycle 的数据用作 training
            if k < 21:
                # 将第 k 个受试者的所有样本数据加载到 examples_personne_training 中
                examples_personne_training.extend(examples[j][k])

                # 将第 k 个受试者的所有样本标签加载到 labels_gesture_personne_training 中
                labels_gesture_personne_training.extend(labels[j][k])
                # 这个 labels_human_personne_training 不知道干什么用的
                labels_human_personne_training.extend(human_number * np.ones(len(labels[j][k])))
            # 后三个 cycle 的数据用作 validation
            else:
                examples_personne_valid.extend(examples[j][k])
                labels_gesture_personne_valid.extend(labels[j][k])
                labels_human_personne_valid.extend(human_number * np.ones(len(labels[j][k])))

        # print(np.shape(examples_personne_training))
        examples_personne_scrambled, labels_gesture_personne_scrambled, labels_human_personne_scrambled = scramble(
            examples_personne_training, labels_gesture_personne_training, labels_human_personne_training)

        examples_personne_scrambled_valid, labels_gesture_personne_scrambled_valid, labels_human_personne_scrambled_valid = scramble(
            examples_personne_valid, labels_gesture_personne_valid, labels_human_personne_valid)

        # 将数据集包装成 tensor 的形式
        train = TensorDataset(torch.from_numpy(np.array(examples_personne_scrambled, dtype=np.float32)),
                              torch.from_numpy(np.array(labels_gesture_personne_scrambled, dtype=np.int64)))
        validation = TensorDataset(torch.from_numpy(np.array(examples_personne_scrambled_valid, dtype=np.float32)),
                                   torch.from_numpy(np.array(labels_gesture_personne_scrambled_valid, dtype=np.int64)))

        # 数据集切片成 batch
        trainLoader = DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
        validationLoader = DataLoader(validation, batch_size=128, shuffle=True, drop_last=True)

        # 将每个人的数据单独都以列表元素的形式存储在 list_train_dataloader / list_validation_dataloader 中
        list_train_dataloader.append(trainLoader)  # <list>
        list_validation_dataloader.append(validationLoader)

        human_number += 1
        # print("Shape training : ", np.shape(examples_personne_scrambled))
        # print("Shape valid : ", np.shape(examples_personne_scrambled_valid))

    return list_train_dataloader, list_validation_dataloader


def targetDataLoader(examples, labels,
                     examples_test0, labels_test0,
                     examples_test1, labels_test_1,
                     training_cycle=3):

    tgt_list_train_dataloader = []
    tgt_list_validation_dataloader = []
    human_number = 0

    for j in range(17):
        # print("CURRENT DATASET : ", j)
        tgtExamples_personne_training = []
        tgtLabels_gesture_personne_training = []
        tgtLabels_human_personne_training = []

        tgtExamples_personne_valid = []
        tgtLabels_gesture_personne_valid = []
        tgtLabels_human_personne_valid = []

        for k in range(len(examples[j])):
            if k < training_cycle * 7:
                tgtExamples_personne_training.extend(examples[j][k])
                tgtLabels_gesture_personne_training.extend(labels[j][k])
                tgtLabels_human_personne_training.extend(human_number * np.ones(len(labels[j][k])))
            else:
                tgtExamples_personne_valid.extend(examples[j][k])
                tgtLabels_gesture_personne_valid.extend(labels[j][k])
                tgtLabels_human_personne_valid.extend(human_number * np.ones(len(labels[j][k])))

        tgtTrain = TensorDataset(torch.from_numpy(np.array(tgtExamples_personne_training, dtype=np.float32)),
                                 torch.from_numpy(np.array(tgtLabels_gesture_personne_training, dtype=np.int64)))
        tgtValidation = TensorDataset(torch.from_numpy(np.array(tgtExamples_personne_valid, dtype=np.float32)),
                                   torch.from_numpy(np.array(tgtLabels_gesture_personne_valid, dtype=np.int64)))

        trainLoader = DataLoader(tgtTrain, batch_size=256, shuffle=True, drop_last=True)
        validationLoader = DataLoader(tgtValidation, batch_size=128, shuffle=True, drop_last=True)

        tgt_list_train_dataloader.append(trainLoader)
        tgt_list_validation_dataloader.append(validationLoader)

        human_number += 1

        # X_test_0, Y_test_0 = [], []
        # for k in range(len(examples_test0)):
        #     X_test_0.extend(examples_test0[j][k])
        #     Y_test_0.extend(labels_test0[j][k])
        # test_0 = TensorDataset(torch.from_numpy(np.array(X_test_0, dtype=np.float32)),
        #                        torch.from_numpy(np.array(Y_test_0, dtype=np.int64)))
        # validationLoader = DataLoader(test_0, batch_size=256, shuffle=True, drop_last=True)
        # tgt_list_validation_dataloader.append(validationLoader)
        #
        # X_test_1, Y_test_1 = [], []
        # for k in range(len(examples_test1)):
        #     X_test_1.extend(examples_test1[j][k])
        #     Y_test_1.extend(labels_test_1[j][k])

    return tgt_list_train_dataloader, tgt_list_validation_dataloader

