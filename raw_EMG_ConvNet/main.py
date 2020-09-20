import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Function

import torchvision.transforms as transforms

import time

from domain_adversarial_network import FeatureExtractor, LabelPredictor, DomainClassifier


# init network components
feature_extractor = FeatureExtractor(number_of_classes=10).cuda()
label_predictor = LabelPredictor(number_of_classes=10).cuda()
domain_classifier = DomainClassifier().cuda()

# criterion configurations
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

# optimizer configurations
optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())


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


def train_evaluate_model(source_data_training, source_labels_training,
                         target_data_training, target_labels_training,
                         target_data_test0, target_labels_test0,
                         target_data_test1, target_labels_test1,
                         training_cycle=4
                         ):
    """
        调用 training_stage() 和 evaluation_stage() 分别对 DaNN 进行训练和测试
    """
    list_train_dataloader = []
    list_validation_dataloader = []
    human_number = 0

    # source set - training & validation data -> 3/4 training 1/4 validation
    for j in range(19):
        source_examples_personne_training = []
        source_labels_personne_training = []
        source_labels_human_personne_training = []

        source_examples_personne_valid = []
        source_labels_personne_valid = []
        source_labels_human_personne_valid = []

        # print(np.shape(source_data_training))
        # time.sleep(1000)
        for k in range(len(source_data_training[j])):
            # 前 3个cycle 的数据用作train
            if k < 21:
                source_examples_personne_training.extend(source_data_training[j][k])
                source_labels_personne_training.extend(source_labels_training[j][k])
                source_labels_human_personne_training.extend((human_number * np.ones(len(source_labels_training[j][k]))))
            # 最后 1个cycle 的数据用作valid
            else:
                source_examples_personne_valid.extend(source_data_training[j][k])
                source_labels_personne_valid.extend(source_labels_training[j][k])
                source_labels_human_personne_valid.extend(human_number * np.ones(len(source_labels_training[j][k])))

        human_number += 1
        print('source_examples_personne_training:   ', np.shape(source_examples_personne_training))
        print('source_examples_personne_valid:      ', np.shape(source_examples_personne_valid))

    print('-' * 50)

    human_number = 0
    # target set - training & validation data -> 3/4 training 1/4 validation
    for j in range(17):
        target_examples_personne_training = []
        target_labels_personne_training = []
        target_labels_human_personne_training = []

        target_examples_personne_valid = []
        target_labels_personne_valid = []
        target_labels_human_personne_valid = []

        for k in range(len(target_data_training[j])):
            if k < 21:
                target_examples_personne_training.extend(target_data_training[j][k])
                target_labels_personne_training.extend(target_labels_training[j][k])
                target_labels_human_personne_training.extend(human_number * np.ones(len(target_labels_training[j][k])))
            else:
                target_examples_personne_valid.extend(target_data_training[j][k])
                target_labels_personne_valid.extend(target_labels_training[j][k])
                target_labels_human_personne_valid.extend((human_number * np.ones(len(target_labels_training[j][k]))))

        human_number += 1
        print('target_examples_personne_training:   ', np.shape(target_examples_personne_training))
        print('target_examples_personne_valid:      ', np.shape(target_examples_personne_valid))






    accuracy_test0, accuracy_test1 = 0, 0
    return accuracy_test0, accuracy_test1


def training_stage():
    pass


def evaluation_stage():
    pass


"""

Load Data:
        source data:    19名受试者 -> 1 round -> 1 * 4 cycles
                        < pre-training0 >
        target data:    17名受试者 -> 3 round -> 3 * 4 cycles
                        < training0 >
                        < test0 >
                        < test1 >
                        
For DaNN training:      < pre-training0 > & < training 0 >  -> 2 * 4 cycles

For DaNN testing:       < test0 > & < test1 >               -> 2 * 4 cycles

"""
# %% load data
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


number_of_training_cycles = 4
number_of_training_repetitions = 1
learning_rate = 0.001
for training_cycle in range(1):
    test_0 = []
    test_1 = []

    # all experimental result are reported as an average of 5 repetitions
    for repetition_time in range(number_of_training_repetitions):
        accuracy_test0, accuracy_test1 = train_evaluate_model(source_data_training, source_labels_training,
                                                              target_data_training, target_labels_training,
                                                              target_data_test0, target_labels_test0,
                                                              target_data_test1, target_labels_test1,
                                                              training_cycle=training_cycle)
        test_0.append(accuracy_test0)
        test_1.append(accuracy_test1)
        print('TEST 0 CURRENT:   ', test_0)
        print('TEST 1 CURRENT:   ', test_1)
        print('CURRENT AVERAGE: ',)
