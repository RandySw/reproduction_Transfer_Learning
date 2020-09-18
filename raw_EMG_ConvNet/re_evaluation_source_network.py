import numpy as np
import re_source_network_raw_emg
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
import copy


def confusion_matrix(pred, Y, number_class=7):
    pass


def scramble(examples, labels, second_labels=[]):

    tmp1 = examples
    tmp2 = labels
    return tmp1, tmp2


def calculate_fitness(examples_training, labels_training,
                      examples_test0, labels_test0,
                      examples_test1, labels_test1,
                      learning_rate=.1, training_cycle=4):
    accuracy_test0 = []
    accuracy_test1 = []

    for j in range(17):
        print('CURRENT DATASET: ', j)
        examples_personne_training = []
        labels_gesture_personne_training = []

        for k in range(len(examples_training[j])):
            if k < training_cycle * 7:
                examples_personne_training.extend(examples_training[j][k])
                labels_gesture_personne_training.extend(labels_training[j][k])

        X_test_0, Y_test_0 = [], []
        for k in range(len(examples_test0)):
            X_test_0.extend(examples_test0[j][k])
            Y_test_0.extend(labels_test0[j][k])

        X_test_1, Y_test_1 = [], []
        for k in range(len(examples_test1)):
            X_test_1.extend(examples_test1[j][k])
            Y_test_1.extend(labels_test1[j][k])

        print(np.shape(examples_personne_training))
        examples_personne_scrambled, labels_gesture_personne_srambled \
            = scramble(examples_personne_training, labels_gesture_personne_training)
        valid_examples = examples_personne_scrambled[0: int(len(examples_personne_scrambled) * 0.1)]
        labels_valid = labels_gesture_personne_srambled[0: int(len())]






