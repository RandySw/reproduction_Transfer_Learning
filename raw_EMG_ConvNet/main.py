import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Function

import torchvision.transforms as transforms

from domain_adversarial_network import FeatureExtractor, LabelPredictor, DomainClassifier


# init network components
feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

# criterion configurations
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

# optimizer configurations
optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())


def train_evaluate_epoch():
    pass





# %% load data
dataset_training = np.load('.../PytorchImplementation/formatted_datasets/saved_evaluation_dataset_training.npy',
                           encoding="bytes", allow_pickle=True)
example_training, labels_training = dataset_training

datasets_test0 = np.load('.../PytorchImplementation/formatted_datasets/saved_evaluation_dataset_test0.npy',
                         encoding="bytes", allow_pickle=True)
example_test0, labels_test0 = datasets_test0

datasets_test1 = np.load('.../PytorchImplementation/formatted_datasets/saved_evaluation_dataset_test1.npy',
                         encoding="bytes", allow_pickle=True)
example_test1, labels_test1 = datasets_test1

number_of_training_cycles = 2
number_of_training_repetitions = 5
learning_rate = 0.001
for training_cycle in range(2):
    test_0 = []
    test_1 = []

    # all experimental result are reported as an average of 5 repetitions
    for repetition_time in range(number_of_training_repetitions):
        accuracy_test0, accuracy_test1 = train_and_evaluation_model(example_training, labels_training,
                                                                    example_test0, labels_test0,
                                                                    example_test1, labels_test1,
                                                                    leanring_rate=learning_rate,
                                                                    training_cycle=training_cycle
                                                                    )
        test_0.append(accuracy_test0)
        test_1.append(accuracy_test1)
        print('TEST 0 CURRENT:   ', test_0)
        print('TEST 1 CURRENT:   ', test_1)
        print('CURRENT AVERAGE: ',)