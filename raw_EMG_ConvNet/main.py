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


def train_evaluate_model(source_data_training, source_labels_training,
                         target_data_training, target_labels_training,
                         target_data_test0, target_labels_test0,
                         target_data_test1, target_labels_test1,
                         training_cycle=4
                         ):

    # 调用 training_stage() 和 evaluation_stage() 分别对 DaNN 进行训练和测试
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
source_set_training = np.load("../formatted_datasets/saved_pre_training_dataset_spectrogram.npy",
                                    encoding="bytes", allow_pickle=True)
source_data_training, source_labels_training = source_set_training

# 原 training0
target_set_training = np.load('.../PytorchImplementation/formatted_datasets/saved_evaluation_dataset_training.npy',
                           encoding="bytes", allow_pickle=True)
target_data_training, target_labels_training = target_set_training

# 原 test0
target_set_test0 = np.load('.../PytorchImplementation/formatted_datasets/saved_evaluation_dataset_test0.npy',
                         encoding="bytes", allow_pickle=True)
target_data_test0, target_labels_test0 = target_set_test0

# 原 test1
target_set_test1 = np.load('.../PytorchImplementation/formatted_datasets/saved_evaluation_dataset_test1.npy',
                         encoding="bytes", allow_pickle=True)
target_data_test1, target_labels_test1 = target_set_test1


number_of_training_cycles = 4
number_of_training_repetitions = 5
learning_rate = 0.001
for training_cycle in range(2):
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
