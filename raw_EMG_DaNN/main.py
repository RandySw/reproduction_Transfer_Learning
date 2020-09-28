import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import time
import os
import copy

from domain_adversarial_network import DaNNet, FeatureExtractor, DomainClassifier, LabelPredictor
from utliz import scramble, load_source_train_data, load_target_train_data, load_target_test_data
from functions import training, validation, testing


torch.cuda.set_device(0)
seed = 0
torch.manual_seed(seed)


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

# scalar
learning_rate = 1e-3
precision = 1e-8
best_loss_F = float('inf')
num_epochs = 10
lamb = 0.1
patience = 10
patience_increase = 10
loop_training_flag = False

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor(number_of_classes=7).cuda()
domain_classifier = DomainClassifier().cuda()

# %% Training Stage
source_train_dataloaders = {'train': list_source_train_dataloader, 'valid': list_source_valid_dataloader}
target_train_dataloaders = {'train': list_target_train_dataloader, 'valid': list_target_valid_dataloader}

for i, (target_subject_train_loader, target_subject_valid_loader) \
        in enumerate(zip(list_target_train_dataloader, list_target_valid_dataloader)):

    if i > 0 and not loop_training_flag:
        break

    # best_weights_feature_extractor = copy.deepcopy(feature_extractor.state_dict())
    # best_weights_domain_classifier = copy.deepcopy(domain_classifier.state_dict())
    # best_weights_label_predictor = copy.deepcopy(label_predictor.state_dict())

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('-' * 20+'\n')
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'valid']:
            # get a random order for the source training subject
            random_vec = np.arange(len(source_train_dataloaders[phase]))  # len(random_vec) = 19
            np.random.seed(0)
            np.random.shuffle(random_vec)

            for dataset_index in random_vec:
                if phase == 'train':
                    source_train_dataloader = source_train_dataloaders[phase][dataset_index]
                    target_train_dataloader = target_subject_train_loader

                    set_D_loss, set_F_loss, set_accuracy = training(source_train_dataloader,
                                                                    target_train_dataloader,
                                                                    feature_extractor,
                                                                    domain_classifier,
                                                                    label_predictor,
                                                                    lamb=lamb)
                    print('Train: subject:{}  D_loss:{:.6f}  F_loss:{:.6f}'.format(dataset_index, set_D_loss, set_F_loss,
                                                                                   set_accuracy))
                else:
                    source_valid_dataloader = source_train_dataloaders[phase][dataset_index]
                    target_valid_dataloader = target_subject_valid_loader

                    src_set_D_loss, src_set_C_loss, src_set_accuracy = validation(source_valid_dataloader,
                                                                                  feature_extractor,
                                                                                  domain_classifier,
                                                                                  label_predictor,
                                                                                  domain='source')

                    tag_set_D_loss, tag_set_C_loss, tag_set_accuracy = validation(target_valid_dataloader,
                                                                                  feature_extractor,
                                                                                  domain_classifier,
                                                                                  label_predictor,
                                                                                  domain='target')
                    D_loss = src_set_D_loss + tag_set_D_loss
                    F_loss = src_set_C_loss + lamb * D_loss
                    print('Valid: subject:{}  D_loss:{:.6f}  F_loss:{:.6f}  src_Acc:{:.6f}  tag_Acc:{:.6}'
                          .format(dataset_index, D_loss, F_loss, src_set_accuracy, tag_set_accuracy))


# %% Test stage:
# 暂时仅使用第一个受试者的 test data
if not loop_training_flag:
    target_subject_test0_dataloader = list_target_test0_dataloader[0]
    target_subject_test1_dataloader = list_target_test1_dataloader[0]

test0_D_loss, test0_C_loss, test0_acc = testing(target_subject_test0_dataloader,
                                                feature_extractor,
                                                domain_classifier,
                                                label_predictor)
test1_D_loss, test1_C_loss, test1_acc = testing(target_subject_test1_dataloader,
                                                feature_extractor,
                                                domain_classifier,
                                                label_predictor)

print('Test0:  D_loss:{:.6f}  C_loss:{:.6f}  Acc:{:.4f}'.format(test0_D_loss, test0_C_loss, test0_acc))
print('Test1:  D_loss:{:.6f}  C_loss:{:.6f}  Acc:{:.4f}'.format(test1_D_loss, test1_C_loss, test1_acc))










