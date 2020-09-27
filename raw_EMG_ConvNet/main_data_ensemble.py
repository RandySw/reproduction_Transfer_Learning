import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt

import time
import os
import copy

from domain_adversarial_network import DaNNet, FeatureExtractor, DomainClassifier, LabelPredictor
from utliz import scramble, load_source_train_data, load_target_train_data, load_target_test_data
from functions import training, validation, testing


torch.cuda.set_device(0)
seed = 0
torch.manual_seed(seed)

os.environ['KMP_DUPLICATE_LIB_OK']= 'True'


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


src_data_train = []
for i in range(len(list_source_train_dataloader)):
    src_data_train.extend(list_source_train_dataloader[i])

src_data_valid = []
for i in range(len(list_source_valid_dataloader)):
    src_data_valid.extend(list_source_valid_dataloader[i])

tag_data_train = []
for i in range(len(list_target_train_dataloader)):
    i = 0
    tag_data_train.extend(list_target_train_dataloader[i])

tag_data_valid = []
for i in range(len(list_target_valid_dataloader)):
    i = 0
    tag_data_valid.extend(list_target_valid_dataloader[i])

tag_data_test0 = []
# for i in range(len(list_target_test0_dataloader)):
#     tag_data_test0.extend(list_target_test0_dataloader[i])
tag_data_test0.extend(list_target_test0_dataloader[0])

tag_data_test1 = []
# for i in range(len(list_target_test1_dataloader)):
#     tag_data_test1.extend(list_target_test1_dataloader[i])
tag_data_test1.extend(list_target_test1_dataloader[0])


feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor(number_of_classes=7).cuda()
domain_classifier = DomainClassifier().cuda()


optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())

precision = 1e-6
scheduler_F = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_F, mode='min', factor=.3, patience=5,
                                                 verbose=True, eps=precision)

scheduler_C = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_C, mode='min', factor=.3, patience=5,
                                                 verbose=True, eps=precision)

scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_D, mode='min', factor=.3, patience=5,
                                                 verbose=True, eps=precision)

best_acc = 0.0
best_F_weights = copy.deepcopy(feature_extractor.state_dict())
best_C_weights = copy.deepcopy(label_predictor.state_dict())
best_D_weights = copy.deepcopy(domain_classifier.state_dict())

epoch_num = 100
patience = 20
patience_increase = 20

eva_loss = []
src_tag_d_train_loss = []
src_d_valid_loss = []
tag_d_valid_loss = []

src_train_acc = []

src_valid_acc = []
tag_valid_acc = []
tag_test0_acc = []
tag_test1_acc = []

for epoch in range(epoch_num):
    epoch_start_time = time.time()

    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

    train_D_loss, train_F_loss, train_src_acc = training(src_data_train, src_data_valid,
                                                         feature_extractor,
                                                         domain_classifier,
                                                         label_predictor,
                                                         optimizer_F, optimizer_C, optimizer_D,
                                                         lamb=0.1)

    val_src_D_loss, val_src_C_loss, val_src_acc = validation(src_data_valid,
                                                             feature_extractor,
                                                             domain_classifier,
                                                             label_predictor,
                                                             domain='source')

    val_tag_D_loss, val_tag_C_loss, val_tag_acc = validation(tag_data_valid,
                                                             feature_extractor,
                                                             domain_classifier,
                                                             label_predictor,
                                                             domain='target')

    test0_D_loss, test0_C_loss, test0_acc = testing(tag_data_test0,
                                                    feature_extractor,
                                                    domain_classifier,
                                                    label_predictor)

    test1_D_loss, test1_C_loss, test1_acc = testing(tag_data_test1,
                                                    feature_extractor,
                                                    domain_classifier,
                                                    label_predictor)

    epoch_val_loss = val_src_C_loss - (val_src_D_loss + val_tag_D_loss)

    scheduler_F.step(epoch_val_loss)
    scheduler_C.step(epoch_val_loss)
    scheduler_D.step(epoch_val_loss)

    print('Training:    D_loss:{:.6f}  F_loss:{:.6f}  src_Acc:{:.6f}'
          .format(train_D_loss, train_F_loss, train_src_acc))
    print('Validation Src:    src_D_loss:{:.6f}  src_C_loss:{:.6f}  src_Acc:{:.6f}'
          .format(val_src_D_loss, val_src_C_loss, val_src_acc))
    print('Validation Tag:    tag_D_loss:{:.6f}  tag_C_loss:{:.6f}  tag_Acc:{:.6f}'
          .format(val_tag_D_loss, val_tag_C_loss, val_tag_acc))
    print('Test0:       D_loss:{:.6f}  C_loss:{:.6f}  Acc:{:.6f}'
          .format(test0_D_loss, test0_C_loss, test0_acc))
    print('Test1:       D_loss:{:.6f}  C_loss:{:.6f}  Acc:{:.6f}'
          .format(test1_D_loss, test1_C_loss, test1_acc))

    if val_tag_acc + precision > best_acc:
        print('New Best Target Validation Accuracy: {:.4f}'.format(val_tag_acc))
        best_acc = val_tag_acc
        best_F_weights = copy.deepcopy(feature_extractor.state_dict())
        best_C_weights = copy.deepcopy(label_predictor.state_dict())
        best_D_weights = copy.deepcopy(domain_classifier.state_dict())
        patience = patience_increase + epoch
        print('Patience: ', patience)

    print('epoch time usage: {:.2f}s'.format(time.time() - epoch_start_time))
    print()

    eva_loss.append(epoch_val_loss)
    src_tag_d_train_loss.append(train_D_loss)
    src_d_valid_loss.append(val_src_D_loss)
    tag_d_valid_loss.append(val_tag_D_loss)

    src_train_acc.append(train_src_acc)
    src_valid_acc.append(val_src_acc)
    tag_valid_acc.append(val_tag_acc)
    tag_test0_acc.append(test0_acc)
    tag_test1_acc.append(test1_acc)

    if epoch > patience:
        break

print('-' * 20 + '\n' + '-' * 20)
print('Best Best Target Validation Accuracy: {:.4f}'.format(best_acc))

torch.save(best_F_weights, r'best_weights/best_feature_extractor_weights.pt')
torch.save(best_C_weights, r'best_weights/best_label_predictor_weights.pt')
torch.save(best_D_weights, r'best_weights/best_domain_classifier_weights.pt')


# plotting curves
plt.plot(eva_loss)
plt.plot(src_tag_d_train_loss)
plt.plot(src_d_valid_loss)
plt.plot(tag_d_valid_loss)
plt.title('Loss - Epoch')
plt.legend(['eva_loss', 'src_tag_d_train_loss', 'src_d_valid_loss', 'tag_d_valid_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(src_train_acc)
plt.plot(src_valid_acc)
plt.plot(tag_valid_acc)
plt.plot(tag_test0_acc)
plt.plot(tag_test1_acc)
plt.title('Accuracy - Epoch')
plt.legend(['src_train_acc', 'src_valid_acc', 'tag_valid_acc', 'tag_test0_acc', 'tag_test1_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('Accuracy-Epoch.jpg')


