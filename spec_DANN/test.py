import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

from functions import scramble, load_source_train_data, load_target_train_data, load_target_test_data
from functions import train, valid, test, training, validation, testing
from model_spec import DANNSpect, CNNSpect


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

torch.cuda.set_device(0)
seed = 0
torch.manual_seed(seed)


src_set_training = np.load('formatted_datasets/saved_pre_training_dataset_spectrogram.npy',
                              encoding='bytes', allow_pickle=True)
src_data_training, src_labels_training = src_set_training

tgt_set_training = np.load('formatted_datasets/saved_evaluation_dataset_training.npy',
                              encoding='bytes', allow_pickle=True)
tgt_data_training, tgt_labels_training = tgt_set_training

tgt_set_test0 = np.load('formatted_datasets/saved_evaluation_dataset_test0.npy',
                        encoding='bytes', allow_pickle=True)
tgt_data_test0, tgt_labels_test0 = tgt_set_test0

tgt_set_test1 = np.load('formatted_datasets/saved_evaluation_dataset_test1.npy',
                        encoding='bytes', allow_pickle=True)
tgt_data_test1, tgt_labels_test1 = tgt_set_test1


list_src_train_dataloader, list_src_valid_dataloader = load_source_train_data(src_data_training, src_labels_training)

list_tgt_train_dataloader, list_tgt_valid_dataloader = load_target_train_data(tgt_data_training, tgt_labels_training)

list_tgt_test0_dataloader = load_target_test_data(tgt_data_test0, tgt_labels_test0)

list_tgt_test1_dataloader = load_target_test_data(tgt_data_test1, tgt_labels_test1)

src_data_train = []
for i in range(len(list_src_train_dataloader)):
    src_data_train.extend(list_src_train_dataloader[i])