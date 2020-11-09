"""

Evaluation_on_test_segments:
    将test sample中 11 * 30 个手势的样本切片
    评估 main_muliti_domain 中得到模型的性能


"""

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import collections
import time
import copy

import data_process
from model import FeatureExtractor, LabelPredictor, DomainClassifier


# --------------------------------------- Load Test Data -------------------------------------------------- #

# path = '../../data_x'
# user_name = ['fg', 'hechangxin', 'lili', 'shengranran', 'suziyi',
#              'tangyugui', 'wujialiang', 'xuzhenjin', 'yangkuo', 'yuetengfei',
#              'zhuofeng']
#
# number_of_classes = 6
# window_size = 52
#
#
# # load testing data -> 30 samples / gesture
# list_total_user_test_data = []
# list_total_user_test_labels = []
# for user in user_name:
#     data = []
#     labels = []
#     user_path = path + '/' + str(user) + '/testing/'
#     file_names = os.listdir(str(user_path))
#     file_paths = [user_path + i for i in file_names]
#     # print(file_paths)
#
#     # print(user + ':')
#     # print(file_names)
#     # print()
#
#     user_data = None
#     user_labels = []
#     count = 0
#
#     for txt_path in file_paths:
#         print(txt_path)
#         emg_array = data_process.txt2array(txt_path)     # <np.ndarray> (400, 8)
#
#         label = data_process.label_indicator(txt_path)
#         print(label)
#
#         # 每人仅使用12个relax的样本
#         if label == 0:
#             if count == 12:
#                 print('relax sample skipped')
#                 continue
#             count += 1
#
#         # pre-processing
#         single_sample_preprocessed = data_process.preprocessing(emg_array)       # <np.ndarray> (8, 400)
#         # detect muscle activation region
#         index_start, index_end = data_process.detect_muscle_activity(single_sample_preprocessed)
#         activation_emg = single_sample_preprocessed[:, np.int(index_start): np.int(index_end)]  # (8, active_length)
#         activation_length = index_end - index_start
#
#         total_silding_size = activation_emg.shape[1] - window_size
#         segments_set = []
#
#         for index in range(total_silding_size):
#             emg_segment = activation_emg[:, index: index + window_size]   # image_emg (8, window_length)
#             segments_set.append(emg_segment)
#         print(len(segments_set))
#
#         single_sample_spectrogram = data_process.calculate_spectrogram(segments_set)         # <list>
#         single_sample_spectrogram_data = np.array(single_sample_spectrogram)    # <np.ndarray> (sample_number, 4, 8, 14)
#         # print(type(single_sample_spectrogram_data))
#         # print(np.shape(single_sample_spectrogram_data))
#
#         if user_data is None:
#             user_data = single_sample_spectrogram_data
#         else:
#             user_data = np.concatenate((user_data, single_sample_spectrogram_data), axis=0)  # <np.ndarray> (n,4,8,14)
#         # user_data = np.c_[user_data, single_sample_spectrogram_data]    # <list> (6 * 35, sample_number, 4, 8, 14)
#         user_labels = user_labels + [label] * single_sample_spectrogram_data.shape[0]       # <list> (n, )
#     list_total_user_test_data.append(user_data)
#     list_total_user_test_labels.append(user_labels)
#
# total_users_test_dataset = [list_total_user_test_data, list_total_user_test_labels]
# print('Test dataset formatted finished.')
#
# np.save("formatted_datasets/saved_total_users_test_dataset_xu.npy", total_users_test_dataset)
# print('Train dataset saved.')

test_dataset = np.load("formatted_datasets/saved_total_users_test_dataset_xu_8_subjects.npy", encoding="bytes", allow_pickle=True)
print(np.shape(test_dataset))


# -------------------------------------------- Load Model ----------------------------------------------------- #
feature_extractor = FeatureExtractor().cuda()
domain_classifier = DomainClassifier().cuda()
label_predictor = LabelPredictor().cuda()

feature_extractor.load_state_dict(torch.load(r'saved_model\feature_extractor_CE_8_subjects.pkl'))
domain_classifier.load_state_dict(torch.load(r'saved_model\domain_classifier_CE_8_subjects.pkl'))
label_predictor.load_state_dict(torch.load(r'saved_model\label_predictor_CE_8_subjects.pkl'))

print(feature_extractor)
print(domain_classifier)
print(label_predictor)

print('Model loaded.')
# time.sleep(1000)

# ------------------------------------------ Load Test Data --------------------------------------------------- #
total_test_data, total_test_labels = test_dataset

# pause
for index in range(len(total_test_labels)):
    print('Sub{}:'.format(index))
    c = collections.Counter(total_test_labels[index])
    print(c)

list_total_test_data = []
for subject_index in range(len(total_test_data)):
    list_total_test_data.extend(total_test_data[subject_index])

list_total_test_labels = []
for subject_index in range(len(total_test_labels)):
    list_total_test_labels.extend(total_test_labels[subject_index])

# shuffle the data set
random_vector = np.arange(len(list_total_test_labels))
np.random.shuffle(random_vector)
new_data = []
new_label = []
for i in random_vector:
    new_data.append(list_total_test_data[i])
    new_label.append(list_total_test_labels[i])

# list to numpy
test_data = np.array(new_data, dtype=np.float32)
print(test_data.shape)
test_label = np.array(new_label, dtype=np.int64)
print(test_label.shape)
# time.sleep(100)

# numpy to tensor
test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))

# tensor to DataLoader
test_dataloader = DataLoader(test_data, batch_size=4096, shuffle=True, drop_last=True)

# ---------------------------------------- Testing Stage ------------------------------------------- #
feature_extractor.eval()
label_predictor.eval()
domain_classifier.eval()


correct_gesture_label, total_num = 0., 0.
total_batch_num = len(test_dataloader)

print('Start testing. BatchSize=4096 ')

for i, (data, label)in enumerate(test_dataloader):

    print('Batch: {} / {}'.format(i + 1, total_batch_num))

    data = data.cuda()
    gesture_label = label.cuda()

    feature = feature_extractor(data)
    pred_gesture_label = label_predictor(feature)
    # print(pred_gesture_label[0])
    # time.sleep(100)

    batch_correct = torch.sum(torch.argmax(pred_gesture_label, dim=1) == gesture_label).item()

    correct_gesture_label += batch_correct
    total_num += data.shape[0]
    print('\tBatch Accuracy: {:.4f}\n'.format(batch_correct / data.shape[0]))

total_accuracy = correct_gesture_label / total_num
print('Total Testing Accuracy: {:.4f}'.format(total_accuracy))





# TODO:
#       1. 加载训练完成的模型，对test的30*11个样本进行测试
#           1.1 切片测试：将所有的样本重新进行预处理，提取活动区间，滑窗切片，然后进行识别计算精度 （先）
#           1.2 完整样本测试：使用之前matlab代码里的多窗投票的形式    （后）




