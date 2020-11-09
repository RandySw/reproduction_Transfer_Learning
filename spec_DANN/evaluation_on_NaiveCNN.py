"""


"""


import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import collections
import time
import copy
import pandas as pd
import joblib
from scipy import signal

import data_process
from model import NaiveCNN


# ---------------------------------------- Test on Segments ------------------------------------------- #

print('Loading test segments...')
test_dataset = np.load("formatted_datasets/saved_total_users_test_dataset_xu_8_subjects.npy", encoding="bytes", allow_pickle=True)

total_test_data, total_test_labels = test_dataset

# pause
for index in range(len(total_test_labels)):
    # print('Sub{}:'.format(index))
    c = collections.Counter(total_test_labels[index])
    # print(c)

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
# print(test_data.shape)
test_label = np.array(new_label, dtype=np.int64)
# print(test_label.shape)
# time.sleep(100)

# numpy to tensor
test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))

# tensor to DataLoader
test_dataloader = DataLoader(test_data, batch_size=4096, shuffle=True, drop_last=True)
print('Test segments loaded. Batch_size = 4096.\n')

print('Loading trained Naive CNN network parameters...')
new_net = NaiveCNN().cuda()
new_net.load_state_dict(torch.load(r'saved_model\naive_cnn.pkl'))

new_net.eval()

total_batch_num = len(test_dataloader)
running_loss = 0.
correct_gesture_label, total_num = 0.0, 0.0
print('Start testing...\n')

for i, (data, label)in enumerate(test_dataloader):

    data = data.cuda()

    gesture_label = label.cuda()

    pred_label = new_net(data)

    batch_correct = torch.sum(torch.argmax(pred_label, dim=1) == gesture_label).item()

    correct_gesture_label += batch_correct
    total_num += data.shape[0]
    # print('\tBatch Accuracy: {:.4f}\n'.format(batch_correct / data.shape[0]))
    print('\rBatch: {} / {}       Batch Accuracy: {:.4f}'.format(i + 1, total_batch_num, batch_correct / data.shape[0]), end='')

total_accuracy = correct_gesture_label / total_num
print('\nTotal Segments Testing Accuracy: {:.4f}'.format(total_accuracy))


# --------------------------------------------- Test on Samples -------------------------------------------------- #

# --------------------------- functions -------------------------------- #

def cal_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window='hann',
                                                                                         scaling='spectrum')

    return spectrogram_of_vector, time_segment_sample, frequencies_samples


def cal_spectrogram(examples):
    """
    :param dataset: (189, (8, 52))
    :return:
    """

    # dataset_spectrogram = []

    # examples -> (8, 52)
    canals = []
    for electrode_vector in examples:   # electrode -> (52, )
        spectrogram_of_vector, time_segment_sample, frequencies_samples = \
            cal_spectrogram_vector(electrode_vector, npserseg=28, noverlap=20)
        # spectrogram_of_vector <ndarray: (15, 4)>
        # remove the low frequency signal as it's useless for sEMG (0-5Hz)
        spectrogram_of_vector = spectrogram_of_vector[1:]   # spectrogram_of_vector <ndarray: (14, 4)>
        canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))     # canals (8, (4, 14))

    # 将通道维度和时间维度作交换，得到按时序（4个时刻）划分的8通道肌电能量谱图
    example_to_classify = np.swapaxes(canals, 0, 1)     # example_to_classify <tuple: (4, 8, 14)>

    return example_to_classify


def calculate_spectrogram(dataset):
    """
    :param dataset: (189, (8, 52))
    :return:
    """

    dataset_spectrogram = []
    for examples in dataset:    # examples -> (8, 52)
        canals = []
        for electrode_vector in examples:   # electrode -> (52, )
            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                cal_spectrogram_vector(electrode_vector, npserseg=28, noverlap=20)
            # spectrogram_of_vector <ndarray: (15, 4)>
            # remove the low frequency signal as it's useless for sEMG (0-5Hz)
            spectrogram_of_vector = spectrogram_of_vector[1:]   # spectrogram_of_vector <ndarray: (14, 4)>
            canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))     # canals (8, (4, 14))

        # 将通道维度和时间维度作交换，得到按时序（4个时刻）划分的8通道肌电能量谱图
        example_to_classify = np.swapaxes(canals, 0, 1)     # example_to_classify <tuple: (4, 8, 14)>
        dataset_spectrogram.append(example_to_classify)

    return dataset_spectrogram


def preprocessing(data):
    """
    :param data:    8*400 emg data <class 'np.ndarray'>    400*8
    :return:        data instance after rectifying and filter  8*400
    """
    # scalar
    data = 2 * (data + 128) / 256 - 1

    # rectify
    data_processed = np.abs(data)

    # transpose (400, 8) -> (8, 400)
    data_processed = np.transpose(data_processed)

    # filter
    wn = 0.05
    order = 4
    b, a = signal.butter(order, wn, btype='low')
    data_processed = signal.filtfilt(b, a, data_processed)      # data_processed <class 'np.ndarray': 8*400>

    return data_processed       # <class 'np.ndarray'> 4*800


# ----------------------- Load Test Samples ------------------------ #

path = '../../data_x'

# user_name = ['fg', 'hechangxin', 'lili', 'shengranran', 'suziyi',
#              'tangyugui', 'wujialiang', 'xuzhenjin', 'yangkuo', 'yuetengfei',
#              'zhuofeng']

user_name = ['fg', 'lili', 'suziyi',
             'tangyugui', 'wujialiang', 'xuzhenjin', 'yuetengfei',
             'zhuofeng']

list_total_user_data = []
list_total_user_labels = []
for user in user_name:
    data = []
    labels = []
    user_path = path + '/' + str(user) + '/testing/'
    file_names = os.listdir(str(user_path))
    file_paths = [user_path + i for i in file_names]

    user_data = []
    user_labels = []
    count = 0

    for txt_path in file_paths:
        # print(txt_path)
        emg_array = data_process.txt2array(txt_path)     # <np.ndarray> (400, 8)
        label = data_process.label_indicator(txt_path)
        # print(label)
        user_data.append(emg_array)
        user_labels.append(label)
    list_total_user_data.append(user_data)
    list_total_user_labels.append(user_labels)

total_user_data = np.array(list_total_user_data, dtype=np.float32)      # <np.ndarray> (11, 180, 400, 8)
total_user_labels = np.array(list_total_user_labels, dtype=np.int64)     # <np.ndarray> (11, 180)


# ------------------------------------------ Test on Samples -------------------------------------------------------- #

window_size = 52
stride = 1
max_fit = 30
jump = 1
threshold = 60 / 128

print('Start Testing...')

for subject_index in range(len(total_user_data)):
    # subject_index = 6
    start_time = time.time()
    user_data = total_user_data[subject_index]
    user_labels = total_user_labels[subject_index]

    iter_times = []
    iter_activity_times = []
    label_prediction = []
    label_truth = []

    # 遍历每个subject的180个样本
    for sample_index in range(len(user_data)):

        sample_label = user_labels[sample_index]
        # print('Sample   {}     True label: {}'.format(sample_index, sample_label))

        emg_sample = user_data[sample_index]        # (400, 8)
        # print(emg_sample)
        # print(emg_sample.shape)
        emg_preprocessed = preprocessing(emg_sample)    # (8, 400)

        max_sliding_length = emg_preprocessed.shape[1] - window_size  # 348
        # print(max_sliding_length)

        gesture_number_vector = [0] * 14
        iter = 0
        iter_activity = 0

        while iter * stride + window_size <= emg_preprocessed.shape[1]:
            index_start = iter * jump               # 0
            index_end = iter * jump + window_size   # 52
            # print(index_start)
            # print(index_end)
            emg_window = emg_preprocessed[:, index_start: index_end]  # (8, 52)

            iter = iter + 1

            if sum(data_process.mav(emg_window)) < threshold:
                pred_gesture_label = 0    # relax
                pos_max = 0
                # print('pass')
            else:
                iter_activity = iter_activity + 1

                emg_spec = cal_spectrogram(emg_window)      # (4, 8, 14)
                # print(emg_spec.shape)
                emg_spec = np.array(emg_spec, dtype=np.float32)
                emg_spec_tensor = torch.from_numpy(emg_spec,)
                # emg_spec_tensor = emg_spec_tensor.cuda()
                input_tensor = emg_spec_tensor.view(1, 4, 8, 14).cuda()

                pred_gesture_label = new_net(input_tensor)

                # feature = feature_extractor(input_tensor)
                # pred_gesture_label = label_predictor(feature)

                pred_pos = torch.argmax(pred_gesture_label, dim=1)
                # print(pred_pos.item())

                if pred_pos.item() < 7:
                    gesture_number_vector[pred_pos.item()] = gesture_number_vector[int(pred_pos.item())] + 1

                max_num = max(gesture_number_vector)
                pos_max = gesture_number_vector.index(max_num)

                if max_num > max_fit:
                    break
                pos_max = 0

        iter_times.append(iter)
        iter_activity_times.append(iter_activity)

        final_prediction = pos_max
        # print('Final Prediction:  ', final_prediction)
        label_prediction.append(final_prediction)
        label_truth.append(sample_label)

    count = 0
    for i in range(len(label_prediction)):
        if label_prediction[i] == label_truth[i]:
            count += 1
    acc = count / len(label_prediction)
    print('Sub.{} accuracy: {:.4f}      Time Usage: {:.2f}s'.format(subject_index, acc, time.time() - start_time))
    # break



