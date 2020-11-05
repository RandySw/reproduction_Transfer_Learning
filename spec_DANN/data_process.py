import os
import time
import numpy as np
import pandas as pd
from scipy import signal


def txt2array(txt_path):
    """
    :param txt_path:    specific path of a single txt file
    :return:            1-dimension preprocessed vector <class 'np.ndarray'>
                        of the input txt file
    """
    table_file = pd.read_table(txt_path, header=None)
    txt_file = table_file.iloc[:, :]
    txt_array = txt_file.values

    return txt_array


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


def detect_muscle_activity(emg_data):
    """
    :param      emg_date: 8 channels of emg data -> 8*400
    :return:
                index_start: star index of muscle activation region
                index_end:   end index of muscle activation region
    """

    # plot emg_data
    # plt.plot(emg_data.transpose())
    # plt.show()

    fs = 200        # sampling frequency
    min_activation_length = 50
    num_frequency_of_spec = 50
    hamming_window_length = 25
    overlap_samples = 10
    threshold_along_frequency = 18

    sumEMG = emg_data.sum(axis=0)   # sum 8 channel data into one vector
    # plt.plot(sumEMG)
    # plt.show()

    f, time, Sxx = signal.spectrogram(sumEMG, fs=fs,
                                   window='hamming',
                                   nperseg=hamming_window_length,
                                   noverlap=overlap_samples,
                                   nfft=num_frequency_of_spec,
                                   detrend=False,
                                   mode='complex')

    # 43.6893
    # test plot
    Sxx = Sxx * 43.6893

    # spec_values = abs(Sxx)
    # plt.pcolormesh(time, f, spec_values, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    #
    # plt.plot(sumEMG)
    # plt.show()

    spec_values = abs(Sxx)
    spec_vector = spec_values.sum(axis=0)

    # plt.plot(spec_vector)
    # plt.show()

    # 使用np.diff 求差分
    # indicated_vector 标记序列中哪些位置的强度值高于阈值
    indicated_vector = np.zeros(shape=(spec_vector.shape[0] + 2),)

    for index, element in enumerate(spec_vector):
        if element > threshold_along_frequency:
            indicated_vector[index+1] = 1

    # print('indicated_vector: %s' % str(indicated_vector))
    # print('indicated_vector.shape: %s' % str(indicated_vector.shape))

    index_greater_than_threshold = np.abs(np.diff(indicated_vector))

    if index_greater_than_threshold[-1] == 1:
        index_greater_than_threshold[-2] = 1

    # 删去最后一个元素
    index_greater_than_threshold = index_greater_than_threshold[:- 1]

    # 找出非零元素的序号
    index_non_zero = np.where(index_greater_than_threshold == 1)[0]

    index_of_samples = np.floor(fs * time - 1)
    num_of_index_non_zero = index_non_zero.shape[0]

    length_of_emg = sumEMG.shape[0]
    # print('length of emg : %f points' % length_of_emg)

    # find the start and end indexes
    if num_of_index_non_zero == 0:
        index_start = 1
        index_end = length_of_emg
    elif num_of_index_non_zero == 1:
        index_start = index_of_samples[index_non_zero]
        index_end = length_of_emg
    else:
        index_start = index_of_samples[index_non_zero[0]]
        index_end = index_of_samples[index_non_zero[-1] - 1]

    num_extra_samples = 25
    index_start = max(1, index_start - num_extra_samples)
    index_end = min(length_of_emg, index_end + num_extra_samples)

    if (index_end - index_start) < min_activation_length:
        index_start = 0
        index_end = length_of_emg - 1

    # print(index_start)
    # print(index_end)

    # return spec_vector, time, spec_values
    return index_start, index_end


def label_indicator(path):
    label = None
    if 'relax' in path:
        label = 0
    elif 'fist' in path:
        label = 1
    elif 'fingersSpread' in path:
        label = 2
    elif 'doubleTap' in path:
        label = 3
    elif 'waveIn' in path:
        label = 4
    elif 'waveOut' in path:
        label = 5
    return label


def cal_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window='hann',
                                                                                         scaling='spectrum')

    return spectrogram_of_vector, time_segment_sample, frequencies_samples


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


def mav(emg_data):
    """
    :param emg_data:    <class 'np.ndarray'>    (8, 40)
    :return:            mav feature vector of the input emg matrix     (8, )
    """
    mav_result = np.mean(abs(emg_data), axis=1)

    return mav_result
