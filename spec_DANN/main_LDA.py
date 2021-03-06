from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

from scipy import signal
import time
import os
import collections
import copy
import data_process
import joblib

# --------------------------------- Data Pre-process --------------------------------------- #


train_dataset = np.load('formatted_datasets/saved_total_users_train_dataset_xu_8_subjects.npy',
                  encoding='bytes', allow_pickle=True)

# 二者都为(11, )，其中total_data 内部元素为<np.ndarry> 而total_labels内部元素为<list>
total_train_data, total_train_labels = train_dataset

# 查看数据包含的样本的标签类别和数量
for index in range(len(total_train_labels)):
    print('Sub{}:'.format(index))                       # subject:  0-10
    c = collections.Counter(total_train_labels[index])  # 1/ 2/ 7/ 8 subject 的样本过少 | 2/ 4/ 5 样本过多
    print(c)                                            # 可以选择性删除以上两组


list_total_train_data = []
list_subject_label = []
for subject_index in range(len(total_train_data)):
    list_total_train_data.extend(total_train_data[subject_index])
    list_subject_label.extend([subject_index] * len(total_train_data[subject_index]))

list_total_train_labels = []
for subject_index in range(len(total_train_labels)):
    list_total_train_labels.extend(total_train_labels[subject_index])

# 检查数据包含的样本的标签类别和数量是否与合并前保持一致
c = collections.Counter(list_subject_label)
print(c)

# shuffle the data set
random_vector = np.arange(len(list_subject_label))
np.random.shuffle(random_vector)
new_data = []
new_gesture_label = []
new_subject_label = []
for i in random_vector:
    new_data.append(list_total_train_data[i])
    new_gesture_label.append(list_total_train_labels[i])
    new_subject_label.append(list_subject_label[i])

train_test_ratio = 0.7
sep = int(train_test_ratio * len(random_vector))

# data split
train_data = new_data[:sep]
test_data = new_data[sep:]

train_gesture_labels = new_gesture_label[:sep]
test_gesture_labels = new_gesture_label[sep:]

train_subject_labels = new_subject_label[:sep]
test_subject_labels = new_subject_label[sep:]

# list to numpy
train_data = np.array(train_data, dtype=np.float32)
test_data = np.array(test_data, dtype=np.float32)

train_gesture_labels = np.array(train_gesture_labels, dtype=np.int64)
test_gesture_labels = np.array(test_gesture_labels, dtype=np.int64)

# train_subject_labels = np.array(train_subject_labels, dtype=np.float32)
# test_subject_labels = np.array(test_subject_labels, dtype=np.float32)
train_subject_labels = np.array(train_subject_labels, dtype=np.int64)
test_subject_labels = np.array(test_subject_labels, dtype=np.int64)


# ------------------------------------------ Training -------------------------------------------- #

train_data = train_data.reshape(train_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)


lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
lda.fit(train_data, train_gesture_labels)

pred_gesture_label = lda.predict(test_data)

correct_gesture_label = sum(pred_gesture_label == test_gesture_labels)

acc = correct_gesture_label / pred_gesture_label.shape[0]

print('Train Accuracy: {:.4f}'.format(acc))

joblib.dump(lda, r'saved_model\LDA_8_subjects.pkl')


# ----------------------------------------- Training on Segments ----------------------------------------- #

print('Loading test segments...')
test_dataset = np.load("formatted_datasets/saved_total_users_test_dataset_xu_8_subjects.npy", encoding="bytes", allow_pickle=True)

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

test_data = test_data.reshape(test_data.shape[0], -1)

# validation
new_lda = joblib.load(r'saved_model\LDA_8_subjects.pkl')

pred_gesture_label = lda.predict(test_data)

correct_gesture_label = sum(pred_gesture_label == test_label)

acc = correct_gesture_label / pred_gesture_label.shape[0]

print('Segments Test Accuracy: {:.4f}'.format(acc))     # ACC = 0.5992


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

# test stage
window_size = 52
stride = 1
max_fit = 20
jump = 1
threshold = 60 / 128


for subject_index in range(len(total_user_data)):
    start_time = time.time()
    # subject_index = 6
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

        # if sample_label == 0:
        #     # relax 则跳出识别循环
        #     break

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

            emg_window = emg_preprocessed[:, index_start: index_end]  # (8, 52)

            iter = iter + 1

            if sum(data_process.mav(emg_window)) < threshold:
                pred_gesture_label = 0    # relax
                # print('pass')
            else:
                iter_activity = iter_activity + 1

                emg_spec = cal_spectrogram(emg_window)      # (4, 8, 14)

                emg_spec = np.array(emg_spec, dtype=np.float32)
                emg_spec = emg_spec.reshape(1, 448)

                pred_label = new_lda.predict(emg_spec)

                # correct_gesture_label = sum(pred_gesture_label == test_label)

                if correct_gesture_label < 7:
                    gesture_number_vector[pred_label] = gesture_number_vector[int(pred_label)] + 1

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

