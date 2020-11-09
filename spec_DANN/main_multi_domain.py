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
from model import FeatureExtractor, DomainClassifier, LabelPredictor
# from model_spec import DANNSpect

'''
本方案使用 multi-domain label 的形式
    即每位Subject 分配一个domain label
    不分source / target domain
    所有的subject的data都混在一起进行训练，而不适用leave-one-subject-oue的形式
    

'''
# ------------------------------- Network Structure -------------------------------------- #
# 带TL的 DANN 模型


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        # ctx.alpha = 0.1
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # ctx.alpha = 0.1
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANNSpect(nn.Module):
    def __init__(self, number_of_class,):
        super(DANNSpect, self).__init__()
        # 下面两条是干什么用的？
        self._input_batchnorm = nn.BatchNorm2d(4, eps=1e-4)
        self._input_prelu = nn.PReLU(4)

        self.Conv1_1 = nn.Sequential(
            nn.Conv2d(2, 12, kernel_size=(4, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.3),
        )

        self.Conv1_2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=(3, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.3),
        )

        self.Conv2_1 = nn.Sequential(
            nn.Conv2d(2, 12, kernel_size=(4, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.3),
        )

        self.Conv2_2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=(3, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.3),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=(3, 3)),
            nn.BatchNorm2d(24),
            nn.PReLU(24),
            nn. Dropout2d(0.3),
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(24 * 1 * 8, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.3),

            nn.Linear(512, number_of_class)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(24 * 1 * 8, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(256),
            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

        self.initialize_weights()

    def first_parallel(self, input):
        feature = self.Conv1_1(input)
        feature = self.Conv1_2(feature)
        return feature

    def second_parallel(self, input):
        feature = self.Conv2_1(input)
        feature = self.Conv2_2(feature)
        return feature

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x, alpha=None):
        x = self._input_prelu(self._input_batchnorm(x))     # (512, 4, 8, 14)

        input_1 = x[:, 0:2, :, :]
        # print(input_1.shape)
        input_2 = x[:, 2:4, :, :]
        # print(input_2.shape)

        branch_1 = self.first_parallel(input_1)     # (512, 12, 3, 10)
        branch_2 = self.second_parallel(input_2)    # (512, 12, 3, 10)

        merged_branch = branch_1 + branch_2         # (512, 12, 3, 10)

        after_conv = self.Conv3(merged_branch)      # (512, 24, 1, 8)

        flatten_tensor = after_conv.view(-1, 24 * 1 * 8)    #

        # reverse_feature = ReverseLayerF.apply(flatten_tensor, alpha)
        label_output = self.label_classifier(flatten_tensor)
        # domain_output = self.domain_classifier(reverse_feature)
        domain_output = self.domain_classifier(flatten_tensor)

        return label_output, domain_output


# ---------------------------------- Load Train Data ------------------------------------------- #

# path = '../../data_x'
# user_name = ['fg', 'hechangxin', 'lili', 'shengranran', 'suziyi',
#              'tangyugui', 'wujialiang', 'xuzhenjin', 'yangkuo', 'yuetengfei',
#              'zhuofeng']
#
# number_of_classes = 6
# window_size = 52
#
# # load training data -> 5 samples / gesture
# list_total_user_data = []
# list_total_user_labels = []
# for user in user_name:
#     data = []
#     labels = []
#     user_path = path + '/' + str(user) + '/training/'
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
#         # 每人仅使用2个relax的样本
#         if label == 0:
#             if count == 2:
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
#             user_data = np.concatenate((user_data, single_sample_spectrogram_data), axis=0) # <np.ndarray>(n,4,8,14)
#         # user_data = np.c_[user_data, single_sample_spectrogram_data]    # <list> (6 * 35, sample_number, 4, 8, 14)
#         user_labels = user_labels + [label] * single_sample_spectrogram_data.shape[0]       # <list> (n, )
#     list_total_user_data.append(user_data)
#     list_total_user_labels.append(user_labels)
#
#
# total_users_train_dataset = [list_total_user_data, list_total_user_labels]
# print('Train dataset formatted finished.')
#
# np.save("formatted_datasets/saved_total_users_train_dataset_xu.npy", total_users_train_dataset)
# print('Train dataset saved.')


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

# numpy to tensor
train_data = TensorDataset(torch.from_numpy(train_data),
                           torch.from_numpy(train_gesture_labels),
                           torch.from_numpy(train_subject_labels))
test_data = TensorDataset(torch.from_numpy(test_data),
                          torch.from_numpy(test_gesture_labels),
                          torch.from_numpy(test_subject_labels))

# tensor to DataLoader
train_dataloader = DataLoader(train_data, batch_size=1024, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=1024, shuffle=True, drop_last=True)


# -------------------------------------- Training Stage ------------------------------------------- #

precision = 1e-8

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(label_predictor.parameters())

scheduler_F = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_F, mode='min', factor=0.1, patience=8,
                                                 verbose=True, eps=precision)
scheduler_C = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_C, mode='min', factor=0.1, patience=8,
                                                   verbose=True, eps=precision)
scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_D, mode='min', factor=0.1, patience=8,
                                                   verbose=True, eps=precision)

epoch_num = 200   # 120
patience = 15   # 12
patience_increase = 15  # 12
best_acc = 0
best_loss = float('inf')

for epoch in range(epoch_num):

    epoch_start_time = time.time()
    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

    len_dataloader = len(train_dataloader)

    running_loss, running_F_loss, running_D_loss, running_C_loss, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
    correct_gesture_label, correct_subject_label = 0.0, 0.0

    # net.train()
    feature_extractor.train()
    label_predictor.train()
    domain_classifier.train()

    for i, (data, gesture_label, subject_label) in enumerate(train_dataloader):
        data = data.cuda()                      # torch.Size([512, 4, 8, 14])
        # print(data.shape)
        gesture_label = gesture_label.cuda()    # torch.Size([512])     1024
        subject_label = subject_label.cuda()    # torch.Size([512])     1024,1

        # subject_label = subject_label.view(subject_label.shape[0])
        subject_label = subject_label.squeeze()

        # gesture_label = gesture_label.view(512, 1)
        # print(gesture_label.shape)
        # print(subject_label.shape)

        # p = float(i + epoch * len_dataloader) / epoch_num / len_dataloader
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # print(alpha)

        # pred_gesture -> torch.Size([512, 7])
        # pred_subject_label -> torch.Size([512, 1])
        # pred_gesture_labels, pred_subject_label = net(data)
        # pred_gesture_labels, pred_subject_label = net(data, alpha)
        # _, pred_gesture_labels = torch.max(pred_gesture.data, 1)    # pred_gesture_labels -> torch.Size([512])

        feature = feature_extractor(data)

        # train domain classifier
        pred_subject_label = domain_classifier(feature.detach())
        loss_D = domain_criterion(pred_subject_label, subject_label)
        running_D_loss += loss_D.item()
        loss_D.backward()
        optimizer_D.step()

        # train label predictor
        pred_gesture_label = label_predictor(feature)
        pred_subject_label = domain_classifier(feature)

        loss = class_criterion(pred_gesture_label, gesture_label)\
            - 0.1 * domain_criterion(pred_subject_label, subject_label)
        running_F_loss += loss.item()

        running_C_loss += class_criterion(pred_gesture_label, gesture_label)

        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        # err_gesture_label = class_criterion(pred_gesture_labels, gesture_label)     # [512,7] [512]
        # err_subject_label = domain_criterion(pred_subject_label, subject_label)     # float32[512,1] int64[512, 1]
        #
        # err = err_gesture_label - 0.5 * err_subject_label
        # running_loss += err.item()
        # running_D_loss += err_subject_label.item()

        # net.zero_grad()
        # optimizer.zero_grad()
        # err.backward()
        # optimizer.step()

        correct_gesture_label += torch.sum(torch.argmax(pred_gesture_label, dim=1) == gesture_label).item()
        total_num += data.shape[0]

    gesture_acc = correct_gesture_label / total_num
    # subject_acc = correct_subject_label / total_num
    D_loss = running_D_loss / (i + 1)
    F_loss = running_F_loss / (i + 1)
    C_loss = running_C_loss / (i + 1)

    print('Train:   D_Loss: {:.4f}   F_Loss: {:.4f}  C_Loss:  {:.4f}  Acc: {:.4f}'
          .format(D_loss, F_loss, C_loss, gesture_acc))


# validation
    running_loss, running_F_loss, running_D_loss, running_C_loss, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
    correct_gesture_label, correct_subject_label = 0.0, 0.0
    # net.eval()
    feature_extractor.eval()
    label_predictor.eval()
    domain_classifier.eval()

    for i, (data, gesture_label, subject_label) in enumerate(test_dataloader):
        data = data.cuda()                      # torch.Size([512, 4, 8, 14])
        # print(data.shape)
        gesture_label = gesture_label.cuda()    # torch.Size([512])
        subject_label = subject_label.cuda()    # torch.Size([512])

        # subject_label = subject_label.view(subject_label.shape[0], 1)
        subject_label = subject_label.squeeze()

        feature = feature_extractor(data)

        pred_subject_label = domain_classifier(feature)
        loss_D = domain_criterion(pred_subject_label, subject_label)
        running_D_loss += loss_D.item()

        pred_gesture_label = label_predictor(feature)
        loss = class_criterion(pred_gesture_label, gesture_label)\
            - 0.1 * domain_criterion(pred_subject_label, subject_label)
        running_F_loss += loss.item()

        running_C_loss += class_criterion(pred_gesture_label, gesture_label)

        correct_gesture_label += torch.sum(torch.argmax(pred_gesture_label, dim=1) == gesture_label).item()
        total_num += data.shape[0]

    gesture_acc = correct_gesture_label / total_num
    D_loss = running_D_loss / (i + 1)
    F_loss = running_F_loss / (i + 1)
    C_loss = running_C_loss / (i + 1)
    total_D_loss = running_D_loss / (i + 1)

    print('Test:   D_Loss: {:.4f}   F_Loss: {:.4f}  C_Loss:  {:.4f}  Acc: {:.4f}'
          .format(D_loss, F_loss, C_loss, gesture_acc))

    print('Time usage: {:.2f}s'.format(time.time()-epoch_start_time))

    # scheduler.step(F_loss)
    scheduler_F.step(F_loss)
    scheduler_D.step(D_loss)
    scheduler_C.step(C_loss)

    if F_loss + precision < best_loss:
        print('New best validation F_loss:  {:.4f}'.format(F_loss))
        best_loss = F_loss
        best_weights_F = copy.deepcopy(feature_extractor.state_dict())
        best_weights_C = copy.deepcopy(label_predictor.state_dict())
        best_weights_D = copy.deepcopy(domain_classifier.state_dict())
        patience = patience_increase + epoch
        print('So Far Patience: ', patience)

    print()

# save model
torch.save(best_weights_F, r'saved_model\feature_extractor.pkl')
torch.save(best_weights_C, r'saved_model\label_predictor.pkl')
torch.save(best_weights_D, r'saved_model\domain_classifier.pkl')
print('Model Saved.')

# torch.save(feature_extractor, r'saved_model\m_feature_extractor.pkl')
# torch.save(label_predictor, r'saved_model\m_label_predictor.pkl')
# torch.save(domain_classifier, r'saved_model\m_domain_classifier.pkl')


