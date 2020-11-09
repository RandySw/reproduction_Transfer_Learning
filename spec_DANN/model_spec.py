import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
import numpy as np
import os

from functions import load_source_train_data


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


# 带TL的 DANN 模型
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

        # self.FC1 = nn.Sequential(
        #     nn.Linear(24 * 1 * 8, 100),
        #     nn.BatchNorm1d(100),
        #     nn.PReLU(100),
        #     nn.Dropout(0.5),
        # )
        #
        # self.FC2 = nn.Sequential(
        #     nn.Linear(100, 100),
        #     nn.BatchNorm1d(100),
        #     nn.PReLU(100),
        #     nn.Dropout(0.5),
        # )
        #
        # self.FC3 = nn.Sequential(
        #     nn.Linear(100, number_of_class),
        # )

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

    def forward(self, x, alpha):
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

        reverse_feature = ReverseLayerF.apply(flatten_tensor, alpha)
        label_output = self.label_classifier(flatten_tensor)
        domain_output = self.domain_classifier(reverse_feature)

        return nn.functional.log_softmax(label_output, dim=1), domain_output

        # fc_1_output = self.FC1(flatten_tensor)      #
        #
        # fc_2_output = self.FC2(fc_1_output)         #
        #
        # fc_3_output = self.FC3(fc_2_output)         #

        # return nn.functional.log_softmax(fc_3_output)


# 非TL的并行卷积模型
class CNNSpect(nn.Module):
    def __init__(self, number_of_class,):
        super(CNNSpect, self).__init__()
        self._input_batchnorm = nn.BatchNorm2d(4, eps=1e-4)
        self._input_prelu = nn.PReLU(4)

        self.Conv1_1 = nn.Sequential(
            nn.Conv2d(2, 12, kernel_size=(4, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.5),
        )

        self.Conv1_2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=(3, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.5),
        )

        self.Conv2_1 = nn.Sequential(
            nn.Conv2d(2, 12, kernel_size=(4, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.5),
        )

        self.Conv2_2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=(3, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.5),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=(3, 3)),
            nn.BatchNorm2d(24),
            nn.PReLU(24),
            nn.Dropout2d(0.5),
        )

        self.FC1 = nn.Sequential(
            nn.Linear(24 * 1 * 8, 100),
            nn.BatchNorm1d(100),
            nn.PReLU(100),
            nn.Dropout2d(0.5),
        )

        self.FC2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.PReLU(100),
            nn.Dropout2d(0.5),
        )

        self.FC3 = nn.Sequential(
            nn.Linear(100, number_of_class),
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

    def forward(self, x):
        x = self._input_prelu(self._input_batchnorm(x))  # (512, 4, 8, 14)

        input_1 = x[:, 0:2, :, :]
        # print(input_1.shape)
        input_2 = x[:, 2:4, :, :]
        # print(input_2.shape)

        branch_1 = self.first_parallel(input_1)  # (512, 12, 3, 10)
        branch_2 = self.second_parallel(input_2)  # (512, 12, 3, 10)

        merged_branch = branch_1 + branch_2  # (512, 12, 3, 10)

        after_conv = self.Conv3(merged_branch)  # (512, 24, 1, 8)

        flatten_tensor = after_conv.view(-1, 24 * 1 * 8)  #

        fc_1_output = self.FC1(flatten_tensor)  #

        fc_2_output = self.FC2(fc_1_output)  #

        fc_3_output = self.FC3(fc_2_output)  #

        return nn.functional.log_softmax(fc_3_output)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    torch.cuda.set_device(0)
    seed = 0
    torch.manual_seed(seed)

    # load data
    src_set_training = np.load('formatted_datasets/saved_pre_training_dataset_spectrogram.npy',
                               encoding='bytes', allow_pickle=True)
    src_data_training, src_labels_training = src_set_training

    list_src_train_dataloader, list_src_valid_dataloader = load_source_train_data(src_data_training,
                                                                                  src_labels_training)

    src_data_train = []
    for i in range(len(list_src_train_dataloader)):
        src_data_train.extend(list_src_train_dataloader[i])

    net = CNNSpect(number_of_class=7).cuda()
    print(net)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(DANNSpect.parameters(), lr=2e-4)

    # training stage
    net.train()
    for i, (data, labels) in enumerate(src_data_train):
        data = data.cuda()      # (512, 4, 8, 14)
        labels = labels.cuda()

        pred_labels = net(data)
    print('Finished.')
        # ToDo: BUG -> 在first_parallel处 input为(512, 2, 8, 14), 而 Conv1_1要求的输入是三维的，输出数据的尺寸不符













