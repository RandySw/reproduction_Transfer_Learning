import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
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
            nn.Dropout2d(0.3),
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
        return flatten_tensor


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(24 * 1 * 8, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.3),

            nn.Linear(512, 7)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, input):
        output = self.layer(input)
        return output


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(24 * 1 * 8, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(256),
            nn.Dropout(0.3),

            # nn.Linear(256, 1)
            nn.Linear(256, 11)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, input):
        out = self.layer(input)
        return out


class ANN(nn.Module):

    def __init__(self, input_channel, number_of_classes):
        super(ANN, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(input_channel, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),

            nn.Linear(512, number_of_classes)
        )

    def forward(self, x):
        # x = x.view(-1, input_channel)
        class_output = self.hidden_layer(x)
        return class_output


class NaiveCNN(nn.Module):
    def __init__(self):
        super(NaiveCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 12, kernel_size=(4, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.3),

            nn.Conv2d(12, 12, kernel_size=(3, 3)),
            nn.BatchNorm2d(12),
            nn.PReLU(12),
            nn.Dropout2d(0.3),

            nn.Conv2d(12, 24, kernel_size=(3, 3)),
            nn.BatchNorm2d(24),
            nn.PReLU(24),
            nn.Dropout2d(0.3),
        )

        self.dense_layer = nn.Sequential(
            nn.Linear(24 * 1 * 8, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.3),

            nn.Linear(512, 7)
        )

    def forward(self, x):
        output = self.feature_extractor(x)      # [1024, 24, 1, 8]
        # print(output.shape)
        output = self.dense_layer(output.view(-1, 24 * 1 * 8))
        return output


class SlowFusionModel(nn.Module):
    def __init__(self):
        super(SlowFusionModel, self).__init__()

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
            nn.Dropout2d(0.3),
        )

        self.predictor = nn.Sequential(
            nn.Linear(24 * 1 * 8, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.3),

            nn.Linear(512, 7)
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

        pred_label = self.predictor(flatten_tensor)

        return pred_label





