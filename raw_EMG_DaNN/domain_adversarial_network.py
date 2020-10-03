import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function

import torchvision.transforms as transforms

import time


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


# %% DaNN
class DaNNet(nn.Module):
    def __init__(self, number_of_classes,):
        super(DaNNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5)),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Dropout2d(0.5),

            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(32, 64, kernel_size=(3, 5)),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Dropout2d(0.5),

            nn.MaxPool2d(kernel_size=(1, 3)),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),

            nn.Linear(512, 1),
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.5),

            nn.Linear(512, number_of_classes)
        )

        self.initialize_weights()

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(-1, 1024)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


class LeNet5(nn.Module):
    def __init__(self, num_of_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(32, 64, kernel_size=(3, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

        self.label_predictor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(1024, num_of_classes),
        )

        self.initialize_weights()

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(-1, 1024)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
             nn.Conv2d(1, 32, kernel_size=(3, 5)),
             nn.BatchNorm2d(32),
             nn.PReLU(32),
             nn.Dropout2d(0.5),

             nn.MaxPool2d(kernel_size=(1, 3)),

             nn.Conv2d(32, 64, kernel_size=(3, 5)),
             nn.BatchNorm2d(64),
             nn.PReLU(64),
             nn.Dropout2d(0.5),

             nn.MaxPool2d(kernel_size=(1, 3)),
         )

        self.initialize_weights()

    def forward(self, x):
        feature_vector = self.conv(x).squeeze()
        return feature_vector

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 1),
        )

        self.initialize_weights()

    def forward(self, feature_vector):
        # print('shape of feature_vector: ', feature_vector.shape)
        # time.sleep(1000)
        # shape of feature vector: [512, 64, 4, 4]
        feature_vector = feature_vector.view(-1, 1024)
        domain_label = self.layer(feature_vector)
        return domain_label

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


class LabelPredictor(nn.Module):

    def __init__(self, number_of_classes,):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Dropout(0.5),

            nn.Linear(512, number_of_classes)
        )

        self.initialize_weights()

    def forward(self, feature_vector):
        feature_vector = feature_vector.view(-1, 1024)
        class_label = self.layer(feature_vector)
        return class_label

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


# %% Pre-processing


if __name__ == '__main__':
    # init network components
    # feature_extractor = FeatureExtractor(number_of_classes=10).cuda()
    # label_predictor = LabelPredictor(number_of_classes=10).cuda()
    # domain_classifier = DomainClassifier().cuda()
    #
    # print(feature_extractor)
    # print(label_predictor)
    # print(domain_classifier)

    DaNN_net = DaNNet(number_of_classes=10).cuda()
    print(DaNN_net)





