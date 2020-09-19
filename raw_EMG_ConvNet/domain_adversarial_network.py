import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function

import torchvision.transforms as transforms


# %% DaNN
class FeatureExtractor(nn.Module):

    def __init__(self, number_of_classes):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
             nn.Conv2d(1, 32, kernel_size=(3, 5)),
             nn.BatchNorm2d(32),
             nn.PReLU(32),
             nn.Dropout(0.5),

             nn.MaxPool2d(kernel_size=(1, 3)),

             nn.Conv2d(32, 64, kernel_size=(3, 5)),
             nn.BatchNorm2d(64),
             nn.PReLU(64),
             nn.Dropout(0.5),

             nn.MaxPool2d(kernel_size=(1, 3)),
         )

        self.initialize_weights()

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bais.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bais.data.zero_()


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
        domain_label = self.layer(feature_vector)
        return domain_label

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bais.data.zero_()


class LabelPredictor(nn.Module):

    def __init__(self, number_of_classes):
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
        output = self.layer(feature_vector)
        flatten_tensor = output.view(-1, 1024)
        label = self.fc(flatten_tensor)
        return label

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bais.data.zero_()


# %% Pre-processing









