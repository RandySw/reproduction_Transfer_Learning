import torch.nn.functional as F
from torch import nn
import torch


class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self, dropout_rate=0):
        super(LeNetEncoder, self).__init__()
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5))  # [32, 6, 48]
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 3))  # [32, 6, 16]
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(dropout_rate)

        self._conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5))  # [64, 4, 12]
        self._pool2 = nn.MaxPool2d(kernel_size=(1, 3))  # [64, 4, 4]   64*4*4 = 1024
        self._batch_norm2 = nn.BatchNorm2d(64)
        self._prelu2 = nn.PReLU(64)
        self._dropout2 = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        pool1 = self._pool1(conv1)
        conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1))))
        pool2 = self._pool2(conv2)
        x = pool2.view(-1, 1024)
        return x
