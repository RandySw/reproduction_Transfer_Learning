import torch.nn as nn
import torch.nn.functional as F


class LeNetClassifier(nn.Module):
    def __init__(self, dropout_rate=.5, number_of_class=7):
        super(LeNetClassifier, self).__init__()
        self._fc1 = nn.Linear(1024, 500)
        self._batch_norm1 = nn.BatchNorm1d(500)
        self._prelu1 = nn.PReLU(500)
        self._dropout1 = nn.Dropout(dropout_rate)

        self._fc2 = nn.Linear(500, 100)
        self._batch_norm2 = nn.BatchNorm1d(100)
        self._prelu2 = nn.PReLU(100)
        self._dropout2 = nn.Dropout(dropout_rate)

        self._output = nn.Linear(100, number_of_class)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        fc1 = self._dropout1(self._prelu1(self._batch_norm1(self._fc1(x))))
        fc2 = self._dropout2(self._prelu2(self._batch_norm2(self._fc2(fc1))))
        output = self._output(fc2)
        return output
