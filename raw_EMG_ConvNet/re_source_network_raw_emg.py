import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, number_of_class,):
        super(Net, self).__init__()
        # torch.nn.Conv2d(in_channels,out_Channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.cnn = nn.Sequential(
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

        self.fc = nn.Sequential(
            nn.Linear(1024, 500),
            nn.BatchNorm1d(500),
            nn.PReLU(500),
            nn.Dropout(0.5),
            nn.Linear(500, number_of_class)
        )

        self.initialize_weights()
        print(self)
        print('Number Parameters: ', self.get_n_parameters())

    def forward(self, x):
        output = self.cnn(x)
        flatten_tensor = output.view(-1, 1024)
        output = self.fc(flatten_tensor)
        return output

    def get_n_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal_(m.weight)     # 网络权重初始化 -> kaiming_normal 分布
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
                # print('conv')
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
                # print('linear')


if __name__ == '__main__':
    x_train = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([[1], [0]])

    model = Net(2).cuda()

    model.train()






