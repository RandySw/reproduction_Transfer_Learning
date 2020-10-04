import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
from sklearn import manifold

import time
import os
import copy

from utliz import scramble, load_source_train_data, load_target_train_data, load_target_test_data

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
        return class_output, domain_output, feature

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


def training(source_dataloader, target_dataloader, net, optim, num_epoch, epoch=1):

    running_loss = 0.0
    source_hit_num, total_num = 0.0, 0.0

    net.train()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    len_dataloader = min(len(source_dataloader), len(target_dataloader))

    optimizer = optim

    src_feature_container = None
    tag_feature_container = None

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        if i > len_dataloader:
            break
        # print('{} batch'.format(i))

        source_data = source_data.cuda()  # [256, 1, 8, 52]
        source_label = source_label.cuda()

        target_data = target_data.cuda()

        s_domain_labels = torch.ones([source_label.shape[0], 1]).cuda()
        t_domain_labels = torch.zeros([target_data.shape[0], 1]).cuda()

        p = float(i + epoch * len_dataloader) / num_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # train with source data
        s_class_output, s_domain_output, feature_vector = net(source_data, alpha=alpha)

        if i == 0:
            src_feature_container = feature_vector.cpu().detach().numpy()
        else:
            src_feature_container = np.r_[src_feature_container, (feature_vector.cpu().detach().numpy())]
        # print('shape', src_feature_container.shape)

        err_s_label = class_criterion(s_class_output, source_label)
        err_s_domain = domain_criterion(s_domain_output, s_domain_labels)

        # train with target data
        _, t_domain_output, feature_vector = net(target_data, alpha=alpha)
        # tag_feature_container.append(feature_vector.cpu().detach().numpy())
        if i == 0:
            tag_feature_container = feature_vector.cpu().detach().numpy()
        else:
            tag_feature_container = np.r_[tag_feature_container, (feature_vector.cpu().detach().numpy())]

        err_t_domain = domain_criterion(t_domain_output, t_domain_labels)

        err = err_t_domain + err_s_domain + err_s_label
        running_loss += err

        net.zero_grad()
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        source_hit_num += torch.sum(torch.argmax(s_class_output, dim=1) == source_label).item()
        total_num += source_data.shape[0]

    s_acc = source_hit_num / total_num
    running_loss = running_loss / (i + 1)
    # src_feature_avr = src_feature_container.mean(axis=0)
    # tag_feature_avr = src_feature_container.mean(axis=0)

    return running_loss, s_acc, alpha, src_feature_container, tag_feature_container


def validation(dataloader, net, alpha=1, domain='source'):

    running_D_loss, running_C_loss = 0.0, 0.0
    correct_pred_num, total_num = 0.0, 0.0

    net.eval()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    feature_container = None

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()
        if domain == 'source':
            domain_labels = torch.ones([data.shape[0], 1]).cuda()
        elif domain == 'target':
            domain_labels = torch.zeros([data.shape[0], 1]).cuda()

        class_output, domain_output, feature_vector = net(data, alpha=alpha)

        if i == 0:
            feature_container = feature_vector.cpu().detach().numpy()
        else:
            feature_container = np.r_[feature_container, (feature_vector.cpu().detach().numpy())]

        # calculate loss
        err_domain = domain_criterion(domain_output, domain_labels)
        err_class = class_criterion(class_output, labels)
        running_D_loss += err_domain.item()
        running_C_loss += err_class.item()

        # calculate accuracy
        correct_pred_num += torch.sum(torch.argmax(class_output, dim=1) == labels).item()
        total_num += data.shape[0]

    dataloader_D_loss = running_D_loss / (i + 1)
    dataloader_C_loss = running_C_loss / (i + 1)
    dataloader_acc = correct_pred_num / total_num
    # feature_avr = feature_container.mean(axis=0)

    return dataloader_D_loss, dataloader_C_loss, dataloader_acc, feature_container


torch.cuda.set_device(0)
seed = 0
torch.manual_seed(seed)

os.environ['KMP_DUPLICATE_LIB_OK']= 'True'


# %% Load raw dataset
source_set_training = np.load("../../PyTorchImplementation/formatted_datasets"
                              "/saved_pre_training_dataset_spectrogram.npy", encoding="bytes", allow_pickle=True)
source_data_training, source_labels_training = source_set_training


target_set_training = np.load('../../PyTorchImplementation/formatted_datasets/saved_evaluation_dataset_training.npy',
                           encoding="bytes", allow_pickle=True)
target_data_training, target_labels_training = target_set_training


list_source_train_dataloader, list_source_valid_dataloader \
    = load_source_train_data(source_data_training, source_labels_training)

list_target_train_dataloader, list_target_valid_dataloader \
    = load_target_train_data(target_data_training, target_labels_training)


src_data_train = []
for i in range(len(list_source_train_dataloader)):
    src_data_train.extend(list_source_train_dataloader[i])

src_data_valid = []
for i in range(len(list_source_valid_dataloader)):
    src_data_valid.extend(list_source_valid_dataloader[i])

tag_data_train = []
for i in range(len(list_target_train_dataloader)):
    # i = 0
    tag_data_train.extend(list_target_train_dataloader[i])

tag_data_valid = []
for i in range(len(list_target_valid_dataloader)):
    # i = 0
    tag_data_valid.extend(list_target_valid_dataloader[i])


Da_net = DaNNet(number_of_classes=7).cuda()

for p in Da_net.parameters():
    p.requires_grad = True

precision = 1e-8

optimizer = optim.Adam(Da_net.parameters(), lr=2e-4)
# optimizer = optim.SGD(Da_net.parameters(), momentum=0.5, lr=2e-4)
# optimizer = optim.SGD(Da_net.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5,
                                                 verbose=True, eps=precision)
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

# training
epoch_num = 120
patience = 20
patience_increase = 20
best_acc = 0
best_loss = float('inf')

record_index = [0, 15, 30, 45, 60, 75, 90]
feature_vectors_s_train = []
feature_vectors_t_train = []
feature_vectors_s_val = []
feature_vectors_t_val = []


for epoch in range(epoch_num):
    epoch_start_time = time.time()

    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

    len_dataloader = min(len(src_data_train), len(tag_data_train))
    # train model with source data

    train_loss, train_acc, alpha, src_feature_train_container, tag_feature_train_container \
        = training(src_data_train, tag_data_train, Da_net, optimizer, epoch_num, epoch=epoch)

    src_valid_D_loss, src_valid_C_loss, src_valid_acc, src_feature_val_container \
        = validation(src_data_valid, net=Da_net, alpha=alpha)

    tag_valid_D_loss, tag_valid_C_loss, tag_valid_acc, tag_feature_val_container \
        = validation(tag_data_valid, net=Da_net, alpha=alpha)

    print('Training: Loss:{:.4f}  src_Acc:{:.4f}'.format(train_loss, train_acc))
    print('Valid Source:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(src_valid_D_loss, src_valid_C_loss, src_valid_acc))
    print('Valid Target:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(tag_valid_D_loss, tag_valid_C_loss, tag_valid_acc))

    print('epoch time usage: {:.2f}s'.format(time.time() - epoch_start_time))

    val_loss = tag_valid_D_loss + src_valid_C_loss + src_valid_D_loss

    scheduler.step(val_loss)

    # if tag_valid_acc + precision > best_acc:
    #     print('New Best Target Validation Accuracy: {:.4f}'.format(tag_valid_acc))
    #     best_acc = tag_valid_acc
    #     best_weights = copy.deepcopy(Da_net.state_dict())
    #     patience = patience_increase + epoch
    #     print('So Far Patience: ', patience)
    # print()

    if val_loss + precision < best_loss:
        print('New Best Validation Loss: {:.4f}'.format(val_loss))
        best_loss = val_loss
        best_weights = copy.deepcopy(Da_net.state_dict())
        patience = patience_increase + epoch
        print('So Far Patience: ', patience)
    print()

    if epoch in record_index:
        feature_vectors_s_train.append(src_feature_train_container)
        feature_vectors_t_train.append(tag_feature_train_container)
        feature_vectors_s_val.append(src_feature_val_container)
        feature_vectors_t_val.append(tag_feature_val_container)

    if epoch > patience:
        break

print('-' * 20 + '\n' + '-' * 20)
print('Best Best Target Validation Accuracy: {:.4f}'.format(best_acc))

# feature visualization
# for index in range(len(feature_vectors_s_train)):

# index = 1
# X = feature_vectors_s_train[index][:300]    # 300: 样本数目
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# X_tsne = tsne.fit_transform(X)
#
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)
# plt.scatter(X_tsne[:10, 0], X_tsne[:10, 1])
# plt.show()


iter_time = len(feature_vectors_s_train)

print('Start t-SNE analysis...')
for i in range(iter_time):
    X = feature_vectors_s_train[i][:6000]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    plt.subplot(1, iter_time, i)
    plt.scatter(X_tsne[:6000, 0], X_tsne[:6000, 1])
    plt.title('src_train: {} epoch'.format(record_index[i]))
    plt.xlabel('t-SNE feature1')
    plt.xlabel('t-SNE feature2')
plt.show()




