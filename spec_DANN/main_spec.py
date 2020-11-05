import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

from functions import scramble, load_source_train_data, load_target_train_data, load_target_test_data
from functions import train, valid, test, training, validation, testing
from model_spec import DANNSpect, CNNSpect


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

torch.cuda.set_device(0)
seed = 0
torch.manual_seed(seed)


src_set_training = np.load('formatted_datasets/saved_pre_training_dataset_spectrogram.npy',
                              encoding='bytes', allow_pickle=True)
src_data_training, src_labels_training = src_set_training

tgt_set_training = np.load('formatted_datasets/saved_evaluation_dataset_training.npy',
                              encoding='bytes', allow_pickle=True)
tgt_data_training, tgt_labels_training = tgt_set_training

tgt_set_test0 = np.load('formatted_datasets/saved_evaluation_dataset_test0.npy',
                        encoding='bytes', allow_pickle=True)
tgt_data_test0, tgt_labels_test0 = tgt_set_test0

tgt_set_test1 = np.load('formatted_datasets/saved_evaluation_dataset_test1.npy',
                        encoding='bytes', allow_pickle=True)
tgt_data_test1, tgt_labels_test1 = tgt_set_test1


list_src_train_dataloader, list_src_valid_dataloader = load_source_train_data(src_data_training, src_labels_training)

list_tgt_train_dataloader, list_tgt_valid_dataloader = load_target_train_data(tgt_data_training, tgt_labels_training)

list_tgt_test0_dataloader = load_target_test_data(tgt_data_test0, tgt_labels_test0)

list_tgt_test1_dataloader = load_target_test_data(tgt_data_test1, tgt_labels_test1)

src_data_train = []
for i in range(len(list_src_train_dataloader)):
    src_data_train.extend(list_src_train_dataloader[i])

src_data_valid = []
for i in range(len(list_src_valid_dataloader)):
    src_data_valid.extend(list_src_valid_dataloader[i])

tgt_data_train = []
for i in range(len(list_tgt_train_dataloader)):
    i = 0
    tgt_data_train.extend(list_tgt_train_dataloader[i])

tgt_data_valid = []
for i in range(len(list_tgt_valid_dataloader)):
    i = 0
    tgt_data_valid.extend(list_tgt_valid_dataloader[i])

tgt_data_test0 = []
# for i in range(len(list_tgt_test0_dataloader)):
#     tgt_data_test0.extend(list_tgt_test0_dataloader[i])
tgt_data_test0.extend(list_tgt_test0_dataloader[0])

tgt_data_test1 = []
# for i in range(len(list_tgt_test1_dataloader)):
#     tgt_data_test1.extend(list_tgt_test1_dataloader[i])
tgt_data_test1.extend(list_tgt_test1_dataloader[0])

net = DANNSpect(number_of_class=7).cuda()
for p in net.parameters():
    p.requires_grad = True

CUDA = True

precision = 1e-8
optimizer = optim.Adam(net.parameters(), lr=2e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5,
                                                 verbose=True, eps=precision)

class_criterion = nn.NLLLoss()
# class_criterion = nn.CrossEntropyLoss()

epoch_num = 200     # 120
patience = 16   # 12
patience_increase = 16  # 12
best_acc = 0
best_loss = float('inf')

# Non-TL curve
train_loss_curve = []
train_acc_curve = []
valid_loss_curve = []
valid_acc_curve = []
test0_loss_curve = []
test0_acc_curve = []
test1_loss_curve = []
test1_acc_curve = []

# TL curve
train_loss_cur = []
train_acc_cur = []
src_val_D_loss_cur = []
src_val_C_loss_cur = []
src_val_acc_cur = []
tag_val_D_loss_cur = []
tag_val_C_loss_cur = []
tag_val_acc_cur = []
tag_test0_D_loss_cur = []
tag_test0_C_loss_cur = []
tag_test0_acc_cur = []
tag_test1_D_loss_cur = []
tag_test1_C_loss_cur = []
tag_test1_acc_cur = []

for epoch in range(epoch_num):
    epoch_start_time = time.time()

    print('epoch: {} / {}'.format(epoch + 1, epoch_num))
    print('-' * 20)

    # TL process
    train_loss, train_acc, alpha, net = training(src_data_train, tgt_data_train, net, optimizer, epoch_num, epoch=epoch)

    src_valid_D_loss, src_valid_C_loss, src_valid_acc = validation(src_data_valid, net=net, alpha=alpha)

    tag_valid_D_loss, tag_valid_C_loss, tag_valid_acc = validation(tgt_data_valid, net=net, alpha=alpha)

    tag_test0_D_loss, tag_test0_C_loss, tag_test0_acc = testing(tgt_data_test0, net=net, alpha=alpha)

    tag_test1_D_loss, tag_test1_C_loss, tag_test1_acc = testing(tgt_data_test1, net=net, alpha=alpha)

    print('Training: Loss:{:.4f}  src_Acc:{:.4f}'.format(train_loss, train_acc))
    print('Valid Source:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(src_valid_D_loss, src_valid_C_loss, src_valid_acc))
    print('Valid Target:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(tag_valid_D_loss, tag_valid_C_loss, tag_valid_acc))
    print('Test0:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(tag_test0_D_loss, tag_test0_C_loss, tag_test0_acc))
    print('Test1:  D_loss:{:.4f}  C_loss:{:.4f}  Acc:{:.4f}'
          .format(tag_test1_D_loss, tag_test1_C_loss, tag_test1_acc))
    print('epoch time usage: {:.2f}s'.format(time.time() - epoch_start_time))

    val_loss = tag_valid_D_loss + src_valid_C_loss + src_valid_D_loss

    scheduler.step(val_loss)

    if val_loss + precision < best_loss:
        print('New Best Validation Loss: {:.4f}'.format(val_loss))
        best_loss = val_loss
        best_weights = copy.deepcopy(net.state_dict())
        patience = patience_increase + epoch
        print('So Far Patience: ', patience)
    print()

    train_loss_cur.append(train_loss)
    train_acc_cur.append(train_acc)
    src_val_D_loss_cur.append(src_valid_D_loss)
    src_val_C_loss_cur.append(src_valid_C_loss)
    src_val_acc_cur.append(src_valid_acc)
    tag_val_D_loss_cur.append(tag_valid_D_loss)
    tag_val_C_loss_cur.append(tag_valid_C_loss)
    tag_val_acc_cur.append(tag_valid_acc)
    tag_test0_D_loss_cur.append(tag_test0_D_loss)
    tag_test0_C_loss_cur.append(tag_test0_C_loss)
    tag_test0_acc_cur.append(tag_test0_acc)
    tag_test1_D_loss_cur.append(tag_test1_D_loss)
    tag_test1_C_loss_cur.append(tag_test1_C_loss)
    tag_test1_acc_cur.append(tag_test1_acc)

    if epoch > patience:
        break

    # Non-TL process
    # train_acc, train_loss = train(src_data_train, net, optimizer, class_criterion)
    # valid_acc, valid_loss = valid(src_data_valid, net, class_criterion)
    # test0_acc, test0_loss = test(tgt_data_test0, net, class_criterion)
    # test1_acc, test1_loss = test(tgt_data_test0, net, class_criterion)

    # print('     Train Loss: {:.4f}  Train Accuracy: {:.4f}'.format(train_loss, train_acc))
    # print('     Valid Loss: {:.4f}  Valid Accuracy: {:.4f}'.format(valid_loss, valid_acc))
    # print('     Test0 Loss: {:.4f}  Test0 Accuracy: {:.4f}'.format(test0_loss, test1_acc))
    # print('     Test1 Loss: {:.4f}  Test1 Accuracy: {:.4f}'.format(test1_loss, test1_acc))
    # print('     Epoch Time Usage:  {:.2f}s'.format(time.time() - epoch_start_time))

    # train_acc_curve.append(train_acc)
    # train_loss_curve.append(train_loss)
    # valid_acc_curve.append(valid_acc)
    # valid_loss_curve.append(valid_loss)
    # test0_acc_curve.append(test0_acc)
    # test0_loss_curve.append(test0_loss)
    # test1_acc_curve.append(test1_acc)
    # test1_loss_curve.append(test1_loss)

    print('-' * 20)

print('-' * 20 + '\n' + '-' * 20)
# print('Best Best Target Validation Accuracy: {:.4f}'.format(best_acc))
print('Best Best Validation Loss: {:.4f}'.format(best_loss))

# TL curve plot
plt.plot(train_loss_cur)
plt.title('Training Loss - Epoch')
plt.legend(['train_loss'])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()

plt.plot(src_val_D_loss_cur)
plt.plot(tag_val_D_loss_cur)
plt.plot(tag_test0_D_loss_cur)
plt.plot(tag_test1_D_loss_cur)
plt.title('Domain Loss - Epoch')
plt.legend(['src_val_D_loss', 'tag_val_D_loss', 'tag_test0_D_loss', 'tag_test1_D_loss'])
plt.xlabel('Epoch')
plt.ylabel('Domain Loss')
plt.show()

plt.plot(src_val_C_loss_cur)
plt.plot(tag_val_C_loss_cur)
plt.plot(tag_test0_C_loss_cur)
plt.plot(tag_test1_C_loss_cur)
plt.title('Classification Loss - Epoch')
plt.legend(['src_val_C_loss', 'tag_val_C_loss', 'tag_test0_C_loss', 'tag_test1_C_loss'])
plt.xlabel('Epoch')
plt.ylabel('Classification Loss')
plt.show()

plt.plot(train_acc_cur)
plt.plot(src_val_acc_cur)
plt.plot(tag_val_acc_cur)
plt.plot(tag_test0_acc_cur)
plt.plot(tag_test1_acc_cur)
plt.title('Classification Accuracy - Epoch')
plt.legend(['src_train_acc', 'src_val_acc', 'tag_val_acc', 'tag_test0_acc', 'tag_test1_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Non-TL curve plot
# plt.plot(train_loss_curve)
# plt.plot(valid_loss_curve)
# plt.plot(test0_loss_curve)
# plt.plot(test1_loss_curve)
# plt.title('Loss - Epoch')
# plt.legend(['Train Loss', 'Valid Loss', 'Test0 Loss', 'Test1 Loss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
#
# plt.plot(train_acc_curve)
# plt.plot(valid_acc_curve)
# plt.plot(test0_acc_curve)
# plt.plot(test1_acc_curve)
# plt.title('Accuracy - Epoch')
# plt.legend(['Train Acc', 'Valid Acc', 'Test0 Acc', 'Test1 Acc'])
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()


# ToDo:
#       1. 检查 load_train_data() 看看是不是这个函数出问题了导致 valid acc 高于 train acc
#       2. 将原始非TL的代码更改成TL的
#       4. log_softmax UserWarning solved
