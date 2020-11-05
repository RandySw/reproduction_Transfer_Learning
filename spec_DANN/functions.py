import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def training(source_dataloader, target_dataloader, net, optim, num_epoch, epoch):

    running_loss = 0.0
    source_hit_num, total_num = 0.0, 0.0

    net.train()

    class_criterion = nn.NLLLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    len_dataloader = min(len(source_dataloader), len(target_dataloader))

    optimizer = optim

    for i, ((source_data, source_label), (target_data, target_label)) in enumerate(zip(source_dataloader, target_dataloader)):
        if i > len_dataloader:
            break
        # print('{} batch'.format(i))

        source_data = source_data.cuda()  # [256, 1, 8, 52]
        source_label = source_label.cuda()

        target_data = target_data.cuda()
        target_label = target_label.cuda()

        s_domain_labels = torch.ones([source_label.shape[0], 1]).cuda()
        t_domain_labels = torch.zeros([target_data.shape[0], 1]).cuda()

        p = float(i + epoch * len_dataloader) / num_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # train with source data
        s_class_output, s_domain_output = net(source_data, alpha=alpha)
        _, s_pred_labels = torch.max(s_class_output.data, 1)

        err_s_label = class_criterion(s_class_output, source_label)     # [4096,7]  [4096]
        err_s_domain = domain_criterion(s_domain_output, s_domain_labels)   # [4096,1] [4096, 1]

        # train with target data
        t_class_output, t_domain_output = net(target_data, alpha=alpha)
        _, t_pred_labels = torch.max(t_class_output.data, 1)

        err_t_domain = domain_criterion(t_domain_output, t_domain_labels)   # float32[4096,1] float32[4096,1]
        err_t_label = class_criterion(t_class_output, target_label)

        # 这里target label的损失也算进去了
        # err = err_t_domain + err_t_label + err_s_domain + err_s_label     # 带 target label 的 loss
        err = err_t_domain + err_s_domain + err_s_label                     # 不带 target label 的 loss
        running_loss += err

        net.zero_grad()
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        # source_hit_num += torch.sum(torch.argmax(s_class_output, dim=1) == source_label).item()
        source_hit_num += torch.sum(s_pred_labels == source_label).item()
        total_num += source_data.shape[0]

    s_acc = source_hit_num / total_num
    running_loss = running_loss / (i + 1)

    return running_loss, s_acc, alpha, net


def validation(dataloader, net, alpha=1, domain='source'):

    running_D_loss, running_C_loss = 0.0, 0.0
    correct_pred_num, total_num = 0.0, 0.0

    net.eval()

    class_criterion = nn.NLLLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()
        if domain == 'source':
            domain_labels = torch.ones([data.shape[0], 1]).cuda()
        elif domain == 'target':
            domain_labels = torch.zeros([data.shape[0], 1]).cuda()

        class_output, domain_output = net(data, alpha=alpha)

        _, pred_labels = torch.max(class_output.data, 1)

        # calculate loss
        err_domain = domain_criterion(domain_output, domain_labels)
        err_class = class_criterion(class_output, labels)
        running_D_loss += err_domain.item()
        running_C_loss += err_class.item()

        # calculate accuracy
        # correct_pred_num += torch.sum(torch.argmax(class_output, dim=1) == labels).item()
        correct_pred_num += torch.sum(pred_labels == labels).item()
        total_num += data.shape[0]

    dataloader_D_loss = running_D_loss / (i + 1)
    dataloader_C_loss = running_C_loss / (i + 1)
    dataloader_acc = correct_pred_num / total_num

    return dataloader_D_loss, dataloader_C_loss, dataloader_acc


def testing(dataloader, net, alpha=1):
    net.eval()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    running_D_loss, running_C_loss = 0.0, 0.0
    correct_pred_num, total_num = 0.0, 0.0

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()
        domain_labels = torch.zeros([data.shape[0], 1]).cuda()

        class_output, domain_output = net(data, alpha=alpha)
        _, pred_labels = torch.max(class_output.data, 1)

        # calculate loss
        err_domain = domain_criterion(domain_output, domain_labels)
        err_class = class_criterion(class_output, labels)
        running_D_loss += err_domain.item()
        running_C_loss += err_class.item()

        # calculate accuracy
        # correct_pred_num += torch.sum(torch.argmax(class_output, dim=1) == labels).item()
        correct_pred_num += torch.sum(pred_labels == labels).item()
        total_num += data.shape[0]

    dataloader_D_loss = running_D_loss / (i + 1)
    dataloader_C_loss = running_C_loss / (i + 1)
    dataloader_acc = correct_pred_num / total_num

    return dataloader_D_loss, dataloader_C_loss, dataloader_acc


def scramble(examples, labels):
    """
    :param examples:
    :param labels:
    :return:
            返回顺序打乱的数据集
    """
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])
    return new_examples, new_labels


def load_source_train_data(source_data_training, source_labels_training):

    list_source_train_dataloader = []
    list_source_valid_dataloader = []

    print('Start loading Source training data...')
    # for j in range(19):
    for j in range(7):
        source_examples_personne_training = []
        source_labels_personne_training = []

        source_examples_personne_valid = []
        source_labels_personne_valid = []

        for k in range(len(source_data_training[j])):
            # 前 3个cycle 的数据用作train
            if k < 21:
                source_examples_personne_training.extend(source_data_training[j][k])
                source_labels_personne_training.extend(source_labels_training[j][k])
            # 最后 1个cycle 的数据用作valid
            else:
                source_examples_personne_valid.extend(source_data_training[j][k])
                source_labels_personne_valid.extend(source_labels_training[j][k])

        # data scrambled
        source_examples_personne_training_scrambled, source_labels_personne_training_scrambled \
            = scramble(source_examples_personne_training, source_labels_personne_training)
        source_examples_personne_valid_scrambled, source_labels_personne_valid_scrambled \
            = scramble(source_examples_personne_valid, source_labels_personne_valid)

        # numpy2tensor
        source_train = TensorDataset(
            torch.from_numpy(np.array(source_examples_personne_training_scrambled, dtype=np.float32)),
            torch.from_numpy(np.array(source_labels_personne_training_scrambled, dtype=np.int64)))
        source_valid = TensorDataset(
            torch.from_numpy(np.array(source_examples_personne_valid_scrambled, dtype=np.float32)),
            torch.from_numpy(np.array(source_labels_personne_valid_scrambled, dtype=np.int64)))

        # tensor2dataloader
        source_train_Loader = DataLoader(source_train, batch_size=3200, shuffle=True, drop_last=True)
        source_valid_Loader = DataLoader(source_valid, batch_size=256, shuffle=True, drop_last=True)

        # 装载 19名 source subject 的 dataloader
        list_source_train_dataloader.append(source_train_Loader)
        list_source_valid_dataloader.append(source_valid_Loader)

        print('No.{} subject'.format(j + 1))
        print('source_examples_personne_training:   ', np.shape(source_examples_personne_training))
        print('source_examples_personne_valid:      ', np.shape(source_examples_personne_valid))
        print('size of source train dataloader:', len(list_source_train_dataloader))
        print('size of source valid dataloader:', len(list_source_valid_dataloader))
        print('-' * 30)

    print('Loading Source training/valid data finished.\n\n')

    return list_source_train_dataloader, list_source_valid_dataloader


def load_target_train_data(target_data_training, target_labels_training):

    list_target_train_dataloader = []
    list_target_valid_dataloader = []

    print('Start loading Target training data...')
    # for j in range(17):
    for j in range(1):
        target_examples_personne_training = []
        target_labels_personne_training = []

        target_examples_personne_valid = []
        target_labels_personne_valid = []

        for k in range(len(target_data_training[j])):
            if k < 21:
                target_examples_personne_training.extend(target_data_training[j][k])
                target_labels_personne_training.extend(target_labels_training[j][k])
            else:
                target_examples_personne_valid.extend(target_data_training[j][k])
                target_labels_personne_valid.extend(target_labels_training[j][k])

        # dataset scrambled
        target_examples_personne_training_scrambled, target_labels_personne_training_scrambled \
            = scramble(target_examples_personne_training, target_labels_personne_training)
        target_examples_personne_valid_scrambled, target_labels_personne_valid_scrambled \
            = scramble(target_examples_personne_valid, target_labels_personne_valid)

        # numpy2tensor
        target_train = TensorDataset(
            torch.from_numpy(np.array(target_examples_personne_training_scrambled, dtype=np.float32)),
            torch.from_numpy(np.array(target_labels_personne_training_scrambled, dtype=np.int64)))
        target_valid = TensorDataset(
            torch.from_numpy(np.array(target_examples_personne_valid_scrambled, dtype=np.float32)),
            torch.from_numpy(np.array(target_labels_personne_valid_scrambled, dtype=np.int64)))

        # data set -> DataLoader
        target_train_loader = DataLoader(target_train, batch_size=256, shuffle=True, drop_last=True)
        target_valid_loader = DataLoader(target_valid, batch_size=256, shuffle=True, drop_last=True)

        # 装载 17名 target subject 的 dataloader
        list_target_train_dataloader.append(target_train_loader)
        list_target_valid_dataloader.append(target_valid_loader)

        print('No.{} subject'.format(j + 1))
        print('source_examples_personne_training:   ', np.shape(target_examples_personne_training))
        print('source_examples_personne_valid:      ', np.shape(target_examples_personne_valid))
        print('size of source train dataloader:', len(list_target_train_dataloader))
        print('size of source valid dataloader:', len(list_target_valid_dataloader))
        print('-' * 30)

    print('Loading target training/valid data finished.\n\n')

    return list_target_train_dataloader, list_target_valid_dataloader


def load_target_test_data(target_data_testing, target_labels_testing):

    print('Start loading Target test data...')
    list_target_test_loader = []

    # for j in range(17):
    for j in range(1):
        target_examples_personne_testing = []
        target_labels_personne_testing = []

        for k in range(len(target_data_testing[j])):
            target_examples_personne_testing.extend(target_data_testing[j][k])
            target_labels_personne_testing.extend(target_labels_testing[j][k])

        # dataset scrambled
        target_examples_personne_testing_scrambled, target_labels_personne_testing_scrambled \
            = scramble(target_examples_personne_testing, target_labels_personne_testing)

        # numpy2tensor
        target_test = TensorDataset(
            torch.from_numpy(np.array(target_examples_personne_testing_scrambled, dtype=np.float32)),
            torch.from_numpy(np.array(target_labels_personne_testing_scrambled, dtype=np.int64)))

        # data set -> DataLoader
        target_test_loader = DataLoader(target_test, batch_size=1024, shuffle=True, drop_last=True)

        # 装载 17名 target subject 的 dataloader
        list_target_test_loader.append(target_test_loader)

        print('No.{} subject'.format(j + 1))
        print('target_examples_personne_testing:   ', np.shape(target_examples_personne_testing))
        print('size of target_test dataloader:', len(list_target_test_loader))
        print('-' * 30)

    print('Loading target testing data finished.\n\n')

    return list_target_test_loader


def train(dataloader, net, optimizer, class_criterion):
    running_loss = 0.0
    running_corrects, total_num = 0, 0

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        net.train()

        outputs = net(data)
        _, pred_labels = torch.max(outputs.data, 1)

        loss = class_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss = loss.data

        running_loss += loss
        running_corrects += torch.sum(pred_labels == labels.data).item()
        total_num += labels.size(0)

    acc = running_corrects / total_num
    running_loss = running_loss / (i + 1)

    return acc, running_loss


def valid(dataloader, net, class_criterion):
    running_loss = 0.0
    running_corrects, total_num = 0, 0

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()

        net.eval()

        outputs = net(data)
        _, pred_labels = torch.max(outputs.data, 1)

        loss = class_criterion(outputs, labels)
        loss = loss.data

        running_loss += loss
        running_corrects += torch.sum(pred_labels == labels.data).item()
        total_num += labels.size(0)

    acc = running_corrects / total_num
    running_loss = running_loss / (i + 1)

    return acc, running_loss


def test(dataloader, net, class_criterion):
    running_loss = 0.0
    running_corrects, total_num = 0, 0

    for i, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        labels = labels.cuda()

        net.eval()

        outputs = net(data)
        _, pred_labels = torch.max(outputs.data, 1)

        loss = class_criterion(outputs, labels)
        loss = loss.data

        running_loss += loss
        running_corrects += torch.sum(pred_labels == labels.data).item()
        total_num += labels.size(0)

    acc = running_corrects / total_num
    running_loss = running_loss / (i + 1)

    return acc, running_loss








