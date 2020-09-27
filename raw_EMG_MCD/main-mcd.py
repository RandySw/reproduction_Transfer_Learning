import os
import numpy as np
import dataProcess
import paramsTrans
from utilsTrans import *
from actualFunc import train, test
from multiprocessing.spawn import freeze_support
from classifierTrans import LeNetClassifier
from extractorTrans import LeNetEncoder

extractor = LeNetEncoder()
classifier1 = LeNetClassifier()
classifier2 = LeNetClassifier()

extractor.apply(init_weights)
classifier1.apply(init_weights)
classifier2.apply(init_weights)

if __name__ == '__main__':
    datasets_pre_training = np.load("../MCD-trans/formatted_datasets/saved_pre_training_dataset_spectrogram.npy",
                                    encoding="bytes", allow_pickle=True)
    examples_pre_training, labels_pre_training = datasets_pre_training

    src_list_train_dataloader, src_list_validation_dataloader = \
        dataProcess.sourceDataLoader(examples_pre_training, labels_pre_training)

    # ----------------------------------- source domain data loader ---------------------------------------------

    datasets_training = np.load("../MCD-trans/formatted_datasets/saved_evaluation_dataset_training.npy",
                                encoding="bytes", allow_pickle=True)
    examples_training, labels_training = datasets_training

    datasets_test0 = np.load("../MCD-trans/formatted_datasets/saved_evaluation_dataset_test0.npy",
                             encoding="bytes", allow_pickle=True)
    examples_test0, labels_test0 = datasets_test0

    datasets_test1 = np.load("../MCD-trans/formatted_datasets/saved_evaluation_dataset_test1.npy",
                             encoding="bytes", allow_pickle=True)
    examples_test1, labels_test1 = datasets_test1

    tgt_list_train_dataloader, tgt_list_validation_dataloader = dataProcess.targetDataLoader(examples_training,
                                                                                             labels_training,
                                                                                             examples_test0,
                                                                                             labels_test0,
                                                                                             examples_test1,
                                                                                             labels_test1)
    # ----------------------------------- target domain data loader ---------------------------------------------

    src_data = []
    for k in range(len(src_list_train_dataloader)):
        src_data.extend(src_list_train_dataloader[k])

    src_data_valid = []
    for m in range(len(src_list_train_dataloader)):
        src_data_valid.extend(src_list_validation_dataloader[m])

    tgt_data = []
    for j in range(len(tgt_list_train_dataloader)):
        tgt_data.extend(tgt_list_train_dataloader[j])

    tgt_data_valid = []
    for g in range(len(tgt_list_train_dataloader)):
        tgt_data_valid.extend(tgt_list_validation_dataloader[g])

    freeze_support()
    for i in range(paramsTrans.num_epoch):
        print("epoch={}".format(i))

        extractor, classifier1, classifier2 = train(src_data, tgt_data,
                                                    extractor, classifier1, classifier2)

        torch.save(extractor, os.path.join(paramsTrans.models_save, "extractor.pth"))
        torch.save(classifier1, os.path.join(paramsTrans.models_save, "classifier1.pth"))
        torch.save(classifier2, os.path.join(paramsTrans.models_save, "classifier2.pth"))

        # 源域测试
        test(src_data_valid, extractor, classifier1, classifier2)
        # 目标域测试
        test(tgt_data_valid, extractor, classifier1, classifier2)
        print()
