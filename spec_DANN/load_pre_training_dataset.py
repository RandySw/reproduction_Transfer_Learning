import numpy as np
import cal_spectrogram

number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5


def format_data_to_train(vector_to_format):
    dataset_example_formatted = []
    example = []
    emg_vector = []
    for value in vector_to_format:
        emg_vector.append(value)
        if (len(emg_vector) >= 8):
            if (example == []):
                example = emg_vector
            else:
                example = np.row_stack((example, emg_vector))
            emg_vector = []
            if (len(example) >= number_of_vector_per_example):
                example = example.transpose()
                dataset_example_formatted.append(example)
                example = example.transpose()
                example = example[size_non_overlap:]
    test = cal_spectrogram.calculate_spectrogram_dataset(dataset_example_formatted)
    return np.array(test)


def shift_electrodes(examples, labels):
    index_normal_class = [1, 2, 6, 2]  # The normal activation of the electrodes.
    class_mean = []
    # For the classes that are relatively invariant to the highest canals activation, we get on average for a
    # subject the most active cannals for those classes
    for classe in range(3, 7):
        X_example = []
        Y_example = []
        for k in range(len(examples)):
            X_example.extend(examples[k])
            Y_example.extend(labels[k])

        spectrogram_add = []
        for j in range(len(X_example)):
            if Y_example[j] == classe:
                if spectrogram_add == []:
                    spectrogram_add = np.array(X_example[j][0])
                else:
                    spectrogram_add += np.array(X_example[j][0])
        class_mean.append(np.argmax(np.sum(np.array(spectrogram_add), axis=0)))

    # We check how many we have to shift for each channels to get back to the normal activation
    new_spectrogram_emplacement_left = ((np.array(class_mean) - np.array(index_normal_class)) % 10)
    new_spectrogram_emplacement_right = ((np.array(index_normal_class) - np.array(class_mean)) % 10)

    shifts_array = []
    for valueA, valueB in zip(new_spectrogram_emplacement_left, new_spectrogram_emplacement_right):
        if valueA < valueB:
            # We want to shift toward the left (the start of the array)
            orientation = -1
            shifts_array.append(orientation * valueA)
        else:
            # We want to shift toward the right (the end of the array)
            orientation = 1
            shifts_array.append(orientation * valueB)

    # We get the mean amount of shift and round it up to get a discrete number representing how much we have to shift
    # if we consider all the canals
    # Do the shifting only if the absolute mean is greater or equal to 0.75
    final_shifting = np.mean(np.array(shifts_array))
    if abs(final_shifting) >= 0.5:
        final_shifting = int(np.round(final_shifting))
    else:
        final_shifting = 0

    # Build the dataset of the candiate with the circular shift taken into account.
    X_example = []
    Y_example = []
    for k in range(len(examples)):
        sub_ensemble_example = []
        for example in examples[k]:
            sub_ensemble_example.append(np.roll(np.array(example), final_shifting))
        X_example.append(sub_ensemble_example)
        Y_example.append(labels[k])
    return X_example, Y_example


def read_data(path):

    print('Reading Data...')
    list_dataset = []
    list_labels = []

    for candidate in range(12):
        labels = []
        examples = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path+'/Male'+str(candidate)+'/training0/classe_%d.dat' % i,
                                              dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example = format_data_to_train(data_read_from_file)
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
        examples, labels = shift_electrodes(examples, labels)
        list_dataset.append(examples)
        list_labels.append(labels)

    for candidate in range(7):
        labels = []
        examples = []
        for i in range(number_of_classes * 4):
            i=0     # szy: 这句什么鬼？？？？ -> 对每个female 重复加载其第一个样本27次 而不是读取其所有的27个样本
            data_read_from_file = np.fromfile(path + '/Female' + str(candidate) + '/training0/classe_%d.dat' % i,
                                              dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example = format_data_to_train(data_read_from_file)
            # print(dataset_example[0][0][0][:5])
            # print()
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
        examples, labels = shift_electrodes(examples, labels)
        list_dataset.append(examples)
        list_labels.append(labels)

    print('Finished Reading Data')
    return list_dataset, list_labels


if __name__ == '__main__':
    examples, labels = read_data('../../PreTrainingDataset')  # 加载数据集
    datasets = [examples, labels]

    np.save("formatted_datasets/saved_pre_training_dataset_spectrogram.npy", datasets)

    datasets_pre_training = np.load("formatted_datasets/saved_pre_training_dataset_spectrogram.npy",
                                    encoding="bytes", allow_pickle=True)
    print(np.shape(datasets_pre_training))


