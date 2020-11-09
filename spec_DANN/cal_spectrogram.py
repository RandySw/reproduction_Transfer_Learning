import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def calculate_spectrogram_dataset(dataset):
    """
    :param dataset: (189, (8, 52))
    :return:
    """

    dataset_spectrogram = []
    for examples in dataset:    # examples -> (8, 52)
        canals = []
        for electrode_vector in examples:   # electrode -> (52, )
            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                cal_spectrogram_vector(electrode_vector, npserseg=28, noverlap=20)
            # spectrogram_of_vector <ndarray: (15, 4)>
            # remove the low frequency signal as it's useless for sEMG (0-5Hz)
            spectrogram_of_vector = spectrogram_of_vector[1:]   # spectrogram_of_vector <ndarray: (14, 4)>
            canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))     # canals (8, (4, 14))

        # 将通道维度和时间维度作交换，得到按时序（4个时刻）划分的8通道肌电能量谱图
        example_to_classify = np.swapaxes(canals, 0, 1)     # example_to_classify <tuple: (4, 8, 14)>
        dataset_spectrogram.append(example_to_classify)

    return dataset_spectrogram


def cal_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window='hann',
                                                                                         scaling='spectrum')

    return spectrogram_of_vector, time_segment_sample, frequencies_samples


def show_spectrogram(frequencies_samples, time_segment_sample, spectrogram_of_vector):
    plt.rcParams.update({'font.size': 36})
    print(np.shape(spectrogram_of_vector))
    print(np.shape(time_segment_sample))
    print(np.shape(frequencies_samples))

    time_segment_sample = [0., 65., 130., 195., 250.]
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off',
    )
    print(time_segment_sample)
    plt.pcolormesh(time_segment_sample, frequencies_samples, spectrogram_of_vector)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [ms]')
    plt.title('STFT')
    plt.show()


