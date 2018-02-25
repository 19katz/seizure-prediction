import scipy as sc
import scipy.signal
import os
import shutil
import numpy as np
from scipy.io import loadmat, savemat
from pandas import DataFrame

def get_files_paths(directory, extension='.mat'):
    filenames = sorted(os.listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames
                            if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension


def filter(x, new_sampling_frequency, data_length_sec, locut, hicut):
    x1 = scipy.signal.resample(x, new_sampling_frequency * data_length_sec, axis=1)

    nyq = 0.5 * new_sampling_frequency
    b, a = sc.signal.butter(5, np.array([locut, hicut]) / nyq, btype='band')
    x_filt = sc.signal.lfilter(b, a, x1, axis=1)
    return np.float32(x_filt)

def group_into_bands(fft, fft_freq, nfreq_bands):
    if nfreq_bands == 3:
        bands = [0.1, 7, 14, 49]
    elif nfreq_bands == 6:
        bands = [0.1, 4, 8, 12, 30, 70, 180]
    elif nfreq_bands == 8:
        bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 180]
    else:
        raise ValueError('Wrong number of frequency bands')
    freq_bands = np.digitize(fft_freq, bands)
    df = DataFrame({'fft': fft, 'band': freq_bands})
    df = df.groupby('band').mean()
    return df.fft[1:-1]


def compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec):
    n_channels = x.shape[0]
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1

    x2 = np.zeros((n_channels, nfreq_bands, n_timesteps))
    for i in range(n_channels):
        xc = np.zeros((nfreq_bands, n_timesteps))
        for frame_num, w in enumerate(range(0, data_length_sec - win_length_sec + 1, stride_sec)):
            xw = x[i, w * sampling_frequency: (w + win_length_sec) * sampling_frequency]
            fft = np.log10(np.absolute(np.fft.rfft(xw)))
            fft_freq = np.fft.rfftfreq(n=xw.shape[-1], d=1.0 / sampling_frequency)
            xc[:nfreq_bands, frame_num] = group_into_bands(fft, fft_freq, nfreq_bands)
        x2[i, :, :] = xc
    return x2


def process_file(raw_file_path, read_dir, write_dir, locut, hicut,
                 nfreq_bands, win_length_sec, stride_sec):
    d = loadmat(raw_file_path)
    sample = ''
    for key in d.keys():
        if 'segment' in key:
            sample = key
            break
    x = np.array(d[sample][0][0][0], dtype='float32')
    data_length_sec = d[sample][0][0][1][0][0]
    if 'test' in raw_file_path or 'holdout' in raw_file_path:
        sequence = np.Inf
    else:
        sequence = d[sample][0][0][4][0][0]


    new_sampling_frequency = 400
    new_x = filter(x, new_sampling_frequency, data_length_sec, locut, hicut)
    x = compute_fft(new_x, data_length_sec, new_sampling_frequency, nfreq_bands, win_length_sec, stride_sec)

    data_dict = {'data': x, 'data_length_sec': data_length_sec,
                 'sampling_frequency': new_sampling_frequency, 'sequence': sequence}

    preprocessed_file_path = raw_file_path.replace(read_dir, write_dir)
    savemat(preprocessed_file_path, data_dict)


def run_preprocessor(raw_data_path, output_data_path, subjects, locut, hicut,
                     nfreq_bands, win_length_sec, stride_sec):
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

    print("preprocessing: ", subjects)

    for subject in subjects:
        print '>> preprocessing ', subject
        read_dir = raw_data_path + '/' + subject
        write_dir = output_data_path + '/' + subject

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        for raw_file_path in get_files_paths(read_dir):
            process_file(raw_file_path, read_dir, write_dir, locut, hicut,
                         nfreq_bands, win_length_sec, stride_sec)

if __name__ == '__main__':
    subjects = ['Dog_2', ]
    run_preprocessor('data', 'processed_data', subjects, 0.1, 180, 3, 60, 60)
