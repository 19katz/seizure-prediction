import os
import numpy as np
from scipy.io import loadmat
from sklearn.cross_validation import StratifiedShuffleSplit
from eeg_cnn_lib import train
import preprocess

def load_train_data(data_path, subject):
    read_dir = data_path + '/' + subject
    filenames = sorted(os.listdir(read_dir))
    train_filenames = []
    for filename in filenames:
        if 'test' not in filename:
            train_filenames.append(filename)

    n = len(train_filenames)
    datum = loadmat(read_dir + '/' + train_filenames[0], squeeze_me=True)
    x = np.zeros(((n,) + datum['data'].shape), dtype='float32')
    y = np.zeros(n, dtype='int8')

    for i, filename in enumerate(train_filenames):
        datum = loadmat(read_dir + '/' + filename, squeeze_me=True)
        x[i] = datum['data']
        y[i] = 1 if 'preictal' in filename else 0

    return x, y


def oversample_minority(train, target):
    ones_ind  = [i for i in range(len(target)) if target[i] == 1]
    zeros_ind = [i for i in range(len(target)) if target[i] == 0]

    print "Calling oversample_minority"

    if len(ones_ind) >= len(zeros_ind) or len(ones_ind) == 0:
        return train, target 

    new_train = np.zeros((2*len(zeros_ind),) + train.shape[1:], dtype='float32')
    new_target = np.zeros(2*len(zeros_ind), dtype='int8')
    ind = 0
    next_one = 0
    
    for i in zeros_ind:
        new_train[ind, :, :, :] = train[i, :, :, :]
        new_target[ind] = target[i]
        ind += 1

        new_train[ind, :, :, :] = train[ones_ind[next_one], :, :, :]
        new_target[ind] = target[ones_ind[next_one]]
        ind += 1
        next_one += 1
        if next_one >= len(ones_ind):
            next_one = 0

    print("Input train shape:", train.shape, ", after oversample, train shape:", new_train.shape)
    return new_train, new_target
                

def channel_id2xy(channel_id, total):
    id2xy_map16 = [(0,0), (0,4), (0,8), (0,12), (4,0), (4,4), (4,8), (4,12),
                   (8,0), (8,4), (8,8), (8,12), (12,0), (12,4), (12,8), (12,12),]

    id2xy_map24 = [(0,0), (0,4), (0,8), (0,12), (0,15), (4,0), (4,4), (4,8), (4,12), (4,15),
                   (8,0), (8,4), (8,8), (8,12), (8,15), (12,0), (12,4), (12,8), (12,12), (12,15),
                   (15,0), (15,4), (15,8), (15,12)]

    return id2xy_map16[channel_id] if total <= 16 else id2xy_map24[channel_id]

def channels_to_imgs(data, img_size=16):
    (n_segments, n_channels, n_bands, n_wins) = data.shape

    segment_imgs = np.zeros((n_segments, n_bands, img_size, img_size), dtype='float32')
    time_win_imgs = np.zeros((n_wins, n_segments, n_bands, img_size, img_size), dtype='float32')

    for i in range(n_segments):
        for j in range(n_channels):
            for k in range(n_bands):
                (x, y) = channel_id2xy(j, n_channels)
                segment_imgs[i, k, x, y] = np.mean(data[i, j, k, :])
                for l in range(n_wins):
                    time_win_imgs[l,i,k,x,y] = data[i,j,k,l]
    
    return (segment_imgs, time_win_imgs)
    
def load_data_and_train(subject, data_path):
    print('Loading data...')

    x, y = load_train_data(data_path, subject)
    print 'Data dimensions: ', x.shape

    skf = StratifiedShuffleSplit(y, n_iter=1, test_size=0.20, random_state=0)

    # This loop is executed only once
    for train_idx, valid_idx in skf:
        x_train, y_train = x[train_idx], y[train_idx]
        x_valid, y_valid = x[valid_idx], y[valid_idx]

        print('============ Training: ', subject, ', x train shape:', x_train.shape,
              ', y train shape:', y_train.shape,  'x valid shape:', x_valid.shape,
              ', y valid shape:', y_valid.shape)

        # (x_train, y_train) = oversample_minority(x_train, y_train)
        (x_train, x_train_win) = channels_to_imgs(x_train)
        (x_valid, x_valid_win) = channels_to_imgs(x_valid)
 
        # data = (x_train, y_train, x_valid, y_valid, None, None)
        # print('Training CNN with single images per segment, labels: ', np.unique(y_train))
        # train(data, 'cnn', batch_size=1)

        data = (x_train_win, y_train, x_valid_win, y_valid, None, None)
        # print('Training CNN-LSTM with image sequences per segment, labels: ', np.unique(y_train))
        train(data, 'lstm', batch_size=10, n_colors=3)
        
    print 'Done!'

def load_crossdata_and_train(data_path, subjects, subject):
    print('Loading data...')

    train_filenames = []
    for s in subjects:
        read_dir = data_path + '/' + s
        filenames = sorted(os.listdir(read_dir))
        for filename in filenames:
            if 'test' not in filename:
                train_filenames.append(read_dir + '/' + filename)

    n = len(train_filenames)
    datum = loadmat(train_filenames[0], squeeze_me=True)
    x = np.zeros(((n,) + datum['data'].shape), dtype='float32')
    y = np.zeros(n, dtype='int8')

    for i, filename in enumerate(train_filenames):
        # print 'Processing ', filename
        datum = loadmat(filename, squeeze_me=True)
        x[i] = datum['data']
        y[i] = 1 if 'preictal' in filename else 0

    x_test, y_test = load_train_data(data_path, subject)
    print 'Test Data dimensions: ', x_test.shape

    skf = StratifiedShuffleSplit(y, n_iter=1, test_size=0.001, random_state=0)
    # This loop is executed only once
    for train_idx, valid_idx in skf:
        x_train, y_train = x[train_idx], y[train_idx]
        x_valid, y_valid = x[valid_idx], y[valid_idx]

        print('============ Training: ', subject, ', x train shape:', x_train.shape,
              ', y train shape:', y_train.shape,  'x valid shape:', x_valid.shape,
              ', y valid shape:', y_valid.shape)

        # (x_train, y_train) = oversample_minority(x_train, y_train)
        (x_train, x_train_win) = channels_to_imgs(x_train)
        (x_valid, x_valid_win) = channels_to_imgs(x_valid)
        (x_test, x_test_win) = channels_to_imgs(x_test)
 
        # print('Training CNN with single images per segment')
        # train(data, 'cnn', batch_size=1)

        data = (x_train_win, y_train, x_valid_win, y_valid, x_test_win, y_test)
        # print('Training CNN-LSTM with image sequences per segment')
        train(data, 'lstm', batch_size=10, n_colors=3)

    print 'Done!'

def run_trainer(data_path):
    locut = 0.1
    hicut = 180
    n_bands = 3
    win_leng_sec = 60
    stride_sec = 60

    if not os.path.exists(data_path):
        preprocess.run_preprocessor('data', data_path, ['Dog_2', ], locut, hicut,
                                    n_bands, win_leng_sec, stride_sec)

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        print '******Subject specific prediction for: ', subject, ' (using own data only)******'
        if not os.path.exists(data_path + '/' + subject):
            preprocess.run_preprocessor('data', data_path, [subject, ],
                                        locut, hicut, n_bands, win_leng_sec, stride_sec)
        load_data_and_train(subject, data_path)

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', ]
    for subject in subjects:
        rest = [subjects[i] for i in range(len(subjects)) if subjects[i] != subject]
        print('Using ', rest, ' to predict for ***********************', subject, '***************************')
        load_crossdata_and_train(rest, subject, data_path)

    print('Using Dogs 1-4 to predict for ***********************Dog_5***************************')
    load_crossdata_and_train(subjects, 'Dog_5', data_path)

if __name__ == '__main__':
    run_trainer('processed_data')
