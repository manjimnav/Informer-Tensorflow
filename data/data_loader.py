import os
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')


class DatasetETT:
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, batch_size=32, shuffle=True, drop_last=True, is_minute=False):

        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.is_minute = is_minute

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        minute_multiplier = 1
        if self.is_minute:
            minute_multiplier = 4

        border1s = [0, 12 * 30 * 24 * minute_multiplier - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 * minute_multiplier - self.seq_len]
        border2s = [12 * 30 * 24 * minute_multiplier,
                    12 * 30 * 24 + 4 * 30 * 24 * minute_multiplier,
                    12 * 30 * 24 + 8 * 30 * 24 * minute_multiplier]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)

        if self.is_minute:
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)

        data_stamp = df_stamp.drop(['date'], 1).values

        self.data_x = data[border1:border2]
        # self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __split_window__(self, window):

        y_start = self.seq_len - self.label_len
        y_end = y_start + self.label_len + self.pred_len

        return window[:self.seq_len], window[y_start:y_end]

    def __process_window__(self, window, window_mark):

        x_enc = window[0]
        x_dec = tf.identity(window[1])
        x_dec = tf.concat([x_dec[:-self.pred_len], tf.zeros((self.pred_len, x_dec.shape[1]), dtype=tf.float64)], 0)

        x_mark_enc = window_mark[0]
        x_mark_dec = window_mark[1]

        y = window[1][-self.pred_len:]

        return (x_enc, x_dec, x_mark_enc, x_mark_dec), y

    def __create_dataset__(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.window(self.seq_len + self.pred_len, shift=1, drop_remainder=self.drop_last)

        # https://stackoverflow.com/questions/55429307/how-to-use-windows-created-by-the-dataset-window-method-in-tensorflow-2-0
        dataset = dataset.flat_map(lambda window: window.batch(self.seq_len + self.pred_len))
        dataset = dataset.map(self.__split_window__)

        return dataset

    def get_dataset(self):

        dataset_features = self.__create_dataset__(self.data_x)
        dataset_stamp = self.__create_dataset__(self.data_stamp)

        dataset = tf.data.Dataset.zip((dataset_features, dataset_stamp))
        dataset = dataset.map(self.__process_window__)

        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size)

        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_last).repeat()

        return dataset.prefetch(5)

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1)//self.batch_size


