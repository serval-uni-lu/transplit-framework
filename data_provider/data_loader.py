import os
import pandas as pd
import os
from torch.utils.data import Dataset, BatchSampler
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from typing import List, Iterator
import warnings
import numpy as np
from math import floor


warnings.filterwarnings('ignore')

class ElectricityDataset(Dataset):
    def __init__(
            self,
            df,
            flag='train',
            seq_len=720,
            pred_len=168,
            add_features=None,
            scale=True,
            split=(0.7, 0.1, 0.2),
            timeenc=0,
            period=24,
            freq='h'
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.split = split

        self.scale = scale
        self.timeenc = timeenc
        self.period = period
        self.freq = freq

        self.add_features = add_features
        self.__read_data__(df)

    def __read_data__(self, df_raw):
        # features loading
        try:
            df_features = pd.read_csv(self.add_features)
            print(f"Additional features loaded: {self.add_features}")
            float_features = df_features.select_dtypes(include=['float'])
            int_features = df_features.select_dtypes(include=['int'])
            # transform each feature to an integer index
            for col in int_features.columns:
                int_features[col] = int_features[col].astype('category').cat.codes
            int_features = int_features.values.astype(np.int64)
            float_features = float_features.values
            self.use_features = True
        except Exception as e:
            print("No additional features are used.")
            # print(e)
            self.int_features = None
            self.float_features = None
            self.use_features = False

        # main data loading
        self.scaler = StandardScaler()
        '''
        df_raw.columns: ['date', features]
        '''
        cols = list(df_raw.columns[1:])
        # print(cols)
        num_train = int(len(df_raw) * self.split[0])
        num_test = int(len(df_raw) * self.split[2])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        border1 = border1 // self.period * self.period
        border2 = border2 // self.period * self.period

        df_data = df_raw[cols]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

            if self.use_features:
                # also scale the additional features
                float_features -= float_features.mean(axis=0)
                float_features /= float_features.std(axis=0)
        else:
            data = df_data.values

        print(f"data.shape: {data.shape}")

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1)
            for col in data_stamp.columns:
                data_stamp[col] = data_stamp[col].astype('category').cat.codes
            data_stamp = data_stamp.values.astype(np.int64)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if not self.use_features:
            # set it to empty arrays
            float_features = data[:,:0]
            int_features = data_stamp[:,:0]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]
        self.float_features = float_features[border1:border2]
        self.int_features = int_features[border1:border2]

        print(f"self.data_x.shape: {self.data_x.shape}")

    def __getitem__(self, index):
        # input and output boudaries
        s_begin = index // self.data_x.shape[1] * self.period
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        f_begin = r_end - self.seq_len  # for additional features

        i = index % self.data_x.shape[1]  # i = consumer id
        seq_x = self.data_x[s_begin:s_end, i:i+1]
        seq_y = self.data_y[r_begin:r_end, i:i+1]
        mean = self.scaler.mean_[i, None, None]
        scale = self.scaler.scale_[i, None, None]

        # concatenate aligned additional time series (future data known in advance and given as input)
        seq_x = np.concatenate([
            seq_x,
            self.float_features[f_begin:r_end]
        ], axis=-1)

        # concatenate additional categorical features to time stamps (known in advance)
        seq_x_mark = np.concatenate([
            self.data_stamp[s_begin:s_end],
            self.int_features[f_begin:r_end]
        ], axis=-1)
        seq_y_mark = np.concatenate([
            self.data_stamp[r_begin:r_end],
            self.int_features[r_begin:r_end]
        ], axis=-1)

        return {
            'seq_x': seq_x,
            'seq_y': seq_y,
            'seq_x_mark': seq_x_mark,
            'seq_y_mark': seq_y_mark,
            'mean': mean,
            'scale': scale
        }

    def __len__(self):
        if len(self.data_x) < self.seq_len + self.pred_len:
            return 0
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        return length // self.period * self.data_x.shape[1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class DatesBatchSampler(BatchSampler):
    """
    Provides batches in which all sequences are aligned on a random date.
    Not currently used.
    """

    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.dataset = dataset
        self.channels = dataset.data_x.shape[1]
        self.data_len = len(dataset) // self.channels
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        time_indices = np.arange(self.data_len)
        for t in time_indices:
            channel_indices = np.arange(self.channels)
            np.random.shuffle(channel_indices)
            for i in range(0, self.channels, self.batch_size):
                c = channel_indices[i:i + self.batch_size]
                if len(c) < self.batch_size and self.drop_last:
                    break
                c += t * self.channels
                batches.append(c.tolist())
        np.random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        # TODO: check formula with and without drop_last
        return self.channels * self.data_len // self.batch_size