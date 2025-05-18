import os
import datetime
import numpy as np
import pandas as pd
import torch
from data_provider.data_loader import Dataset_text
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings

warnings.filterwarnings('ignore')




class Dataset_Custom_Text(Dataset, Dataset_text):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', text_len=4):
        # size [seq_len, label_len, pred_len]
        assert flag in ['train', 'test', 'val']
        self.args = args
        self.dataset = self.args.model_id.split('_')[0]
        # get size
        features, text_len, freq = self.get_feature(self.dataset)
        # info
        self.seq_len = self.args.seq_len
        self.label_len = self.args.label_len
        self.token_len = self.args.token_len
        self.percent = self.args.percent
        if flag == 'test':
            self.pred_len = size[2]
        else:
            self.pred_len = self.token_len

        # init
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.text_len = text_len

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_text_dataset_size(self, dataset):
        assert dataset in ['Algriculture', 'Climate', 'Economy', 'Energy', 'Environment', 'Health', 'Security', 'SocialGood', 'Traffictext']
        # [96,84,12] [36,24,12] [8,7,1]
        size_dict = {
            'Algriculture': [8,7,1],
            'Climate': [8,7,1],
            'Economy': [8,7,1],
            'Energy': [32,28,4],
            'Environment': [96,84,12],
            'Health': [32,28,4],
            'Security': [8,7,1],
            'SocialGood': [8,7,1],
            'Traffictext': [8,7,1],
        }
        return size_dict[dataset]

    def get_feature(self, dataset):
        assert dataset in ['Algriculture', 'Climate', 'Economy', 'Energy', 'Environment', 'Health', 'Security', 'SocialGood', 'Traffictext']
        # [96,84,12] [36,24,12] [8,7,1]
        feature_dict = {
            'Algriculture': 'S',
            'Climate': 'S',
            'Economy': 'S',
            'Energy': 'S',
            'Environment': 'S',
            'Health': 'S',
            'Security': 'S',
            'SocialGood': 'S',
            'Traffictext': 'S',
        }
        text_dict = {
            'Algriculture': 4,
            'Climate': 4,
            'Economy': 4,
            'Energy': 4,
            'Environment': 4,
            'Health': 4,
            'Security': 4,
            'SocialGood': 4,
            'Traffictext': 4,
        }
        freq_dict = {
            'Algriculture': 'M',
            'Climate': 'M',
            'Economy': 'M',
            'Energy': 'W',
            'Environment': 'D',
            'Health': 'W',
            'Security': 'M',
            'SocialGood': 'M',
            'Traffictext': 'M',
        }
        return feature_dict[dataset], text_dict[dataset], freq_dict[dataset]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        # print(df_raw.iloc[1247])
        text_name='Final_Search_'+str(self.text_len)
        df_raw = df_raw[['date'] + cols + [self.target]+['prior_history_avg']+['start_date']+['end_date']+[text_name]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # few shot
        if self.set_type == 0:
            print(f'Using few shot of {self.percent}/100')
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            df_data_prior = df_raw[['prior_history_avg']]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_prior = df_raw[['prior_history_avg']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.scaler, data = self.setup(self.scaler, data, self.dataset)
            data_prior = self.scaler.transform(df_data_prior.values[:,-1].reshape(-1, 1))
        else:
            data = df_data.values
            data_prior = df_data_prior.values


        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_prior = data_prior[border1:border2]


        self.data_stamp = data_stamp
        self.date=df_raw[['date']][border1:border2].values
        self.start_date=df_raw[['start_date']][border1:border2].values
        self.end_date=df_raw[['end_date']][border1:border2].values
        self.text=df_raw[[text_name]][border1:border2].values
    def get_prior_y(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends 
        r_ends = r_begins + self.pred_len
        prior_y=np.array([self.data_prior[r_beg:r_end] for r_beg, r_end in zip(r_begins, r_ends)])
        return prior_y
    def get_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        
        text=np.array([self.text[s_end] for s_end in s_ends])
        return text
    def get_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends - self.label_len
        r_ends = r_begins + self.label_len + self.pred_len

        x_start_dates = np.array([self.start_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        x_end_dates = np.array([self.end_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        return x_start_dates, x_end_dates
    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)