import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import config as cf

'''
kdd_data = np.load(os.path.join(cf.data_dir, cf.data_filename))['kdd']
df = pd.DataFrame(kdd_data)
normal_df = df[df.iloc[:, -1] == 0]
print('df.shape: ', df.shape)
print('normal shape: ', normal_df.shape)
print('normal ratio: %.4f' % (normal_df.shape[0]/df.shape[0]))
'''

class Intrusion_Dataset(data.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        
        kdd_data = np.load(os.path.join(cf.data_dir, cf.data_filename))['kdd']
        kdd_data = torch.from_numpy(kdd_data).float()
        normal_data = kdd_data[kdd_data[:, -1] == 0]
        abnormal_data = kdd_data[kdd_data[:, -1] == 1]

        train_normal_mark = int(normal_data.shape[0] * cf.train_ratio)
        train_abnormal_mark = int(abnormal_data.shape[0] * cf.train_ratio)

        train_normal_data = normal_data[:train_normal_mark, :]
        train_abnormal_data = abnormal_data[:train_abnormal_mark, :]
        self.train_data = np.concatenate((train_normal_data, train_abnormal_data), axis=0)
        np.random.shuffle(self.train_data)
        
        test_normal_data = normal_data[train_normal_mark:, :]
        test_abnormal_data = abnormal_data[train_abnormal_mark:, :]
        self.test_data = np.concatenate((test_normal_data, test_abnormal_data), axis=0)
        np.random.shuffle(self.test_data)
        
    def __len__(self):
        if self.mode == 'train':
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]
        
    def __getitem__(self, index):
        if self.mode == 'train':
            x = self.train_data[index, :-1]
            y = self.train_data[index, -1]
        else:
            x = self.test_data[index, :-1]
            y = self.test_data[index, -1]
        return x, y
    
    def set_mode(self, mode):
        self.mode = mode