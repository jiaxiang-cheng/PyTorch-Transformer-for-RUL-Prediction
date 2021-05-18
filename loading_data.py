
import pandas as pd
from add_remaining_useful_life import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def loading_FD001():
    # load data FD001.py
    # define filepath to read data
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # drop non-informative features, derived from EDA
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors
    train.drop(labels=drop_labels, axis=1, inplace=True)

    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]

    data_norm = (data - data.min()) / (data.max() - data.min())

    train_norm = pd.concat([title, data_norm], axis=1)

    train_norm = add_remaining_useful_life(train_norm)
    train_norm['RUL'].clip(upper=140, inplace=True)

    group = train_norm.groupby(by="unit_nr")

    test.drop(labels=drop_labels, axis=1, inplace=True)

    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]

    data_norm = (data - data.min()) / (data.max() - data.min())

    test_norm = pd.concat([title, data_norm], axis=1)

    group_test = test_norm.groupby(by="unit_nr")

    return group, y_test, group_test