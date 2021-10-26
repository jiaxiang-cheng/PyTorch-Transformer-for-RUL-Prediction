import pandas as pd
from add_remaining_useful_life import *


def loading_FD001():

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

    # drop non-informative features in training set
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors
    train.drop(labels=drop_labels, axis=1, inplace=True)

    # separate title information and sensor data
    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]

    # min-max normalization of the sensor data
    data_norm = (data - data.min()) / (data.max() - data.min())
    train_norm = pd.concat([title, data_norm], axis=1)

    # add piece-wise target remaining useful life
    train_norm = add_remaining_useful_life(train_norm)
    train_norm['RUL'].clip(upper=125, inplace=True) # in the paper the MAX RUL is mentioned as 125

    # group the training set with unit
    group = train_norm.groupby(by="unit_nr")

    # drop non-informative features in testing set
    test.drop(labels=drop_labels, axis=1, inplace=True)
    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())
    test_norm = pd.concat([title, data_norm], axis=1)

    # group the testing set with unit
    group_test = test_norm.groupby(by="unit_nr")

    return group, y_test, group_test
