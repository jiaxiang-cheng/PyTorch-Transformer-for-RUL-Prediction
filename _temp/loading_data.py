import seaborn
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np

seaborn.set_context(context="talk")


def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row (scaled Weibull)
    min_engine = np.exp(-pow((result_frame["max_cycle"] / 225.02895), 4.40869))
    result_frame["min"] = min_engine

    remaining_useful_life = (np.exp(-pow((result_frame["time_cycles"] / 225.02895), 4.40869)) - result_frame["min"]) / (
            1 - result_frame["min"])

    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    result_frame = result_frame.drop("min", axis=1)

    return result_frame


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

group = train_norm.groupby(by="unit_nr")

i = 1

while i <= 100:
    x = group.get_group(i).to_numpy()
    total_loss = 0
    for t in range(x.shape[0]):
        if t == 0:
            X = np.append([np.zeros(14)], x[t:t + 2, 2:-1], axis=0)
            y = x[t, -1:]
        elif t == x.shape[0] - 1:
            X = np.append(x[t - 1:, 2:-1], [np.zeros(14)], axis=0)
        else:
            X = x[t - 1:t + 2, 2:-1]
        y = x[t, -1:]
        X_train_tensors = Variable(torch.Tensor(X))
        y_train_tensors = Variable(torch.Tensor(y))
        X_train_tensors_final = X_train_tensors.reshape((1, 1, X_train_tensors.shape[0], X_train_tensors.shape[1]))

    i += 1
