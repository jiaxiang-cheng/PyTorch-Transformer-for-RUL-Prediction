import copy
import math
import seaborn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

seaborn.set_context(context="talk")


class Transformer(nn.Module):
    def __init__(self, m, d_model, N, heads):
        super().__init__()
        self.gating = Gating(d_model, m)
        self.encoder = Encoder(d_model, N, heads, m)
        self.out = nn.Linear(d_model, 1)

    def forward(self, src, t):
        e_i = self.gating(src)
        e_outputs = self.encoder(e_i, t)
        output = self.out(e_outputs)
        return output.reshape(1)


class Gating(nn.Module):
    def __init__(self, d_model, m):
        super().__init__()
        self.m = m

        # the reset gate r_i
        self.W_r = nn.Parameter(torch.Tensor(m, m))
        self.V_r = nn.Parameter(torch.Tensor(m, m))
        self.b_r = nn.Parameter(torch.Tensor(m))

        # the update gate u_i
        self.W_u = nn.Parameter(torch.Tensor(m, m))
        self.V_u = nn.Parameter(torch.Tensor(m, m))
        self.b_u = nn.Parameter(torch.Tensor(m))

        # the update gate u_i
        self.W_e = nn.Parameter(torch.Tensor(m, d_model))
        self.b_e = nn.Parameter(torch.Tensor(d_model))

        self.init_weights()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 1), stride=1),
        )

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.m)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x_i = x[:, :, 1:2, :]
        h_i = self.cnn_layers(x)
        # print(x_i.size())
        # print(h_i.size())

        # r_i = torch.sigmoid(torch.matmul(self.W_r, h_i) + torch.matmul(self.V_r, x_i) + self.b_r)
        # u_i = torch.sigmoid(torch.matmul(self.W_u, h_i) + torch.matmul(self.V_u, x_i) + self.b_u)
        r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
        u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)

        # the output of the gating mechanism
        hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)

        return torch.matmul(hh_i, self.W_e) + self.b_e


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, m):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, t):
        src = src.reshape(1, 128)
        # print(src.size())
        x = self.pe(src, t)
        for i in range(self.N):
            x = self.layers[i](x, None)
        return self.norm(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x, t):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        pe = np.zeros(self.d_model)
        for i in range(0, self.d_model, 2):
            pe[i] = math.sin(t / (10000 ** ((2 * i) / self.d_model)))
            pe[i + 1] = math.cos(t / (10000 ** ((2 * (i + 1)) / self.d_model)))

        x = x + Variable(torch.Tensor(pe))
        return x


# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # print("x:", x.size())
        x2 = self.norm_1(x)
        # print("x2:", x2.size())
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
        # self.linear_reg = nn.Linear(d_model, 1)

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        # output = torch.sigmoid(self.linear_reg(norm))
        # print("Norm:", norm.size())
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
    # scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    min_engine = np.exp(-pow((result_frame["max_cycle"] / 225.02895), 4.40869))
    result_frame["min"] = min_engine

    remaining_useful_life = (np.exp(-pow((result_frame["time_cycles"] / 225.02895), 4.40869)) - result_frame["min"]) / (
            1 - result_frame["min"])

    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    result_frame = result_frame.drop("min", axis=1)

    return result_frame


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

# train = add_remaining_useful_life(train)
# train['RUL'].clip(upper=140, inplace=True)
#
# title = train.iloc[:, 0:2]
# data = train.iloc[:, 2:]
#
# data_norm = (data - data.min()) / (data.max() - data.min())
#
# train_norm = pd.concat([title, data_norm], axis=1)
#
# group = train_norm.groupby(by="unit_nr")

title = train.iloc[:, 0:2]
data = train.iloc[:, 2:]

data_norm = (data - data.min()) / (data.max() - data.min())

train_norm = pd.concat([title, data_norm], axis=1)

train_norm = add_remaining_useful_life(train_norm)
# train['RUL'].clip(upper=140, inplace=True)

group = train_norm.groupby(by="unit_nr")

num_epochs = 4
d_model = 128
heads = 4
N = 2
m = 14

model = Transformer(m, d_model, N, heads)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# this code is very important! It initialises the parameters with a
# range of values that stops the signal fading or getting too big.
# See this blog for a mathematical explanation.
# optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
optim = torch.optim.Adam(model.parameters(), lr=1 / 2e+100)

criterion = torch.nn.MSELoss()  # mean-squared error for regression

scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[1,
                                                                    2,
                                                                    3,
                                                                    4,
                                                                    5,
                                                                    6,
                                                                    7,
                                                                    8,
                                                                    9,
                                                                    10,
                                                                    11,
                                                                    12,
                                                                    13,
                                                                    14,
                                                                    15,
                                                                    16,
                                                                    17,
                                                                    18,
                                                                    19,
                                                                    20,
                                                                    21,
                                                                    22,
                                                                    23,
                                                                    24,
                                                                    25,
                                                                    26,
                                                                    27,
                                                                    28,
                                                                    29,
                                                                    30,
                                                                    31,
                                                                    32,
                                                                    33,
                                                                    34,
                                                                    35,
                                                                    36,
                                                                    37,
                                                                    38,
                                                                    39,
                                                                    40,
                                                                    41,
                                                                    42,
                                                                    43,
                                                                    44,
                                                                    45,
                                                                    46,
                                                                    47,
                                                                    48,
                                                                    49,
                                                                    50,
                                                                    51,
                                                                    52,
                                                                    53,
                                                                    54,
                                                                    55,
                                                                    56,
                                                                    57,
                                                                    58,
                                                                    59,
                                                                    60,
                                                                    61,
                                                                    62,
                                                                    63,
                                                                    64,
                                                                    65,
                                                                    66,
                                                                    67,
                                                                    68,
                                                                    69,
                                                                    70,
                                                                    71,
                                                                    72,
                                                                    73,
                                                                    74,
                                                                    75,
                                                                    76,
                                                                    77,
                                                                    78,
                                                                    79,
                                                                    80,
                                                                    81,
                                                                    82,
                                                                    83,
                                                                    84,
                                                                    85,
                                                                    86,
                                                                    87,
                                                                    88,
                                                                    89,
                                                                    90,
                                                                    91,
                                                                    92,
                                                                    93,
                                                                    94,
                                                                    95,
                                                                    96,
                                                                    97,
                                                                    98,
                                                                    99,
                                                                    100], gamma=2)

i = 1
epoch_loss = []
while i <= 100:
    x = group.get_group(i).to_numpy()
    total_loss = 0
    optim.zero_grad()
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
        # train_x = train_x.reshape(train_x.shape[0], 1, 15, nf)
        # forward pass
        outputs = model.forward(X_train_tensors_final, t)

        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)

        # calculates the loss of the loss function
        # loss.backward()

        # improve from loss, i.e back propagation
        # optim.step()

        total_loss += loss.item()

        # predictions = model(inputs)  # Forward pass
        # loss = loss_function(predictions, labels)  # Compute loss function
        loss = loss / x.shape[0]  # Normalize our loss (if averaged)
        loss.backward()  # Backward pass
        if t == x.shape[0] - 1:  # Wait for several backward steps
            optim.step()  # Now we can do an optimizer step
            optim.zero_grad()  # Reset gradients tensors
            # if (i + 1) % evaluation_steps == 0:  # Evaluate the model when we...
            #     evaluate_model()

            # print("Epoch: %d, No. %d, loss: %1.5f" % (epoch, i, total_loss / x.shape[0]))
    # if epoch % 2 == 0:
    i += 1
    epoch_loss.append(total_loss / x.shape[0])
    scheduler.step()

plt.figure(figsize=(10, 6))
plt.axvline(x=100, c='r', linestyle='--')
plt.plot(epoch_loss, label='Actual Data')
plt.title('Remaining Useful Life Prediction')
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Remaining Useful Life")
plt.show()