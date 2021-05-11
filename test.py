import copy
import math
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import time

seaborn.set_context(context="talk")


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, m, t):
        super().__init__()
        self.m = m
        self.t = t

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(1, m)
        for i in range(0, m, 2):
            pe[1, i] = \
                math.sin(t / (10000 ** ((2 * i) / m)))
            pe[1, i + 1] = \
                math.cos(t / (10000 ** ((2 * (i + 1)) / m)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.m)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


batch = next(iter(train_iter))
input_seq = batch.English.transpose(0, 1)
input_pad = EN_TEXT.vocab.stoi['<pad>']
# creates mask with 0s wherever there is padding in the input
input_msk = (input_seq != input_pad).unsqueeze(1)

# create mask as before
target_seq = batch.French.transpose(0, 1)
target_pad = FR_TEXT.vocab.stoi['<pad>']
target_msk = (target_seq != target_pad).unsqueeze(1)

size = target_seq.size(1)  # get seq_len for matrix

nopeak_mask = np.triu(np.ones(1, size, size),
                      k=1).astype('uint8')
nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)

target_msk = target_msk & nopeak_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
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
    scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
        self.linear_reg = nn.Linear(d_model, 1)

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        output = torch.sigmoid(self.linear_reg(norm))
        return output


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, heads, dropout=0.1):
#         super().__init__()
#         self.norm_1 = Norm(d_model)
#         self.norm_2 = Norm(d_model)
#         self.norm_3 = Norm(d_model)
#
#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)
#         self.dropout_3 = nn.Dropout(dropout)
#
#         self.attn_1 = MultiHeadAttention(heads, d_model)
#         self.attn_2 = MultiHeadAttention(heads, d_model)
#         self.ff = FeedForward(d_model).cuda()
#
#     def forward(self, x, e_outputs, src_mask, trg_mask):
#         x2 = self.norm_1(x)
#         x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
#         x2 = self.norm_2(x)
#         x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
#                                            src_mask))
#         x2 = self.norm_3(x)
#         x = x + self.dropout_3(self.ff(x2))
#         return x


# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, t, d_model, N, heads, m=14):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(m, t)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        # x = self.embed(src)
        x = self.pe(src)
        for i in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)


# class Decoder(nn.Module):
#     def __init__(self, vocab_size, d_model, N, heads):
#         super().__init__()
#         self.N = N
#         self.embed = Embedder(vocab_size, d_model)
#         self.pe = PositionalEncoder(d_model)
#         self.layers = get_clones(DecoderLayer(d_model, heads), N)
#         self.norm = Norm(d_model)
#
#     def forward(self, trg, e_outputs, src_mask, trg_mask):
#         x = self.embed(trg)
#         x = self.pe(x)
#         for i in range(self.N):
#             x = self.layers[i](x, e_outputs, src_mask, trg_mask)
#         return self.norm(x)


class Gating(nn.Module):
    def __init__(self, d_model, m=14):
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
        self.W_e = nn.Parameter(torch.Tensor(d_model, m))
        self.b_e = nn.Parameter(torch.Tensor(d_model))

        self.init_weights()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 1, kernel_size=(3, 1), stride=1, padding=1),
        )

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x_i = x[, 1:2]
        h_i = self.cnn_layers(x)

        r_i = torch.sigmoid(self.W_r @ h_i + self.V_r @ x_i + self.b_r)
        u_i = torch.sigmoid(self.W_u @ h_i + self.V_u @ x_i + self.b_u)

        # the output of the gating mechanism
        hh_i = np.multiply(h_i, u_i) + np.multiply(x_i, r_i)

        # linear mapping to encoder layer
        output = self.W_e @ hh_i + self.b_e

        return output


class Transformer(nn.Module):
    def __init__(self, m, t, d_model, N, heads):
        super().__init__()
        self.gating = Gating(d_model, m)
        self.encoder = Encoder(t, d_model, N, heads)
        # self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, 1)

    def forward(self, src, trg, src_mask, trg_mask):
        e_i = self.gating(src)
        e_outputs = self.encoder(e_i, src_mask)
        # d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(e_outputs)
        return output


# we don't perform softmax on the output as this will be handled
# automatically by our loss function


d_model = 128
heads = 4
N = 2

# src_vocab = len(EN_TEXT.vocab)
# trg_vocab = len(FR_TEXT.vocab)
model = Transformer(src_vocab, trg_vocab, d_model, N, heads)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# this code is very important! It initialises the parameters with a
# range of values that stops the signal fading or getting too big.
# See this blog for a mathematical explanation.
optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

criterion = torch.nn.MSELoss()  # mean-squared error for regression


def train_model(epochs, print_every=100):
    model.train()

    start = time.time()
    temp = start

    total_loss = 0

    for epoch in range(epochs):

        for i, batch in enumerate(train_iter):
            # src = batch.English.transpose(0, 1)
            # trg = batch.French.transpose(0, 1)
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next

            # trg_input = trg[:, :-1]
            #
            # # the words we are trying to predict
            #
            # targets = trg[:, 1:].contiguous().view(-1)

            # create function to make masks using mask code above

            # src_mask, trg_mask = create_masks(src, trg_input)

            # preds = model(src, trg_input, src_mask, trg_mask)
            preds = model.forward(i, src)

            # forward pass
            # outputs = lstm1.forward(X_train_tensors_final)
            # calculate the gradient, manually setting to 0
            optim.zero_grad()

            # obtain the loss function
            loss = criterion(preds, y_train_tensors)

            # calculates the loss of the loss function
            loss.backward()

            # improve from loss, i.e back propagation
            optim.step()

            i += 1
            # if (i + 1) % print_every == 0:
            #     loss_avg = total_loss / print_every
            #     print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % (
            #         (time.time() - start) // 60, epoch + 1, i + 1, loss_avg, time.time() - temp, print_every))
            #     total_loss = 0
            #     temp = time.time()
            if epoch % 2 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
