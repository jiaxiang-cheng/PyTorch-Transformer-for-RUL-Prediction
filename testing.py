import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable


def testing(group_test, y_test, model):
    rmse = 0
    j = 1
    result = []

    while j <= 100:
        x_test = group_test.get_group(j).to_numpy()
        data_predict = 0
        for t in range(x_test.shape[0]): # iterate to the end of each sequence
            if t == 0:
                continue
            elif t == x_test.shape[0] - 1: # for last one row append a zero padding
                X_test = np.append(x_test[t - 1:, 2:], [np.zeros(14)], axis=0)
            else:
                X_test = x_test[t - 1:t + 2, 2:]

            X_test_tensors = Variable(torch.Tensor(X_test))

            X_test_tensors_final = X_test_tensors.reshape((1, 1, X_test_tensors.shape[0], X_test_tensors.shape[1]))

            test_predict = model.forward(X_test_tensors_final, t)
            data_predict = test_predict.data.numpy()[-1]
            
            # block for linearily decreasing the RUL after each iteration
            if data_predict - 1 < 0:
                data_predict = 0
            else:
                data_predict -= 1

        result.append(data_predict)
        rmse += np.power((data_predict - y_test.to_numpy()[j - 1]), 2)
        j += 1

    rmse = np.sqrt(rmse / 100)

    result = y_test.join(pd.DataFrame(result))
    result = result.sort_values('RUL', ascending=False)

    return rmse, result
