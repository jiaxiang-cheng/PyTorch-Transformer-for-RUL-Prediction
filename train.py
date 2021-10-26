import torch
torch.manual_seed(1)
from model import *
from loading_data import *
from testing import *
from visualize import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse




def training():
    for epoch in range(num_epochs):  # iteration of epoch
        i = 1
        epoch_loss = 0

        # training step
        model.train()

        while i <= 100:  # iteration of unit

            # fetch the data of unit i
            x = group.get_group(i).to_numpy()
            total_loss = 0
            optim.zero_grad()

            for t in range(x.shape[0] - 1):
                if t == 0:  # skip the first and last for convolution without padding
                    continue
                else:
                    X = x[t - 1:t + 2, 2:-1]  # fetch the 3 * 14 feature as input

                y = x[t, -1:]  # fetch the corresponding target rul as label

                X_train_tensors = Variable(torch.Tensor(X))
                y_train_tensors = Variable(torch.Tensor(y))
                
                X_train_tensors_final = X_train_tensors.reshape(
                    (1, 1, X_train_tensors.shape[0], X_train_tensors.shape[1]))
  

                # forward pass
                outputs = model.forward(X_train_tensors_final, t)


                # obtain the loss function
                loss = criterion(outputs, y_train_tensors)

                # summarize the loss
                total_loss += loss.item()

                loss = loss / (x.shape[0] - 2)  # normalize the loss
                loss.backward()  # backward pass

                # only update after finishing one unit
                if t == x.shape[0] - 2:  # Wait for several backward steps
                    optim.step()  # Now we can do an optimizer step
                    optim.zero_grad()  # Reset gradients tensors

            i += 1
            epoch_loss += total_loss / x.shape[0]

        # evaluate model
        model.eval()

        with torch.no_grad():
            rmse, result = testing(group_test, y_test, model)

        print("Epoch: %d, training loss: %1.5f, testing rmse: %1.5f" % (epoch, epoch_loss / 100, rmse))

    return result, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD001', help='which dataset to run')
    opt = parser.parse_args()
    
    num_epochs = 4 # Number of training epochs
    d_model = 128  # dimension in encoder
    heads = 4  # number of heads in multi-head attention
    N = 2  # number of encoder layers
    m = 14  # number of features


    
    if opt.dataset == 'FD001':
        # loading training and testing sets
        group, y_test, group_test = loading_FD001()
        
        #setting the drooput rate
        dropout = 0.1
        
        # define and load model
        model = Transformer(m, d_model, N, heads, dropout)

        # initialization
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # initialize Adam optimizer
        optim = torch.optim.Adam(model.parameters(), lr=0.001)

        # mean-squared error for regression
        criterion = torch.nn.MSELoss()

        # training with evaluation
        result, rmse = training()

        # testing already done in training() for each epoch to see live testing rmse, or
        # can be done once after finish training
        # model.eval()
        # with torch.no_grad():
        #     rmse, result = testing(group_test, y_test, model)

        # visualize the testing result
        visualize(result, rmse)
    else:
        print('Either dataset not implemented or not defined')

