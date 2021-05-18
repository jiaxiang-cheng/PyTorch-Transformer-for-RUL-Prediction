from model import *
from loading_data import *
from testing import *
from visualize import *

group, y_test, group_test = loading_FD001()

pred_names = ['id', 'cycle', 'pred', 'ref']
full_predict = pd.DataFrame(columns=pred_names)

num_epochs = 5
d_model = 128
heads = 4
N = 2
m = 14

model = Transformer(m, d_model, N, heads)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = torch.nn.MSELoss()  # mean-squared error for regression

for epoch in range(num_epochs):
    i = 1
    epoch_loss = 0
    model.train()
    while i <= 100:
        x = group.get_group(i).to_numpy()
        total_loss = 0
        optim.zero_grad()
        for t in range(x.shape[0] - 1):
            if t == 0:
                continue
            # elif t == x.shape[0] - 1:
            #     X = np.append(x[t - 1:, 2:-1], [np.zeros(14)], axis=0)
            else:
                X = x[t - 1:t + 2, 2:-1]
            y = x[t, -1:]
            X_train_tensors = Variable(torch.Tensor(X))
            y_train_tensors = Variable(torch.Tensor(y))
            X_train_tensors_final = X_train_tensors.reshape((1, 1, X_train_tensors.shape[0], X_train_tensors.shape[1]))

            # forward pass
            outputs = model.forward(X_train_tensors_final, t)

            # obtain the loss function
            loss = criterion(outputs, y_train_tensors)

            total_loss += loss.item()

            loss = loss / (x.shape[0] - 2)  # Normalize our loss (if averaged)
            loss.backward()  # Backward pass
            if t == x.shape[0] - 2:  # Wait for several backward steps
                optim.step()  # Now we can do an optimizer step
                optim.zero_grad()  # Reset gradients tensors

        i += 1
        epoch_loss += total_loss / x.shape[0]

    model.eval()

    with torch.no_grad():
        rmse, result = testing(group_test, y_test, model)

    print("Epoch: %d, loss: %1.5f, rmse: %1.5f" % (epoch, epoch_loss / 100, rmse))

visualize(result, rmse)
