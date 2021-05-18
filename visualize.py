import matplotlib.pyplot as plt


def visualize(result, rmse):

    # the true remaining useful life of the testing samples
    true_rul = result.iloc[:, 0:1].to_numpy()
    # the predicted remaining useful life of the testing samples
    pred_rul = result.iloc[:, 1:].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.axvline(x=100, c='r', linestyle='--')
    plt.plot(true_rul, label='Actual Data')
    plt.plot(pred_rul, label='Predicted Data')
    plt.title('RUL Prediction on CMAPSS Data')
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")
    plt.savefig('Transformer({}).png'.format(rmse))
    plt.show()