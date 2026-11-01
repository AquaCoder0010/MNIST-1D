# custom implementation
from custom.loss import *
from custom.model import *

# dataset
import mnist1d
from mnist1d.data import make_dataset, get_dataset_args

# pytorch implementation
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    args = mnist1d.data.get_dataset_args()
    args.num_samples = 10000
    args.train_split = 0.8
    args.corr_noise_scale = 0.25
    args.iid_noise_scale = 2e-2
    data = mnist1d.data.get_dataset(
        args, path='./mnist1d_data.pkl', download=False)
    return data


def train_custom_model(x, y, x_test, y_test, epochs=150, batch_size=100, l_r=0.03, hidden=200):
    mnist1d_model = [
        Linear(40, hidden),
        ReLU(),
        Linear(hidden, hidden),
        ReLU(),
        Linear(hidden, 10)
    ]

    criterion = MultiClassCrossEntropyLoss(10)

    loss_per_epoch = []
    test_loss_per_epoch = []

    train_accuracy_per_epoch = []
    test_accuracy_per_epoch = []

    for i in range(epochs):
        # random selection, this is a crude implementation (i.e if the batch size is not divisible it won't work, but it is night time and I am lazy to find another solution sooo)
        indices = np.random.choice(x.shape[0], x.shape[0], replace=False).reshape(-1, batch_size)
        for idx, inds in enumerate(indices):
            x_b, y_b = x[inds], y[inds]
            out = forward(x_b, mnist1d_model)
            probs = criterion.softmax(out)
            criterion.backward(mnist1d_model, x_b, y_b, l_r)

        #
        train_loss = criterion.get_loss(mnist1d_model, x, y)
        test_loss = criterion.get_loss(mnist1d_model, x_test, y_test)
        #
        accuracy_train = np.sum(forward(x, mnist1d_model).argmax(axis=1) == y) / len(y)
        accuracy_test = np.sum(forward(x_test, mnist1d_model).argmax(axis=1) == y_test) / len(y_test)

        loss_per_epoch.append(train_loss)
        test_loss_per_epoch.append(test_loss)
        train_accuracy_per_epoch.append(accuracy_train)
        test_accuracy_per_epoch.append(accuracy_test)

        print("="*10)
        print(f"Finished training epoch : {i+1}")
        print(f"Train loss: {train_loss}, Test loss: {test_loss}, Train Accuracy: {accuracy_train},  Test Accuracy: {accuracy_test}")

    return loss_per_epoch, test_loss_per_epoch, train_accuracy_per_epoch, test_accuracy_per_epoch

DEVICE = "cpu"
def train_cnn(x, y, x_test, y_test, epochs=150, batch_size=100, l_r=0.03, momentum=0.9):
    model = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=15, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(60, 10)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=l_r, momentum=momentum)

    # reshape to (N, 1, 40): Conv1d expects a channel dimension
    train_dataset = TensorDataset(
        torch.from_numpy(x).float().unsqueeze(1),
        torch.from_numpy(y).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x_test_t = torch.from_numpy(x_test).float().unsqueeze(1)
    y_test_t = torch.from_numpy(y_test).long()

    def full_eval(x_t, y_t):
        model.eval()
        with torch.no_grad():
            out = model(x_t)
            loss = criterion(out, y_t).item()
            accuracy = (out.argmax(dim=1) == y_t).float().mean().item()
        model.train()
        return loss, accuracy

    loss_per_epoch = []
    test_loss_per_epoch = []

    train_accuracy_per_epoch = []
    test_accuracy_per_epoch = []

    for i in range(epochs):
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()

        train_loss, accuracy_train = full_eval(
            torch.from_numpy(x).float().unsqueeze(1), torch.from_numpy(y).long())
        test_loss, accuracy_test = full_eval(x_test_t, y_test_t)

        loss_per_epoch.append(train_loss)
        test_loss_per_epoch.append(test_loss)
        train_accuracy_per_epoch.append(accuracy_train)
        test_accuracy_per_epoch.append(accuracy_test)

        print("="*10)
        print(f"Finished training epoch : {i+1}")
        print(f"Train loss: {train_loss}, Test loss: {test_loss}, Train Accuracy: {accuracy_train},  Test Accuracy: {accuracy_test}")

    return loss_per_epoch, test_loss_per_epoch, train_accuracy_per_epoch, test_accuracy_per_epoch


data = get_data()
x, y = data['x'], data['y']
x_test, y_test = data['x_test'], data['y_test']

loss_per_epoch, test_loss_per_epoch, train_accuracy_per_epoch, test_accuracy_per_epoch = train_custom_model(x, y, x_test, y_test)
cnn_loss_per_epoch, cnn_test_loss_per_epoch, cnn_train_accuracy_per_epoch, cnn_test_accuracy_per_epoch = train_cnn(x, y, x_test, y_test)

## loss on the left, accuracy on the right, custom model vs CNN
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs_axis = np.arange(1, len(loss_per_epoch) + 1)

# color = model, linestyle = train/test
ax1.plot(epochs_axis, loss_per_epoch,                  color="tab:blue",   label="custom (train)")
ax1.plot(epochs_axis, test_loss_per_epoch,             color="tab:blue",   linestyle="--", label="custom (test)")
ax1.plot(epochs_axis, cnn_loss_per_epoch,              color="tab:orange", label="CNN (train)")
ax1.plot(epochs_axis, cnn_test_loss_per_epoch,         color="tab:orange", linestyle="--", label="CNN (test)")
ax1.set_title("loss per epoch")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()

ax2.plot(epochs_axis, train_accuracy_per_epoch,        color="tab:blue",   label="custom (train)")
ax2.plot(epochs_axis, test_accuracy_per_epoch,         color="tab:blue",   linestyle="--", label="custom (test)")
ax2.plot(epochs_axis, cnn_train_accuracy_per_epoch,    color="tab:orange", label="CNN (train)")
ax2.plot(epochs_axis, cnn_test_accuracy_per_epoch,     color="tab:orange", linestyle="--", label="CNN (test)")
ax2.set_title("accuracy per epoch")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)  # save BEFORE show: show() clears the figure after the window closes
plt.show()

##
