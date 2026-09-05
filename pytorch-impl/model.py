import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import mnist1d

DEVICE = "cpu"
class MNISTNN(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(40, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
           nn.Linear(hidden_size, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

def get_data():
    args = mnist1d.data.get_dataset_args()
    args.num_samples = 8000
    args.train_split = 0.5
    args.corr_noise_scale = 0.25
    args.iid_noise_scale = 2e-2
    data = mnist1d.data.get_dataset(
        args, path='./mnist1d_data.pkl', download=True)

def fit_model(hidden_size, lr, momentum, batch_size, n_epoch, dropout=False):
    global data
    model = None
    if dropout:
        model = MNISTNN(hidden_size=hidden_size).to(DEVICE)
    else:
        model = MNISTNND(hidden_size=hidden_size).to(DEVICE)
    
    assert model != None
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    x_train = torch.tensor(data['x'].astype('float32'))
    y_train = torch.tensor(data['y'].transpose().astype('long'))
    x_test = torch.tensor(data['x_test'].astype('float32'))
    y_test = torch.tensor(data['y_test'].astype('long'))

    data_loader = DataLoader(TensorDataset(
        x_train, y_train), batch_size=batch_size, shuffle=True, worker_init_fn=np.random.seed(1))

    info_list = []
    for epoch in range(n_epoch):
        # loop over batches
        for i, batch in enumerate(data_loader):
            # retrieve inputs and labels for this batch
            x_batch, y_batch = batch
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass -- calculate model output
            pred = model(x_batch)
            # compute the loss
            loss = loss_function(pred, y_batch)
            # backward pass
            loss.backward()
            # SGD update
            optimizer.step()
    
        # Run whole dataset to get statistics -- normally wouldn't do this
        pred_train = model(x_train)
        pred_test = model(x_test)
        _, predicted_train_class = torch.max(pred_train.data, 1)
        _, predicted_test_class = torch.max(pred_test.data, 1)
        errors_train = 100 - 100 * \
            (predicted_train_class == y_train).float().sum() / len(y_train)
        errors_test = 100 - 100 * \
            (predicted_test_class == y_test).float().sum() / len(y_test)
        losses_train = loss_function(pred_train, y_train).item()
        losses_test = loss_function(pred_test, y_test).item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch:5d}, train loss {losses_train:.6f}, train error {errors_train:3.2f},  test loss {losses_test:.6f}, test error {errors_test:3.2f}')
        info = {"curr_epoch": epoch, "errors_train": errors_train, "errors_test": errors_test, "losses_train": losses_train, "losses_test": losses_test}
        info_list.append(info)
    
    return info_list




