import matplotlib.pyplot as plt
from loss import *
from model import *

from mnist1d.data import make_dataset, get_dataset_args

def forward(x, model_layers):
    assert all(isinstance(layer, (Linear, ReLU)) for layer in model_layers)
    out = x
    for layer in model_layers:
        out = layer.forward(out)
    return out

# the data 
defaults = get_dataset_args()
data = make_dataset(defaults)
x, y, t = data['x'], data['y'], data['t']

mnist1d_model = [
    Linear(40, 100), 
    ReLU(), 
    Linear(100, 100), 
    ReLU(),
    Linear(100, 10)
]

out = forward(x, mnist1d_model)
criterion = MultiClassCrossEntropyLoss(10)
loss = criterion.get_loss(mnist1d_model, x, y)

epochs = 10
batch_size = 16
l_r = 0.1
loss_per_step = []
for i in range(epochs):
    # random selection, this is a crude implementation (i.e if the batch size is not divisible it won't work, but it is night time and I am lazy to find another solution sooo)
    indices = np.random.choice(x.shape[0], x.shape[0], replace=False).reshape(-1, batch_size)
    for idx, inds in enumerate(indices):
        x_b, y_b = x[inds], y[inds]

        out = forward(x_b, mnist1d_model)
        probs = criterion.softmax(out)
        criterion.backward(mnist1d_model, x_b, y_b, l_r)
        if idx % 10 == 0:
            loss = criterion.get_loss(mnist1d_model, x_b, y_b)
            loss_per_step.append(loss)
            print(f"AHH -> {len(loss_per_step)}")

            print(f"Batch idx : {idx}, Loss: {loss}")


    loss = criterion.get_loss(mnist1d_model, x, y)
    print("="*10)
    print(f"Finished training epoch : {i+1}")
    print(f"Current Loss: {loss}")



print(len(loss_per_step))
plt.plot(loss_per_step)
plt.xlabel("training step (logged every 10 batches)")
plt.ylabel("cross-entropy loss")
plt.title("MNIST-1D training loss")
plt.grid(alpha=0.3)
plt.savefig("Implementation-details/loss_curve.png")
plt.show()
