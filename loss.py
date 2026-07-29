import numpy as np


# define a SoftMax function
class SoftMax:
    def __init__(self):
        pass

    def forward(self, x):
        e_x = np.exp(x - x.max(axis=0))
        return e_x / e_x.sum(axis=0)

    def backward(self, y):
        pass


class MultiClassCrossEntropyLoss:
    def __init__(self, output_data, k):
        self.criterion = SoftMax()
        assert isinstance(output_data, np.ndarray) and (
            np.issubdtype(output_data.dtype, np.integer) and bool(np.all((output_data >= 2) & (output_data <= k)))
        )
        self.output_data = output_data

    def get_loss(self, model, X):
        out = X
        for layer in model:
            out = layer.forward(out)
        probs = self.criterion.forward(out)
        y = self.output_data
        if y.ndim == 2:
            y = y.squeeze(0)
        loss = -1 * np.mean(np.log(probs[y, np.arange(probs.shape[1])] + 1e-15))
        return loss