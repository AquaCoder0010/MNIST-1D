import numpy as np


# define a SoftMax function
class SoftMax:
    def __init__(self):
        pass

    def forward(self, x):
        e_x = np.exp(x - x.max(axis=1, keepdims=True))
        self.output = e_x / e_x.sum(axis=1, keepdims=True)
        return self.output

    def backward(self, grad):
        p = self.output
        return p * (grad - np.sum(grad * p, axis=1, keepdims=True))


class MultiClassCrossEntropyLoss:
    def __init__(self, k):
        self.criterion = SoftMax()
        self.k = k


    def get_loss(self, model, X, y):
        assert isinstance(y, np.ndarray) and (
            np.issubdtype(y.dtype, np.integer) and bool(np.all((y >= 0) & (y < self.k)))
        )
        out = X
        for layer in model:
            out = layer.forward(out)
        probs = self.criterion.forward(out)
        loss = -1 * np.mean(np.log(probs[np.arange(probs.shape[0]), y] + 1e-15))
        return loss

    def backward(self, model, X, y):
        out = X
        for layer in model:
            out = layer.forward(out)
        probs = self.criterion.forward(out)
        N = X.shape[0]
        grad = probs.copy()
        grad[np.arange(N), y] -= 1
        grad /= N
        for layer in reversed(model):
            grad = layer.backward(grad)
        return grad