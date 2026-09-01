import numpy as np

class MultiClassCrossEntropyLoss:
    def __init__(self, k):
        self.k = k
    def softmax(self, x):
        e_x = np.exp(x - x.max(axis=1, keepdims=True))
        output = e_x / e_x.sum(axis=1, keepdims=True)
        return output
    def get_loss(self, model, X, y):
        assert isinstance(y, np.ndarray) and (
            np.issubdtype(y.dtype, np.integer) and bool(np.all((y >= 0) & (y < self.k)))
        )
        out = X
        for layer in model:
            out = layer.forward(out)
        probs = self.softmax(out)
        loss = -1 * np.mean(np.log(probs[np.arange(probs.shape[0]), y] + 1e-15))
        return loss
    
    def backward(self, model, X, y, learning_rate=0.3):
        out = X
        for layer in model:
            out = layer.forward(out)
        probs = self.softmax(out)
        N = X.shape[0]
        grad = probs.copy()
        grad[np.arange(N), y] -= 1
        grad /= N
        for layer in reversed(model):
            grad = layer.backward(grad, learning_rate)
        return grad