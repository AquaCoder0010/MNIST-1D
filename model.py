import numpy as np


# define a Linear unit class
class Linear:
    def __init__(self, input_layer, output_layer):
        self.weight = np.random.random((input_layer, output_layer))
        self.bias = np.random.random((1, output_layer))
        pass;

    def forward(self, x):
        assert isinstance(x, np.ndarray) == True
        assert x.shape[1] == self.weight.shape[-1] and len(x.shape) != 1
        self.x_input = x
        return x @ self.weight + self.bias    

    def backward(self, grad):
        self.dW = grad.T @ self.x_input
        self.db = np.sum(grad, axis=0, keepdims=True).T
        return grad @ self.weight
        
# define a RELU function        
class ReLU:
    def __init__(self):
        pass;
    def forward(self, x):
        self.x_input = x
        return np.maximum(x, 0)
    def backward(self, grad):
        return grad * (self.x_input > 0)



