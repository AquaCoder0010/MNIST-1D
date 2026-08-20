import numpy as np

class Linear:
    def __init__(self, input_layer, output_layer):
        self.weight = np.random.randn(input_layer, output_layer) * 0.1
        self.bias = np.zeros((1, output_layer))
    def forward(self, x):
        assert isinstance(x, np.ndarray) == True
        assert x.shape[1] == self.weight.shape[0] and len(x.shape) != 1
        self.x_input = x
        return x @ self.weight + self.bias    
    def backward(self, grad, learning_rate):
        grad_out = grad @ self.weight.T
        self.weight -= (self.x_input.T @ grad) * learning_rate
        self.bias -= np.sum(grad, axis=0, keepdims=True) * learning_rate
        return grad_out
        
class ReLU:
    def __init__(self):
        pass;
    def forward(self, x):
        self.x_input = x
        return np.maximum(x, 0)
    def backward(self, gradm, learning_rate):
        return gradm * (self.x_input > 0)



