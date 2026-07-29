import numpy as np


# define a Linear unit class
class Linear:
    def __init__(self, input_layer, output_layer):
        self.weight = np.random.random((output_layer, input_layer))
        self.bias = np.random.random((output_layer, 1))
        pass;

    def forward(self, x):
        assert isinstance(x, np.ndarray) == True
        assert x.shape[0] == self.weight.shape[-1] and len(x.shape) != 1
        return self.weight @ x + self.bias    

    def backward(self, y):
        pass;
        
# define a RELU function        
class ReLU:
    def __init__(self):
        pass;
    def forward(self, x):
        return np.maximum(x, 0)
    def backward(self, y):
        pass;



