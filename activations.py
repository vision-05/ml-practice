import numpy as np

class Activation:
    def __init__(self):
        pass

    def forward(self, Z):
        pass

    def backward(self, dA, Z, **kwargs):
        pass

class Linear(Activation):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.bias_pred = True
        self.weights = np.random.randn(out_features, in_features) * np.sqrt(2 / in_features)
        self.biases = np.ones((out_features, 1)) * 0.01

        self.A_prev = None
        self.dW = None
        self.dB = None

    def forward(self, inputs):
        self.A_prev = inputs
        if self.bias_pred == False:
            self.bias = 0

        return np.dot(self.weights, inputs) + self.biases
    
    def predict(self, inputs):
        if self.bias_pred == False:
            self.bias = 0

        return np.dot(self.weights, inputs) + self.biases
    
    def backward(self, dZ, **kwargs):
        m = self.A_prev.shape[1]

        lambd = kwargs.get("lambd", 0.0)

        L2_grad = lambd*self.weights/m

        self.dW = np.dot(dZ, self.A_prev.T)/m + L2_grad
        self.dB = np.sum(dZ, axis=1, keepdims=True)/m

        new_grad = np.dot(self.weights.T, dZ)
        return new_grad

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0,Z)
    
    def predict(self, Z):
        return np.maximum(0,Z)

    def backward(self, dZ, **kwargs):
        new_grad = dZ*(self.Z>0)
        return new_grad
    
class LeakyReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.Z = Z
        return np.maximum(Z*0.01, Z)
    
    def predict(self, Z):
        return np.maximum(Z*0.01, Z)
    
    def backward(self, dZ, **kwargs):
        new_grad = dZ*(self.Z > 0) + dZ*(self.Z<=0)*0.01
        return new_grad

class sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.Z = Z
        return 1/(1+np.exp(-Z))
    
    def predict(self, Z):
        return 1/(1+np.exp(-Z))

    def backward(self, dZ, **kwargs):
        s = sigmoid(self.Z)
        return dZ*s*(1-s)

class softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.Z = Z
        stableZ = Z - np.max(Z, axis=0, keepdims=True)
        expZ = np.exp(stableZ)
        denominator = np.sum(expZ, axis=0, keepdims=True)
        return expZ/denominator
    
    def predict(self, Z):
        stableZ = Z - np.max(Z, axis=0, keepdims=True)
        expZ = np.exp(stableZ)
        denominator = np.sum(expZ, axis=0, keepdims=True)
        return expZ/denominator

    def backward(self, dZ, **kwargs):
        pass