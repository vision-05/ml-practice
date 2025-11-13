import numpy as np
import activations

class Optimiser:
    def __init__(self):
        pass

class EWMA(Optimiser):
    def __init__(self):
        super().__init__()

    def _init_state(self, model_obj):
        pass

class SGD(Optimiser):
    def __init__(self):
        super().__init__()

    def run(self, X, Y, batch_size, m, learning_rate, lambd, model_obj):
        
        epoch_cost = 0
        permutation = np.random.permutation(m)

        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        no_batches = m//batch_size

        for j in range(no_batches):
            batch_X = X_shuffled[:,batch_size*j:batch_size*(j+1)]
            batch_Y = Y_shuffled[:,batch_size*j:batch_size*(j+1)]
            batch_Y_pred = model_obj.forward(batch_X)

            cost = model_obj.cost_fn.compute_cost(batch_Y_pred, batch_Y, model_obj.fns, lambd)
            epoch_cost += cost

            model_obj.backward(batch_Y_pred, batch_Y)

            for fn in model_obj.fns:
                if isinstance(fn, activations.Linear):
                    fn.weights -= learning_rate * fn.dW
                    fn.biases -= learning_rate * fn.dB

        return epoch_cost/no_batches

class Momentum(EWMA):
    def __init__(self, beta=0.9):
        super().__init__()
        self.beta = beta
        self.V = None #store our momentum statefully

    def _init_state(self, model_obj):
        self.V = {}
        for i, fn in enumerate(model_obj.fns):
            if isinstance(fn, activations.Linear):
                self.V[f"dW{i}"] = np.zeros_like(fn.weights)
                self.V[f"dB{i}"] = np.zeros_like(fn.biases)

    def run(self, X, Y, batch_size, m, learning_rate, lambd, model_obj):
        epoch_cost = 0
        permutation = np.random.permutation(m)

        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        no_batches = m//batch_size
        for j in range(no_batches):
            batch_X = X_shuffled[:,batch_size*j:batch_size*(j+1)]
            batch_Y = Y_shuffled[:,batch_size*j:batch_size*(j+1)]
            batch_Y_pred = model_obj.forward(batch_X)

            cost = model_obj.cost_fn.compute_cost(batch_Y_pred, batch_Y, model_obj.fns, lambd)
            epoch_cost += cost

            model_obj.backward(batch_Y_pred, batch_Y)

            for i, fn in enumerate(model_obj.fns):
                if isinstance(fn, activations.Linear):
                    dW = fn.dW
                    dB = fn.dB

                    self.V[f"dW{i}"] = self.beta*self.V[f"dW{i}"] + (1-self.beta)*dW
                    self.V[f"dB{i}"] = self.beta*self.V[f"dB{i}"] + (1-self.beta)*dB
                    fn.weights -= learning_rate * self.V[f"dW{i}"]
                    fn.biases -= learning_rate * self.V[f"dB{i}"]
        
        return epoch_cost/no_batches


class RMSProp(EWMA):
    def __init__(self, beta=0.9, epsilon=1e-8):
        super().__init__()
        self.beta = beta
        self.S = None
        self.epsilon = epsilon

    def _init_state(self, model_obj):
        self.S = {}
        for i, fn in enumerate(model_obj.fns):
            if isinstance(fn, activations.Linear):
                self.S[f"dW{i}"] = np.zeros_like(fn.weights)
                self.S[f"dB{i}"] = np.zeros_like(fn.biases)

    def run(self, X, Y, batch_size, m, learning_rate, lambd, model_obj):
        epoch_cost = 0
        permutation = np.random.permutation(m)

        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        no_batches = m//batch_size
        for j in range(no_batches):
            batch_X = X_shuffled[:,batch_size*j:batch_size*(j+1)]
            batch_Y = Y_shuffled[:,batch_size*j:batch_size*(j+1)]
            batch_Y_pred = model_obj.forward(batch_X)

            cost = model_obj.cost_fn.compute_cost(batch_Y_pred, batch_Y, model_obj.fns, lambd)
            epoch_cost += cost

            model_obj.backward(batch_Y_pred, batch_Y)

            for i, fn in enumerate(model_obj.fns):
                if isinstance(fn, activations.Linear):
                    dW = fn.dW
                    dB = fn.dB

                    self.S[f"dW{i}"] = self.beta*self.S[f"dW{i}"] + (1-self.beta)*dW**2
                    self.S[f"dB{i}"] = self.beta*self.S[f"dB{i}"] + (1-self.beta)*dB**2
                    fn.weights -= learning_rate * dW / np.sqrt(self.S[f"dW{i}"] + self.epsilon)
                    fn.biases -= learning_rate * dB / np.sqrt(self.S[f"dB{i}"] + self.epsilon)
        
        return epoch_cost/no_batches

class ADAM(EWMA):
    def __init__(self, beta1=0.999, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.S = None
        self.S_hat = None
        self.M = None
        self.M_hat = None

    def _init_state(self, model_obj):
        self.S = {}
        self.S_hat = {}
        self.M = {}
        self.M_hat = {}

        for i, fn in enumerate(model_obj.fns):
            if isinstance(fn, activations.Linear):
                self.S[f"dW{i}"] = np.zeros_like(fn.weights)
                self.S[f"dB{i}"] = np.zeros_like(fn.biases)
                self.S_hat[f"dW{i}"] = np.zeros_like(fn.weights)
                self.S_hat[f"dB{i}"] = np.zeros_like(fn.biases)
                self.M[f"dW{i}"] = np.zeros_like(fn.weights)
                self.M[f"dB{i}"] = np.zeros_like(fn.biases)
                self.M_hat[f"dW{i}"] = np.zeros_like(fn.weights)
                self.M_hat[f"dB{i}"] = np.zeros_like(fn.biases)
        self.t = 0


    def run(self, X, Y, batch_size, m, learning_rate, lambd, model_obj):
        epoch_cost = 0
        permutation = np.random.permutation(m)

        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        no_batches = m//batch_size
        for j in range(no_batches):
            batch_X = X_shuffled[:,batch_size*j:batch_size*(j+1)]
            batch_Y = Y_shuffled[:,batch_size*j:batch_size*(j+1)]
            batch_Y_pred = model_obj.forward(batch_X)

            cost = model_obj.cost_fn.compute_cost(batch_Y_pred, batch_Y, model_obj.fns, lambd)
            epoch_cost += cost

            model_obj.backward(batch_Y_pred, batch_Y)

            t += 1

            for i, fn in enumerate(model_obj.fns):
                if isinstance(fn, activations.Linear):
                    dW = fn.dW
                    dB = fn.dB

                    self.M[f"dW{i}"] = self.beta1*self.M[f"dW{i}"] + (1-self.beta1)*dW
                    self.M[f"dB{i}"] = self.beta1*self.M[f"dB{i}"] + (1-self.beta1)*dB
                    M_hat_dW = self.M[f"dW{i}"] / (1-self.beta1**self.t)
                    M_hat_dB = self.M[f"dB{i}"] / (1-self.beta1**self.t)

                    self.S[f"dW{i}"] = self.beta2*self.S[f"dW{i}"] + (1-self.beta2)*dW**2
                    self.S[f"dB{i}"] = self.beta2*self.S[f"dB{i}"] + (1-self.beta2)*dB**2
                    S_hat_dW = self.S[f"dW{i}"] / (1-self.beta2**self.t)
                    S_hat_dB = self.S[f"dB{i}"] / (1-self.beta2**self.t)

                    fn.weights -= learning_rate * M_hat_dW / (np.sqrt(S_hat_dW) + self.epsilon)
                    fn.biases -= learning_rate * M_hat_dB / (np.sqrt(S_hat_dB) + self.epsilon)
        
        return epoch_cost/no_batches
