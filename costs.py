import numpy as np
import activations

class Cost:
    def __init__(self):
        pass

class CrossEntropy(Cost):
    def __init__(self):
        super().__init__()

    def compute_cost(self, Y_pred, Y, layers, lambd=0.0):
        m = Y.shape[1]

        L2_penalty = 0
        if lambd > 0:
            for fn in layers:
                if isinstance(fn, activations.Linear):
                    L2_penalty += np.sum(np.square(fn.weights))

        L2_penalty = lambd*L2_penalty/(2*m)

        epsilon = 1e-15
        Y_pred_stable = np.clip(Y_pred, epsilon, 1-epsilon)
        log_prob = np.log(Y_pred_stable)
        loss = Y*log_prob
        cost = -np.sum(loss)/m + L2_penalty
        return np.squeeze(cost)

class BinaryCrossEntropy(Cost):
    def __init__(self):
        super().__init__()

    def compute_cost(self, y_pred, y, layers, lambd=0.0):
        m = y.shape[1]

        L2_penalty = 0
        if lambd > 0:
            for fn in layers:
                if isinstance(fn, activations.Linear):
                    L2_penalty += np.sum(np.square(fn.weights))

        L2_penalty = lambd*L2_penalty/(2*m)
    
        epsilon = 1e-15
        y_pred_stable = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.sum(np.log(y_pred_stable)*y + np.log(1-y_pred_stable)*(1-y))/m + L2_penalty
        return np.squeeze(cost)
