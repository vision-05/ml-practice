import numpy as np

class Cost:
    def __init__(self):
        pass

class CrossEntropy(Cost):
    def __init__(self):
        super().__init__()

    def compute_cost(self, Y_pred, Y):
        epsilon = 1e-15
        Y_pred_stable = np.clip(Y_pred, epsilon, 1-epsilon)
        log_prob = np.log(Y_pred_stable)
        loss = Y*log_prob
        cost = -np.sum(loss)/Y_pred.shape[1]
        return np.squeeze(cost)

class BinaryCrossEntropy(Cost):
    def __init__(self):
        super().__init__()

    def compute_cost(self, y_pred, y):
        epsilon = 1e-15
        y_pred_stable = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.sum(np.log(y_pred_stable)*y + np.log(1-y_pred_stable)*(1-y))/y_pred.shape[1]
        return np.squeeze(cost)
