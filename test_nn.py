import nn
import activations as a
import costs as c
import sklearn.datasets
import numpy as np

mnist = sklearn.datasets.fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

m = 70000
X_full = mnist.data[:m]
Y_raw = mnist.target[:m].astype(int)

def one_hot_encoder(Y_raw, C):
    m = Y_raw.shape[0]
    Y_one_hot = np.zeros((C,m))
    Y_one_hot[Y_raw,np.arange(m)] = 1
    return Y_one_hot

m_full = X_full.shape[0]
shuffled_indices = np.random.permutation(m_full)

X_shuffled = X_full[shuffled_indices]
Y_shuffled_raw = Y_raw[shuffled_indices]

m_train = 50000
m_val = 10000
m_test = 10000

X_train = (X_shuffled[:m_train].T)/255.
Y_train_raw = Y_shuffled_raw[:m_train]

X_val = (X_shuffled[m_train:m_train+m_val].T)/255.
Y_val_raw = Y_shuffled_raw[m_train:m_train+m_val]

X_test = (X_shuffled[m_train+m_val:].T)/255.
Y_test_raw = Y_shuffled_raw[m_train+m_val:]


Y_train = one_hot_encoder(Y_train_raw, 10)
Y_val = one_hot_encoder(Y_val_raw, 10)
Y_test = one_hot_encoder(Y_test_raw, 10)

mod = nn.nn()
mod.model(
    a.Linear(784,128),
    a.LeakyReLU(),
    a.Linear(128,64),
    a.LeakyReLU(),
    a.Linear(64,10),
    a.softmax(),
    c.CrossEntropy()
)
mod.train(X_train, Y_train, X_val, Y_val, 250, 0.07)
Y_test_pred = mod.predict(X_test)
final_acc = mod.accuracy(Y_test_pred, Y_test)

print(f"Final accuracy {final_acc*100:.2f}%")

def test_dimensions():
    assert mod.fns[0].weights.shape[0] == 128 and mod.fns[0].weights.shape[1] == 784
    assert mod.fns[2].weights.shape[0] == 64 and mod.fns[2].weights.shape[1] == 128
    assert mod.fns[4].biases.shape[0] == 10


