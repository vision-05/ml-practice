# Candle, my own take on Torch

A place where I implement concepts I am learning in my Machine Learning modules, turning them into a fully fledged python library

Currently with the ability to create multilayered DNNs for classification, to be expanded to more types of ML algorithms in the future

# How to:
There are plans to release this on PyPI in the future as a module, but until more development is done, simply clone this repo and write your main file in the folder.

```python
import nn
import activations as a
import costs as c
import optimisers as o

X = #your data here
Y = #your data here

neural_net = nn.nn()
neural_net.model(
  a.Linear(10, 5),
  a.LeakyReLU(),
  a.Linear(5,3),
  a.LeakyReLU(),
  a.Linear(3,5),
  a.softmax(),
  c.CrossEntropy(),
  o.ADAM()
)

neural_net.train(X_train, Y_train, X_val, Y_val, no_epochs=250, learning_rate=0.001, batch_size=500, lambd=0.2)

Y_pred = neural_net.predict(X)

neural_net.accuracy(Y_pred, Y)
```

It's designed so you can add your cost function and optimiser in the model. Note that you must put the layers in the correct order but you can put the cost and optimiser anywhere you want without affecting the model
