full_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True
)
all_X = full_dataset.data.numpy().reshape(60000, 784).T / 255.0
all_Y = full_dataset.targets.numpy()

X_train = all_X[:, :50000]
Y_train = all_Y[:50000]

X_test = all_X[:, 50000:]
Y_test = all_Y[50000:]

mo.md(f"Training set size: {X_train.shape[1]} | Test set size: {X_test.shape[1]}")

def init_params():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

    def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def deriv_sigmoid(A):
    return A * (1 - A)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def one_hot(Y):
    oh = np.zeros((Y.size, 10))
    oh[np.arange(Y.size), Y] = 1
    return oh.T

    def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

    def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    Y_oh = one_hot(Y)

    dZ2 = A2 - Y_oh
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_sigmoid(A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

    def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

    def train_model(X, Y, iterations=500, alpha=0.1):
    W1, b1, W2, b2 = init_params()
    history = []

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 50 == 0:
            predictions = np.argmax(A2, axis=0)
            accuracy = np.sum(predictions == Y) / Y.size
            history.append(accuracy)
            print(f"Iteration {i}: Accuracy {accuracy:.4f}")

    return W1, b1, W2, b2, history

    params = train_model(X_train, Y_train, iterations=500, alpha=0.5)
W1, b1, W2, b2, stats = params

_, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
test_predictions = np.argmax(A2_test, axis=0)
test_accuracy = np.sum(test_predictions == Y_test) / Y_test.size
mo.md(f"**Test accuracy: {test_accuracy:.4f}** ({int(test_accuracy * len(Y_test))}/{len(Y_test)} correct)")

fig, ax = plt.subplots()
ax.plot(range(0, len(stats) * 50, 50), stats, marker='o')
ax.set_xlabel("Iteration")
ax.set_ylabel("Training accuracy")
ax.set_title("Training accuracy over time")
ax.set_ylim(0, 1)
ax.grid(True)
mo.as_html(fig)

import marimo as mo
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"