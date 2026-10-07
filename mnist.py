import argparse
import sys
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST


PROJECT_DIR = Path(__file__).resolve().parent


def load_dataset(data_dir):
    dataset = MNIST(root=str(data_dir), train=True, download=True)
    images = dataset.data.numpy()
    labels = dataset.targets.numpy()
    inputs = images.reshape(len(images), 784).T.astype(np.float32) / 255.0
    return inputs[:, :50000], labels[:50000], inputs[:, 50000:], labels[50000:], images, labels


def init_params(seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((10, 784), dtype=np.float32) * 0.01
    b1 = np.zeros((10, 1), dtype=np.float32)
    W2 = rng.standard_normal((10, 10), dtype=np.float32) * 0.01
    b2 = np.zeros((10, 1), dtype=np.float32)
    return W1, b1, W2, b2


def sigmoid(Z):
    expZ = np.exp(-np.abs(Z))
    return np.where(Z >= 0, 1 / (1 + expZ), expZ / (1 + expZ))


def deriv_sigmoid(A):
    return A * (1 - A)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / expZ.sum(axis=0, keepdims=True)


def one_hot(Y):
    encoded = np.zeros((10, Y.size), dtype=np.float32)
    encoded[Y, np.arange(Y.size)] = 1
    return encoded


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    dZ2 = A2 - one_hot(Y)
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = (W2.T @ dZ2) * deriv_sigmoid(A1)
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


def cross_entropy(A2, Y):
    probabilities = A2[Y, np.arange(Y.size)]
    return float(-np.mean(np.log(np.clip(probabilities, np.finfo(A2.dtype).tiny, 1))))


def train_model(X, Y, iterations=500, alpha=0.5, seed=0, log_every=50, on_progress=None, should_stop=None, verbose=True):
    W1, b1, W2, b2 = init_params(seed)
    history = []

    for iteration in range(iterations + 1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        if not np.isfinite(A2).all():
            raise ValueError("Training produced non-finite predictions. Try a lower --alpha.")

        stopped = should_stop is not None and should_stop()
        if iteration % log_every == 0 or iteration == iterations or stopped:
            accuracy = float(np.mean(np.argmax(A2, axis=0) == Y))
            loss = cross_entropy(A2, Y)
            history.append((iteration, accuracy, loss))
            if verbose:
                print(f"Iteration {iteration}/{iterations}: accuracy={accuracy:.2%}, loss={loss:.4f}", flush=True)
            if on_progress is not None:
                on_progress(iteration, accuracy, loss)

        if iteration == iterations or stopped:
            break

        gradients = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, *gradients, alpha)

    return W1, b1, W2, b2, history


def plot_results(params, history, images, labels, image_index, output_dir, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    steps, accuracies, losses = np.array(history).T
    training_fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(steps, accuracies, marker="o")
    axes[0].set(xlabel="Iteration", ylabel="Training accuracy", ylim=(0, 1))
    axes[1].plot(steps, losses, marker="o", color="tab:orange")
    axes[1].set(xlabel="Iteration", ylabel="Cross-entropy loss")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    training_fig.suptitle("MNIST training: 784 → 10 → 10")
    training_fig.savefig(output_dir / "training.png", dpi=160)

    image = images[image_index]
    X = image.reshape(784, 1).astype(np.float32) / 255.0
    _, _, _, probabilities = forward_prop(*params, X)
    prediction = int(np.argmax(probabilities[:, 0]))
    confidence = float(probabilities[prediction, 0])
    prediction_fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    axes[0].imshow(image, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title(f"Label: {labels[image_index]} | Predicted: {prediction}")
    axes[1].bar(range(10), probabilities[:, 0])
    axes[1].set(xticks=range(10), xlabel="Digit", ylabel="Probability", ylim=(0, 1))
    axes[1].set_title(f"Confidence: {confidence:.2%}")
    prediction_fig.savefig(output_dir / "prediction.png", dpi=160)
    print(f"Image {image_index}: label={labels[image_index]}, prediction={prediction}, confidence={confidence:.2%}")
    print(f"Plots saved to {output_dir}")

    if show:
        plt.show()
    plt.close("all")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the professor's NumPy MNIST network without Marimo.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--image-index", type=int, default=50000)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening windows")
    args = parser.parse_args()
    if args.iterations < 1 or args.log_every < 1:
        parser.error("--iterations and --log-every must be positive integers")
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be a positive finite number")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not 0 <= args.image_index < 60000:
        parser.error("--image-index must be between 0 and 59999")
    return args


def main():
    args = parse_args()
    print(f"Python {sys.version.split()[0]} | NumPy {np.__version__} | Training on CPU", flush=True)
    print("Loading MNIST (downloaded only when missing)...", flush=True)
    X_train, Y_train, X_test, Y_test, images, labels = load_dataset(args.data_dir)
    print(f"Training set size: {Y_train.size} | Held-out set size: {Y_test.size}", flush=True)
    print(f"Network: 784 → 10 → 10 | Full-batch gradient descent | Learning rate: {args.alpha}", flush=True)

    W1, b1, W2, b2, history = train_model(
        X_train, Y_train, args.iterations, args.alpha, args.seed, args.log_every
    )
    params = W1, b1, W2, b2
    _, _, _, probabilities = forward_prop(*params, X_test)
    predictions = np.argmax(probabilities, axis=0)
    correct = int(np.sum(predictions == Y_test))
    print(f"Held-out accuracy: {correct / Y_test.size:.2%} ({correct}/{Y_test.size} correct)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "model.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    np.savetxt(
        args.output_dir / "history.csv",
        np.array(history),
        delimiter=",",
        header="iteration,accuracy,loss",
        comments="",
    )
    print(f"Model saved to {args.output_dir / 'model.npz'}", flush=True)
    plot_results(params, history, images, labels, args.image_index, args.output_dir, not args.no_show)


if __name__ == "__main__":
    main()
