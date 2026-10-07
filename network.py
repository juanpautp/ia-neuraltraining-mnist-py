from pathlib import Path

import numpy as np

from mnist import cross_entropy, deriv_sigmoid, one_hot, sigmoid, softmax


def initialize_network(hidden_layers=(10,), seed=0):
    if not 1 <= len(hidden_layers) <= 2 or any(not isinstance(size, int) or size < 1 for size in hidden_layers):
        raise ValueError("Choose one or two hidden layers with positive integer sizes")
    rng = np.random.default_rng(seed)
    sizes = (784, *hidden_layers, 10)
    return tuple(
        (rng.standard_normal((output_size, input_size), dtype=np.float32) * 0.01,
         np.zeros((output_size, 1), dtype=np.float32))
        for input_size, output_size in zip(sizes, sizes[1:])
    )


def architecture(params):
    return [params[0][0].shape[1], *(weights.shape[0] for weights, _ in params)]


def parameter_count(params):
    return sum(weights.size + biases.size for weights, biases in params)


def forward_network(params, X):
    activations = [X]
    for index, (weights, biases) in enumerate(params):
        Z = weights @ activations[-1] + biases
        activations.append(softmax(Z) if index == len(params) - 1 else sigmoid(Z))
    return activations


def backward_network(params, activations, Y):
    delta = activations[-1] - one_hot(Y)
    gradients = [None] * len(params)
    for index in range(len(params) - 1, -1, -1):
        weights, _ = params[index]
        gradients[index] = (delta @ activations[index].T / Y.size, delta.sum(axis=1, keepdims=True) / Y.size)
        if index:
            delta = (weights.T @ delta) * deriv_sigmoid(activations[index])
    return tuple(gradients)


def evaluate_network(params, X, Y):
    probabilities = forward_network(params, X)[-1]
    if not np.isfinite(probabilities).all():
        raise ValueError("Training produced non-finite predictions. Try a lower alpha.")
    return float(np.mean(np.argmax(probabilities, axis=0) == Y)), cross_entropy(probabilities, Y)


def train_network(X, Y, iterations=500, alpha=0.5, seed=0, log_every=1, hidden_layers=(10,),
                  validation=None, validation_every=50, on_progress=None, should_stop=None):
    params = initialize_network(hidden_layers, seed)
    history = []
    for iteration in range(iterations + 1):
        activations = forward_network(params, X)
        probabilities = activations[-1]
        if not np.isfinite(probabilities).all():
            raise ValueError("Training produced non-finite predictions. Try a lower alpha.")
        stopped = should_stop is not None and should_stop()
        final = iteration == iterations or stopped
        validate = validation is not None and (iteration % validation_every == 0 or final)
        if iteration % log_every == 0 or validate or final:
            event = {
                "iteration": iteration,
                "accuracy": float(np.mean(np.argmax(probabilities, axis=0) == Y)),
                "loss": cross_entropy(probabilities, Y),
                "validation_accuracy": None,
                "validation_loss": None,
            }
            if validate:
                event["validation_accuracy"], event["validation_loss"] = evaluate_network(params, *validation)
            history.append(event)
            if on_progress is not None:
                on_progress(dict(event))
        if final:
            break
        gradients = backward_network(params, activations, Y)
        for (weights, biases), (weight_gradient, bias_gradient) in zip(params, gradients):
            weights -= alpha * weight_gradient
            biases -= alpha * bias_gradient
    return params, history


def save_network(path, params):
    values = {"architecture": np.array(architecture(params), dtype=np.int64)}
    for index, (weights, biases) in enumerate(params, start=1):
        values[f"W{index}"] = weights
        values[f"b{index}"] = biases
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def load_network(path):
    with np.load(path, allow_pickle=False) as archive:
        layer_count = sum(key.startswith("W") and key[1:].isdigit() for key in archive.files)
        if layer_count not in (2, 3):
            raise ValueError("Invalid number of network layers")
        params = tuple((archive[f"W{index}"].copy(), archive[f"b{index}"].copy()) for index in range(1, layer_count + 1))
        expected_input = 784
        for weights, biases in params:
            if weights.ndim != 2 or not 1 <= weights.shape[0] <= 256 or weights.shape[1] != expected_input or biases.shape != (weights.shape[0], 1):
                raise ValueError("Invalid model dimensions")
            if not np.isfinite(weights).all() or not np.isfinite(biases).all():
                raise ValueError("Invalid model weights")
            expected_input = weights.shape[0]
        if expected_input != 10:
            raise ValueError("The output layer must contain ten digits")
        if "architecture" in archive and archive["architecture"].tolist() != architecture(params):
            raise ValueError("Model architecture does not match its weights")
    return params
