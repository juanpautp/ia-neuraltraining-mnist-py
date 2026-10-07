import tempfile
import unittest
from pathlib import Path

import numpy as np

from mnist import cross_entropy, forward_prop, train_model
from network import architecture, backward_network, forward_network, initialize_network, load_network, save_network, train_network


class NetworkTests(unittest.TestCase):
    def test_matches_professor_network(self):
        rng = np.random.default_rng(8)
        X = rng.random((784, 5), dtype=np.float32)
        Y = np.array([0, 1, 4, 6, 8])
        original = train_model(X, Y, iterations=3, alpha=0.5, log_every=1, verbose=False)
        params, history = train_network(X, Y, iterations=3, alpha=0.5)
        for expected, actual in zip(original[:4], [values for layer in params for values in layer]):
            np.testing.assert_allclose(actual, expected, atol=1e-8)
        np.testing.assert_allclose(forward_network(params, X)[-1], forward_prop(*original[:4], X)[-1], atol=1e-8)
        self.assertEqual([(event["iteration"], event["accuracy"], event["loss"]) for event in history], original[-1])

    def test_gradients_for_one_and_two_hidden_layers(self):
        rng = np.random.default_rng(12)
        X = rng.normal(0, 0.1, (784, 3))
        Y = np.array([2, 5, 7])
        for hidden_layers in ((4,), (4, 3)):
            with self.subTest(hidden_layers=hidden_layers):
                params = tuple((weights.astype(np.float64) * 20, biases.astype(np.float64)) for weights, biases in initialize_network(hidden_layers, 6))
                gradients = backward_network(params, forward_network(params, X), Y)
                for layer, gradient_layer in zip(params, gradients):
                    for parameter, gradient in zip(layer, gradient_layer):
                        for index in (tuple(0 for _ in parameter.shape), tuple(size - 1 for size in parameter.shape)):
                            original = parameter[index]
                            parameter[index] = original + 1e-5
                            upper = cross_entropy(forward_network(params, X)[-1], Y)
                            parameter[index] = original - 1e-5
                            lower = cross_entropy(forward_network(params, X)[-1], Y)
                            parameter[index] = original
                            np.testing.assert_allclose(gradient[index], (upper - lower) / 2e-5, rtol=1e-4, atol=1e-8)

    def test_variable_model_round_trip_and_legacy_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            params = initialize_network((32, 16), seed=4)
            save_network(path, params)
            recovered = load_network(path)
            self.assertEqual(architecture(recovered), [784, 32, 16, 10])
            for layer, saved_layer in zip(params, recovered):
                for values, saved_values in zip(layer, saved_layer):
                    np.testing.assert_array_equal(values, saved_values)
            params = initialize_network((10,))
            np.savez(path, W1=params[0][0], b1=params[0][1], W2=params[1][0], b2=params[1][1])
            self.assertEqual(architecture(load_network(path)), [784, 10, 10])

    def test_validation_does_not_modify_training(self):
        rng = np.random.default_rng(4)
        X = rng.random((784, 5), dtype=np.float32)
        Y = np.array([0, 1, 4, 6, 8])
        plain, _ = train_network(X, Y, iterations=3, hidden_layers=(4, 3))
        validated, history = train_network(X, Y, iterations=3, hidden_layers=(4, 3), validation=(X, Y), validation_every=2)
        self.assertEqual([event["iteration"] for event in history if event["validation_accuracy"] is not None], [0, 2, 3])
        for layer, saved_layer in zip(plain, validated):
            for values, saved_values in zip(layer, saved_layer):
                np.testing.assert_array_equal(values, saved_values)


if __name__ == "__main__":
    unittest.main()
