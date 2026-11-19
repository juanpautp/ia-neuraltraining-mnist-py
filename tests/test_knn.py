import unittest
from collections import Counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from comparison import compare_models
from knn import EvaluationCancelled, KNNClassifier, classification_metrics, evaluate_knn
from network import initialize_network


class KNNTests(unittest.TestCase):
    def test_matches_professor_distance_and_counter_vote(self):
        rng = np.random.default_rng(14)
        X = rng.normal(size=(40, 7))
        y = rng.integers(0, 4, size=40)
        queries = rng.normal(size=(9, 7))
        for k in (1, 3, 8):
            model = KNNClassifier(k).fit(X, y)
            expected = []
            for query in queries:
                distances = [np.sqrt(np.sum((query - reference) ** 2)) for reference in X]
                indices = np.argsort(distances)[:k]
                expected.append(Counter(y[indices]).most_common(1)[0][0])
            np.testing.assert_array_equal(model.predict(queries, batch_size=3), expected)

    def test_vote_and_distance_ties_are_deterministic(self):
        X = np.array([[0], [1], [-1], [2]], dtype=np.float32)
        y = np.array([8, 3, 4, 3])
        model = KNNClassifier(2).fit(X, y)
        distances, indices = model.kneighbors([[0]], 3)
        np.testing.assert_array_equal(indices, [[0, 1, 2]])
        np.testing.assert_allclose(distances, [[0, 1, 1]])
        self.assertEqual(model.predict_one([0]), 8)
        np.testing.assert_allclose(model.predict_proba([[0]])[0], [.5, 0, .5])

    def test_iris_scaling_and_predictions_match_sklearn(self):
        iris = load_iris()
        X_train, X_validation, y_train, _ = train_test_split(iris.data, iris.target, random_state=42, test_size=.2, stratify=iris.target)
        scaler = StandardScaler().fit(X_train)
        expected = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(X_train), y_train)
        model = KNNClassifier(5, standardize=True).fit(X_train, y_train)
        np.testing.assert_allclose(model.mean, scaler.mean_, rtol=1e-5)
        np.testing.assert_allclose(model.scale, scaler.scale_, rtol=1e-5)
        np.testing.assert_array_equal(model.predict(X_validation), expected.predict(scaler.transform(X_validation)))
        np.testing.assert_allclose(model.predict_proba(X_validation), expected.predict_proba(scaler.transform(X_validation)), atol=1e-7)

    def test_sweep_reuses_neighbors_and_matches_individual_predictions(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [4, 4], [4, 5], [5, 4]], dtype=np.float32)
        y = np.array([0, 0, 0, 1, 1, 1])
        queries = np.array([[.1, .2], [4.2, 4.1]], dtype=np.float32)
        progress = []
        model = KNNClassifier(3).fit(X, y)
        results, elapsed = evaluate_knn(model, queries, np.array([0, 1]), [1, 3], [0, 1], batch_size=1, on_progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertGreaterEqual(elapsed, 0)
        for result in results:
            expected = KNNClassifier(result["k"]).fit(X, y).predict(queries)
            np.testing.assert_array_equal(result["predictions"], expected)
            self.assertEqual(result["accuracy"], 1)
        with self.assertRaises(EvaluationCancelled):
            evaluate_knn(model, queries, y[:2], [1], [0, 1], should_stop=lambda: True)

    def test_validation_and_constant_features(self):
        with self.assertRaises(ValueError):
            KNNClassifier(0)
        with self.assertRaises(ValueError):
            KNNClassifier().predict([[0]])
        with self.assertRaises(ValueError):
            KNNClassifier(2).fit([[0]], [0])
        with self.assertRaises(ValueError):
            KNNClassifier(1).fit([[np.nan]], [0])
        model = KNNClassifier(1, standardize=True).fit([[1, 1], [1, 2]], [0, 1])
        self.assertTrue(np.isfinite(model.X_train).all())
        with self.assertRaises(ValueError):
            model.predict([[1, 2, 3]])

    def test_metrics_and_same_sample_comparison(self):
        metrics = classification_metrics([0, 0, 1], [0, 1, 1], [0, 1, 2])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["report"][2]["f1"], 0)
        X = np.zeros((4, 784), dtype=np.float32)
        X[np.arange(4), np.arange(4)] = 1
        y = np.arange(4)
        model = KNNClassifier(1).fit(X, y)
        params = initialize_network((4,), 0)
        copies = tuple(tuple(value.copy() for value in layer) for layer in params)
        result = compare_models(params, model, X, y, batch_size=2)
        self.assertEqual(result["models"]["knn"]["accuracy"], 1)
        self.assertEqual(sum(result["agreements"].values()), 4)
        for layer, original in zip(params, copies):
            for value, saved in zip(layer, original):
                np.testing.assert_array_equal(value, saved)


if __name__ == "__main__":
    unittest.main()
