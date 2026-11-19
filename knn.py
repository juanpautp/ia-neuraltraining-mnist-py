import time
from collections import Counter

import numpy as np


class EvaluationCancelled(Exception):
    pass


class KNNClassifier:
    def __init__(self, k=5, metric="euclidean", standardize=False):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if metric != "euclidean":
            raise ValueError("Only Euclidean distance is supported")
        self.k = k
        self.metric = metric
        self.standardize = standardize
        self.X_train = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite matrix with one sample per row")
        if y.ndim != 1 or len(y) != len(X) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain one integer label per sample")
        if self.k > len(X):
            raise ValueError("k cannot exceed the number of reference samples")
        self.mean = X.mean(axis=0) if self.standardize else np.zeros(X.shape[1], dtype=np.float32)
        self.scale = X.std(axis=0) if self.standardize else np.ones(X.shape[1], dtype=np.float32)
        self.scale[self.scale == 0] = 1
        self.X_train = np.ascontiguousarray((X - self.mean) / self.scale)
        self.y_train = y.astype(np.int64, copy=True)
        self.classes_ = np.unique(self.y_train)
        self.squared_norms = np.einsum("ij,ij->i", self.X_train, self.X_train)
        return self

    def _queries(self, X):
        if self.X_train is None:
            raise ValueError("Fit the classifier before predicting")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1] or not np.isfinite(X).all():
            raise ValueError("Query dimensions must match the reference samples")
        return np.ascontiguousarray((X - self.mean) / self.scale)

    def kneighbors(self, X, k=None):
        queries = self._queries(X)
        k = self.k if k is None else k
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(self.X_train):
            raise ValueError("k must be between 1 and the number of reference samples")
        distances = queries @ self.X_train.T
        distances *= -2
        distances += self.squared_norms
        distances += np.einsum("ij,ij->i", queries, queries)[:, None]
        np.maximum(distances, 0, out=distances)
        selected = np.argpartition(distances, k - 1, axis=1)[:, :k]
        ordered = np.empty_like(selected)
        for row, candidates in enumerate(selected):
            threshold = distances[row, candidates].max()
            closer = np.flatnonzero(distances[row] < threshold)
            tied = np.flatnonzero(distances[row] == threshold)
            candidates = np.concatenate((closer, tied[:k - len(closer)]))
            order = np.lexsort((candidates, distances[row, candidates]))
            ordered[row] = candidates[order]
        values = np.take_along_axis(distances, ordered, axis=1)
        return np.sqrt(values), ordered

    def predict_one(self, x):
        return int(self.predict(np.asarray(x)[None, :])[0])

    def predict(self, X, batch_size=64):
        X = np.asarray(X)
        predictions = []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(X), batch_size):
            _, indices = self.kneighbors(X[start:start + batch_size])
            predictions.extend(vote_labels(self.y_train[indices]))
        return np.asarray(predictions, dtype=np.int64)

    def predict_proba(self, X, batch_size=64):
        X = np.asarray(X)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        probabilities = np.empty((len(X), len(self.classes_)), dtype=np.float32)
        for start in range(0, len(X), batch_size):
            _, indices = self.kneighbors(X[start:start + batch_size])
            labels = self.y_train[indices]
            probabilities[start:start + len(indices)] = np.stack(
                [(labels == label).mean(axis=1) for label in self.classes_], axis=1
            )
        return probabilities

    @property
    def model_bytes(self):
        if self.X_train is None:
            return 0
        return sum(values.nbytes for values in (
            self.X_train, self.y_train, self.squared_norms, self.mean, self.scale, self.classes_
        ))


def vote_labels(neighbor_labels):
    return np.array([Counter(row.tolist()).most_common(1)[0][0] for row in neighbor_labels], dtype=np.int64)


def classification_metrics(expected, predicted, classes):
    expected = np.asarray(expected)
    predicted = np.asarray(predicted)
    classes = np.asarray(classes)
    if expected.ndim != 1 or expected.shape != predicted.shape or not len(expected):
        raise ValueError("Expected and predicted labels must have matching nonempty shapes")
    positions = {int(label): index for index, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, guess in zip(expected, predicted):
        matrix[positions[int(actual)], positions[int(guess)]] += 1
    report = []
    for index, label in enumerate(classes):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        guessed = int(matrix[:, index].sum())
        precision = true_positive / guessed if guessed else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report.append({"label": int(label), "precision": precision, "recall": recall, "f1": f1, "support": support})
    correct = int(np.trace(matrix))
    return {
        "accuracy": correct / len(expected), "correct": correct, "errors": len(expected) - correct,
        "macro_f1": float(np.mean([row["f1"] for row in report])),
        "confusion_matrix": matrix.tolist(), "report": report,
    }


def evaluate_knn(model, X, y, k_values, classes, batch_size=64, on_progress=None, should_stop=None):
    k_values = sorted(set(k_values))
    if not k_values or min(k_values) < 1 or max(k_values) > len(model.y_train):
        raise ValueError("Choose k values within the reference sample count")
    predictions = {k: np.empty(len(y), dtype=np.int64) for k in k_values}
    started = time.perf_counter()
    for start in range(0, len(y), batch_size):
        if should_stop is not None and should_stop():
            raise EvaluationCancelled()
        stop = min(start + batch_size, len(y))
        _, indices = model.kneighbors(X[start:stop], max(k_values))
        labels = model.y_train[indices]
        for k in k_values:
            predictions[k][start:stop] = vote_labels(labels[:, :k])
        if on_progress is not None:
            on_progress(stop, len(y))
    elapsed = time.perf_counter() - started
    results = [
        {"k": k, **classification_metrics(y, predictions[k], classes), "predictions": predictions[k].tolist()}
        for k in k_values
    ]
    return results, elapsed
