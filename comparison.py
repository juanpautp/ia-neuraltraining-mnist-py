import time

import numpy as np

from knn import EvaluationCancelled, classification_metrics
from network import forward_network


def compare_models(params, knn_model, X, y, batch_size=64, on_progress=None, should_stop=None):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y) or not len(y) or X.shape[1] != params[0][0].shape[1]:
        raise ValueError("Both models must receive the same nonempty evaluation samples")
    classes = np.arange(params[-1][0].shape[0])
    predictions = {"network": np.empty(len(y), dtype=np.int64), "knn": np.empty(len(y), dtype=np.int64)}
    elapsed = {"network": 0.0, "knn": 0.0}
    forward_network(params, X[:1].T)
    knn_model.predict(X[:1])
    for start in range(0, len(y), batch_size):
        if should_stop is not None and should_stop():
            raise EvaluationCancelled()
        stop = min(start + batch_size, len(y))
        batch = X[start:stop]
        for method in ("network", "knn") if (start // batch_size) % 2 == 0 else ("knn", "network"):
            started = time.perf_counter()
            if method == "network":
                predicted = np.argmax(forward_network(params, batch.T)[-1], axis=0)
            else:
                predicted = knn_model.predict(batch, batch_size=batch_size)
            elapsed[method] += time.perf_counter() - started
            predictions[method][start:stop] = predicted
        if on_progress is not None:
            on_progress(stop, len(y))
    results = {}
    for method in predictions:
        results[method] = {
            **classification_metrics(y, predictions[method], classes),
            "prediction_seconds": elapsed[method], "milliseconds_per_image": elapsed[method] * 1000 / len(y),
            "predictions": predictions[method].tolist(),
        }
    network_correct = predictions["network"] == y
    knn_correct = predictions["knn"] == y
    agreements = {
        "both_correct": int(np.sum(network_correct & knn_correct)),
        "network_only": int(np.sum(network_correct & ~knn_correct)),
        "knn_only": int(np.sum(~network_correct & knn_correct)),
        "both_wrong": int(np.sum(~network_correct & ~knn_correct)),
    }
    disagreements = np.flatnonzero(predictions["network"] != predictions["knn"])
    return {"models": results, "agreements": agreements, "disagreement_positions": disagreements[:24].tolist()}
