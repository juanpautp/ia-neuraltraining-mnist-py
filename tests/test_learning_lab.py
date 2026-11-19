import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from knn import EvaluationCancelled
from learning_lab import LearningLab, MNIST_DATASET, create_lab_router, sample_indices
from network import initialize_network, save_network


class LearningLabTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        X = np.zeros((784, 12), dtype=np.float32)
        y = np.tile(np.arange(4), 3)
        X[y, np.arange(12)] = 1
        validation = X[:, :4].copy()
        images = (np.concatenate((X, validation), axis=1).T.reshape(16, 28, 28) * 255).astype(np.uint8)
        dataset = X, y, validation, np.arange(4), images, np.concatenate((y, np.arange(4)))
        network_dir = self.root / "neural"
        model_dir = network_dir / "experiments" / "network1"
        model_dir.mkdir(parents=True)
        save_network(model_dir / "model.npz", initialize_network((4,), 0))
        record = {"id": "network1", "name": "Network fixture", "dataset": MNIST_DATASET, "duration": 12.5}

        def get_experiment(experiment_id):
            if experiment_id != "network1":
                raise HTTPException(404, "Unknown experiment")
            return record

        self.manager = SimpleNamespace(dataset=dataset, phase="ready", output_dir=network_dir, get_experiment=get_experiment)
        self.lab = LearningLab(lambda: self.manager, self.root / "lab")
        app = FastAPI()
        app.include_router(create_lab_router(self.lab))
        self.client = TestClient(app)

    def tearDown(self):
        self.lab.stop_event.set()
        deadline = time.monotonic() + 3
        while self.lab.busy and time.monotonic() < deadline:
            time.sleep(.01)
        self.client.close()
        self.directory.cleanup()

    def wait(self, phase="completed"):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = self.client.get("/api/lab/state").json()
            if state["phase"] not in ("running", "stopping"):
                self.assertEqual(state["phase"], phase, state)
                return state
            time.sleep(.01)
        self.fail("Evaluation did not finish")

    def create_knn(self, **changes):
        settings = {"dataset": "mnist", "k_values": [1, 3], "train_count": 12, "evaluation_count": 4, **changes}
        response = self.client.post("/api/lab/knn", json=settings)
        self.assertEqual(response.status_code, 202, response.text)
        state = self.wait()
        return self.client.get(f"/api/lab/runs/{state['run_id']}").json()

    def test_knn_saved_reloaded_predicted_and_deleted(self):
        record = self.create_knn()
        self.assertEqual(record["summary"]["accuracy"], 1)
        self.assertEqual(record["summary"]["best_k"], 1)
        self.assertEqual(set(record["reference_indices"]), set(range(12)))
        prediction = self.client.post("/api/lab/predict", json={"run_id": record["id"], "index": 12}).json()
        self.assertEqual(prediction["prediction"], 0)
        self.assertEqual(sum(prediction["votes"].values()), 1)
        reloaded = LearningLab(lambda: self.manager, self.root / "lab")
        self.assertEqual(reloaded.get_record(record["id"]), record)
        model, _ = reloaded.model_for_record(record)
        np.testing.assert_array_equal(model.predict(self.manager.dataset[2].T), np.arange(4))
        self.assertEqual(self.client.get(f"/api/lab/runs/{record['id']}/download").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/lab/runs/{record['id']}").json()["deleted"], 1)
        self.assertEqual(self.client.get(f"/api/lab/runs/{record['id']}").status_code, 404)
        self.assertTrue((self.manager.output_dir / "experiments" / "network1" / "model.npz").exists())

    def test_comparison_evaluates_identical_images_and_preserves_weights(self):
        record = self.create_knn()
        path = self.manager.output_dir / "experiments" / "network1" / "model.npz"
        previous = path.read_bytes()
        settings = {"network_experiment_id": "network1", "knn_run_id": record["id"], "k": 1, "evaluation_count": 4}
        response = self.client.post("/api/lab/compare", json=settings)
        self.assertEqual(response.status_code, 202, response.text)
        state = self.wait()
        result = self.client.get(f"/api/lab/runs/{state['run_id']}").json()
        self.assertEqual(result["evaluation_indices"], [12, 13, 14, 15])
        self.assertEqual(result["expected"], [0, 1, 2, 3])
        self.assertEqual(result["models"]["knn"]["accuracy"], 1)
        self.assertEqual(sum(result["agreements"].values()), 4)
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(self.client.delete("/api/lab/runs?kind=comparison").json()["deleted"], 1)
        self.assertEqual(len(self.client.get("/api/lab/runs?kind=knn").json()["runs"]), 1)

    def test_comparison_rejects_different_reference_counts(self):
        record = self.create_knn(train_count=8)
        response = self.client.post("/api/lab/compare", json={"network_experiment_id": "network1", "knn_run_id": record["id"], "evaluation_count": 4})
        self.assertEqual(response.status_code, 422)
        self.assertIn("igualdad", response.json()["detail"])

    def test_iris_works_without_mnist_and_has_vote_probabilities(self):
        self.manager.dataset = None
        record = self.create_knn(dataset="iris", standardize=True, seed=42)
        self.assertEqual(record["summary"]["reference_count"], 120)
        self.assertEqual(record["summary"]["evaluation_count"], 30)
        self.assertFalse(set(record["reference_indices"]) & set(record["evaluation_indices"]))
        response = self.client.post("/api/lab/predict", json={"run_id": record["id"], "features": [5.1, 3.5, 1.4, .2]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction_name"], "setosa")
        self.assertEqual(sum(response.json()["votes"].values()), record["summary"]["best_k"])

    def test_validation_unknown_ids_and_no_train_validation_leakage(self):
        for changes in ({"k_values": [0]}, {"k_values": [51]}, {"k_values": [1.5]}, {"k_values": [True]}, {"train_count": 13}, {"evaluation_count": 5}, {"standardize": True}):
            response = self.client.post("/api/lab/knn", json={"train_count": 12, "evaluation_count": 4, **changes})
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.get("/api/lab/runs/missing").status_code, 404)
        self.assertEqual(self.client.delete("/api/lab/runs/missing").status_code, 404)
        labels = np.repeat(np.arange(3), 10)
        indices = sample_indices(labels, 12, 4)
        np.testing.assert_array_equal(np.bincount(labels[indices]), [4, 4, 4])
        np.testing.assert_array_equal(indices, sample_indices(labels, 12, 4))
        self.assertFalse(np.array_equal(indices, sample_indices(labels, 12, 5)))

    def test_cancel_and_mutual_exclusion(self):
        entered = threading.Event()

        def delayed(*args, **kwargs):
            entered.set()
            self.lab.stop_event.wait(3)
            raise EvaluationCancelled()

        with patch("learning_lab.evaluate_knn", side_effect=delayed):
            self.assertEqual(self.client.post("/api/lab/knn", json={"train_count": 12, "evaluation_count": 4}).status_code, 202)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.client.post("/api/lab/knn", json={}).status_code, 409)
            self.assertEqual(self.client.delete("/api/lab/runs?kind=knn").status_code, 409)
            self.assertEqual(self.client.post("/api/lab/stop").status_code, 200)
            self.wait("stopped")
        self.assertEqual(self.lab.list_records(), [])
        self.manager.phase = "training"
        self.assertEqual(self.client.post("/api/lab/knn", json={}).status_code, 409)


if __name__ == "__main__":
    unittest.main()
