import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import dashboard
from mnist import init_params, train_model
from network import load_network, train_network


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.directory.name)
        W1, b1, W2, b2 = init_params(0)
        np.savez(self.output_dir / "model.npz", W1=W1, b1=b1, W2=W2, b2=b2)
        self.previous_manager = dashboard.manager
        dashboard.manager = dashboard.TrainingManager(self.output_dir)
        rng = np.random.default_rng(4)
        X = rng.random((784, 4), dtype=np.float32)
        Y = np.array([0, 1, 2, 3])
        images = np.broadcast_to(np.zeros((28, 28), dtype=np.uint8), (60000, 28, 28))
        labels = np.arange(60000) % 10
        dashboard.manager.dataset = (X, Y, X, Y, images, labels)
        self.previous_fixture = dashboard.manager.dataset
        dashboard.manager.phase = "ready"
        dashboard.manager.prepare = lambda: None
        self.client_context = TestClient(dashboard.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        dashboard.manager = self.previous_manager
        self.directory.cleanup()

    def wait_for_completion(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = self.client.get("/api/state").json()
            if state["phase"] not in ("training", "stopping"):
                return state
            time.sleep(0.01)
        self.fail("Training did not finish")

    def test_training_stream_and_saved_model(self):
        response = self.client.post("/api/train", json={"iterations": 3, "alpha": 0.1, "log_every": 1, "seed": 7})
        self.assertEqual(response.status_code, 202)
        state = self.wait_for_completion()
        self.assertEqual(state["phase"], "completed")
        self.assertEqual([event["iteration"] for event in state["events"]], [0, 1, 2, 3])
        self.assertEqual(state["settings"]["alpha"], 0.1)
        self.assertTrue(0 <= state["held_out_accuracy"] <= 1)
        delta = self.client.get("/api/state", params={"since": 2, "run_id": state["run_id"]}).json()
        self.assertEqual([event["iteration"] for event in delta["events"]], [2, 3])
        stale = self.client.get("/api/state", params={"since": 2, "run_id": "previous-run"}).json()
        self.assertEqual(len(stale["events"]), 4)
        reloaded = dashboard.TrainingManager(self.output_dir)
        self.assertEqual(reloaded.settings.seed, 7)
        self.assertEqual(len(reloaded.events), 4)
        self.assertIsNotNone(reloaded.params)
        probabilities = self.client.get("/api/predict?index=50000").json()["probabilities"]
        self.assertEqual(len(probabilities), 10)
        self.assertAlmostEqual(sum(probabilities), 1, places=5)

    def test_validation_and_digit_selection(self):
        for settings in ({"alpha": 0}, {"iterations": 0}, {"seed": -1}, {"iterations": 5001}):
            self.assertEqual(self.client.post("/api/train", json=settings).status_code, 422)
        self.assertEqual(self.client.get("/api/predict?index=60000").status_code, 422)
        sample = self.client.get("/api/sample?digit=8").json()
        self.assertEqual(sample["label"], 8)
        self.assertTrue(50000 <= sample["index"] < 60000)
        dashboard.manager.params = None
        self.assertEqual(self.client.get("/api/predict").status_code, 409)

    def test_uploaded_digit_normalization(self):
        image = Image.new("RGB", (100, 80), "white")
        ImageDraw.Draw(image).line([(45, 15), (45, 65)], fill="black", width=8)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        response = self.client.post("/api/predict-image", content=buffer.getvalue(), headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        pixels = np.array(result["pixels"])
        self.assertEqual(pixels.shape, (28, 28))
        self.assertEqual(pixels[0].max(), 0)
        self.assertGreater(pixels.max(), 200)
        self.assertIsNone(result["label"])
        self.assertIsNone(result["index"])
        self.assertEqual(self.client.post("/api/predict-image", content=b"invalid").status_code, 422)
        buffer = io.BytesIO()
        Image.new("RGB", (28, 28), "white").save(buffer, format="PNG")
        self.assertEqual(self.client.post("/api/predict-image", content=buffer.getvalue()).status_code, 422)

    def test_concurrent_training_and_stop(self):
        gate = threading.Event()
        entered = threading.Event()
        original = train_network

        def delayed_training(*args, **kwargs):
            entered.set()
            gate.wait(3)
            return original(*args, **kwargs)

        with patch("dashboard.train_network", side_effect=delayed_training):
            self.assertEqual(self.client.post("/api/train", json={"iterations": 100}).status_code, 202)
            self.assertTrue(entered.wait(1))
            self.assertEqual(self.client.post("/api/train", json={"iterations": 1}).status_code, 409)
            self.assertEqual(self.client.delete("/api/experiments").status_code, 409)
            self.assertEqual(self.client.delete("/api/experiments/unknown").status_code, 409)
            self.assertEqual(self.client.post("/api/stop").status_code, 200)
            gate.set()
            state = self.wait_for_completion()
        self.assertEqual(state["phase"], "stopped")
        self.assertEqual(state["latest"]["iteration"], 0)
        self.assertIsNotNone(dashboard.manager.params)

    def test_variable_architectures_and_experiment_recovery(self):
        first_settings = {"iterations": 3, "alpha": 0.2, "hidden_neurons": 4, "experiment_name": "One hidden layer", "validation_every": 2}
        self.assertEqual(self.client.post("/api/train", json=first_settings).status_code, 202)
        first = self.wait_for_completion()
        first_id = first["run_id"]
        first_model = load_network(self.output_dir / "experiments" / first_id / "model.npz")
        self.assertEqual(first["architecture"], [784, 4, 10])
        self.assertEqual(first["parameter_count"], 3190)
        validation_steps = [event["iteration"] for event in first["events"] if event["validation_accuracy"] is not None]
        self.assertEqual(validation_steps, [0, 2, 3])
        second_settings = {**first_settings, "second_hidden_enabled": True, "second_hidden_neurons": 3, "experiment_name": "Two hidden layers"}
        self.assertEqual(self.client.post("/api/train", json=second_settings).status_code, 202)
        second = self.wait_for_completion()
        self.assertEqual(second["architecture"], [784, 4, 3, 10])
        self.assertEqual(second["parameter_count"], 3195)
        self.assertEqual(len(self.client.get("/api/experiments").json()["experiments"]), 2)
        record = self.client.get(f"/api/experiments/{first_id}").json()
        self.assertEqual(record["settings"], dashboard.TrainingSettings(**first_settings).model_dump())
        self.assertEqual(len(record["history"]), 4)
        after = load_network(self.output_dir / "experiments" / first_id / "model.npz")
        for old_layer, saved_layer in zip(first_model, after):
            for old_values, saved_values in zip(old_layer, saved_layer):
                np.testing.assert_array_equal(old_values, saved_values)
        prediction = self.client.get("/api/predict", params={"experiment_id": first_id}).json()
        self.assertEqual(prediction["architecture"], [784, 4, 10])
        self.assertEqual(prediction["experiment_id"], first_id)
        self.assertEqual(self.client.get("/api/predict?experiment_id=unknown").status_code, 404)
        reloaded = dashboard.TrainingManager(self.output_dir)
        self.assertEqual(reloaded.settings.hidden_layers, (4, 3))
        self.assertEqual(len(reloaded.list_experiments()), 2)
        self.assertEqual(reloaded.model_experiment_id, second["run_id"])
        dashboard.manager = reloaded
        reloaded.dataset = self.previous_fixture
        restored = self.client.get("/api/predict", params={"experiment_id": first_id}).json()
        np.testing.assert_array_equal(restored["probabilities"], prediction["probabilities"])

    def test_architecture_validation(self):
        for settings in ({"hidden_neurons": 0}, {"hidden_neurons": 257}, {"second_hidden_neurons": 0}, {"validation_every": 0}):
            self.assertEqual(self.client.post("/api/train", json=settings).status_code, 422)

    def test_automatic_names_follow_settings_and_distinguish_repeated_runs(self):
        settings = {"iterations": 2, "alpha": 0.8, "seed": 7, "hidden_neurons": 4,
                    "second_hidden_enabled": True, "second_hidden_neurons": 3, "experiment_name": "Stale name"}
        names = []
        for _ in range(2):
            self.assertEqual(self.client.post("/api/train", json=settings).status_code, 202)
            state = self.wait_for_completion()
            record = self.client.get(f"/api/experiments/{state['run_id']}").json()
            self.assertEqual(record["name"], f"Alpha 0.8 · 4 → 3 neuronas · 2 iteraciones · semilla 7 · #{state['run_id'][:8]}")
            names.append(record["name"])
        self.assertNotEqual(*names)

    def test_old_names_are_corrected_without_changing_model_or_metrics(self):
        experiment_id = self.create_experiment()
        path = self.output_dir / "experiments" / experiment_id / "run.json"
        record = json.loads(path.read_text())
        record["name"] = "Wrong alpha and iterations"
        path.write_text(json.dumps(record))
        model_path = path.parent / "model.npz"
        previous_bytes = model_path.read_bytes()
        reloaded = dashboard.TrainingManager(self.output_dir)
        restored = json.loads(path.read_text())
        self.assertEqual(restored["name"], f"Alpha 0.5 · 4 neuronas · 1 iteraciones · semilla 0 · #{experiment_id[:8]}")
        self.assertEqual(reloaded.experiments[experiment_id], restored)
        self.assertEqual({k: v for k, v in record.items() if k != "name"}, {k: v for k, v in restored.items() if k != "name"})
        self.assertEqual(previous_bytes, model_path.read_bytes())

    def create_experiment(self, hidden_neurons=4):
        response = self.client.post("/api/train", json={"iterations": 1, "hidden_neurons": hidden_neurons})
        self.assertEqual(response.status_code, 202)
        state = self.wait_for_completion()
        self.assertEqual(state["phase"], "completed")
        return state["run_id"]

    def test_delete_previous_experiment_preserves_current_model(self):
        first_id = self.create_experiment()
        second_id = self.create_experiment(5)
        current_bytes = (self.output_dir / "model.npz").read_bytes()
        response = self.client.delete(f"/api/experiments/{first_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], 1)
        self.assertEqual(response.json()["state"]["model_experiment_id"], second_id)
        self.assertEqual(current_bytes, (self.output_dir / "model.npz").read_bytes())
        self.assertFalse((self.output_dir / "experiments" / first_id).exists())
        self.assertNotIn(first_id, dashboard.manager.experiment_models)
        self.assertEqual(self.client.get(f"/api/experiments/{first_id}").status_code, 404)
        self.assertEqual(self.client.get("/api/predict", params={"experiment_id": first_id}).status_code, 404)
        self.assertEqual(self.client.delete("/api/experiments/unknown").status_code, 404)
        reloaded = dashboard.TrainingManager(self.output_dir)
        self.assertEqual(reloaded.model_experiment_id, second_id)
        self.assertEqual(len(reloaded.list_experiments()), 1)

    def test_delete_current_model_restores_previous_and_last_delete_stays_empty(self):
        first_id = self.create_experiment()
        first_prediction = self.client.get("/api/predict").json()
        second_id = self.create_experiment(5)
        response = self.client.delete(f"/api/experiments/{second_id}")
        self.assertEqual(response.status_code, 200)
        state = response.json()["state"]
        self.assertEqual(state["run_id"], first_id)
        self.assertEqual(state["model_architecture"], [784, 4, 10])
        np.testing.assert_array_equal(self.client.get("/api/predict").json()["probabilities"], first_prediction["probabilities"])
        self.assertEqual(dashboard.TrainingManager(self.output_dir).model_experiment_id, first_id)
        legacy_project = self.output_dir / "legacy"
        (legacy_project / "outputs").mkdir(parents=True)
        legacy_path = legacy_project / "outputs" / "model.npz"
        legacy_path.write_bytes((self.output_dir / "model.npz").read_bytes())
        response = self.client.delete(f"/api/experiments/{first_id}")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["state"]["has_model"])
        self.assertEqual(response.json()["state"]["events"], [])
        self.assertEqual(self.client.get("/api/predict").status_code, 409)
        for name in ("model.npz", "history.csv", "run.json"):
            self.assertFalse((self.output_dir / name).exists())
        with patch("dashboard.PROJECT_DIR", legacy_project):
            reloaded = dashboard.TrainingManager(self.output_dir)
        self.assertIsNone(reloaded.params)
        self.assertEqual(reloaded.list_experiments(), [])
        self.assertTrue(legacy_path.exists())
        self.assertIs(dashboard.manager.dataset, self.previous_fixture)

    def test_delete_all_experiments_and_train_again(self):
        self.create_experiment()
        self.create_experiment(5)
        unrelated = self.output_dir / "dashboard.png"
        unrelated.write_bytes(b"keep")
        response = self.client.delete("/api/experiments")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], 2)
        self.assertEqual(self.client.get("/api/experiments").json()["experiments"], [])
        self.assertEqual(dashboard.manager.experiment_models, {})
        self.assertIsNone(dashboard.manager.params)
        self.assertTrue(unrelated.exists())
        self.assertIs(dashboard.manager.dataset, self.previous_fixture)
        self.assertEqual(list((self.output_dir / "experiments").iterdir()), [])
        self.assertIsNone(dashboard.TrainingManager(self.output_dir).params)
        new_id = self.create_experiment(6)
        self.assertEqual(dashboard.TrainingManager(self.output_dir).model_experiment_id, new_id)
        self.assertEqual(self.client.get("/api/predict").status_code, 200)

    def test_progress_callback_matches_cli_metrics(self):
        X, Y = dashboard.manager.dataset[:2]
        events = []
        result = train_model(X, Y, iterations=2, log_every=1, on_progress=lambda *values: events.append(values), verbose=False)
        self.assertEqual(events, result[-1])
        self.assertEqual([event[0] for event in events], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
