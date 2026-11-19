import copy
import json
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from comparison import compare_models
from knn import EvaluationCancelled, KNNClassifier, evaluate_knn, vote_labels
from network import load_network


MNIST_DATASET = "mnist-train-first-50000-validation-last-10000"


class KNNSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    dataset: Literal["mnist", "iris"] = "mnist"
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 7, 9], min_length=1, max_length=20)
    train_count: int = Field(default=50000, ge=1, le=50000)
    evaluation_count: int = Field(default=1000, ge=1, le=10000)
    seed: int = Field(default=0, ge=0, le=4294967295)
    validation_fraction: float = Field(default=0.2, ge=0.1, le=0.4)
    standardize: bool = False

    @field_validator("k_values", mode="before")
    @classmethod
    def validate_k_values(cls, values):
        if not isinstance(values, list) or not values or len(values) > 20:
            raise ValueError("Elige entre uno y veinte valores de K")
        if any(type(value) is not int or not 1 <= value <= 50 for value in values):
            raise ValueError("Cada K debe ser un entero entre 1 y 50")
        return sorted(set(values))


class ComparisonSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    network_experiment_id: str = Field(min_length=1, max_length=64)
    knn_run_id: str = Field(min_length=1, max_length=64)
    k: int = Field(default=5, ge=1, le=50)
    evaluation_count: int = Field(default=1000, ge=1, le=10000)
    seed: int = Field(default=0, ge=0, le=4294967295)


class PredictionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    run_id: str = Field(min_length=1, max_length=64)
    k: int | None = Field(default=None, ge=1, le=50)
    index: int = Field(default=50000, ge=0, lt=60000)
    features: list[float] | None = Field(default=None, min_length=4, max_length=4)


def sample_indices(labels, count, seed):
    labels = np.asarray(labels)
    if not 1 <= count <= len(labels):
        raise ValueError(f"La cantidad debe estar entre 1 y {len(labels)}")
    if count == len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    quotas = counts * count / len(labels)
    allocation = np.floor(quotas).astype(int)
    for index in np.argsort(-(quotas - allocation), kind="stable")[:count - allocation.sum()]:
        allocation[index] += 1
    indices = np.concatenate([
        rng.choice(np.flatnonzero(labels == label), size=size, replace=False)
        for label, size in zip(classes, allocation)
    ])
    rng.shuffle(indices)
    return indices


def take_rows(values, indices):
    if len(indices) == len(values) and np.array_equal(indices, np.arange(len(values))):
        return values
    return values[indices]


def prepare_data(settings, mnist=None, reference_indices=None, evaluation_indices=None):
    if settings.dataset == "iris":
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        iris = load_iris()
        reference, evaluation = train_test_split(
            np.arange(len(iris.target)), test_size=settings.validation_fraction,
            random_state=settings.seed, stratify=iris.target,
        )
        if reference_indices is not None:
            reference = np.asarray(reference_indices)
            evaluation = np.asarray(evaluation_indices)
        return {
            "X_train": iris.data[reference], "y_train": iris.target[reference],
            "X_evaluation": iris.data[evaluation], "y_evaluation": iris.target[evaluation],
            "reference_indices": reference, "evaluation_indices": evaluation,
            "classes": np.arange(3), "class_names": iris.target_names.tolist(),
            "dataset_id": "sklearn-iris-150", "evaluation_offset": 0,
        }
    if mnist is None:
        raise ValueError("MNIST todavía está cargando. Espera unos segundos")
    if settings.standardize:
        raise ValueError("MNIST usa píxeles entre 0 y 1, igual que la red neuronal; desactiva la estandarización")
    X_train, y_train, X_evaluation, y_evaluation = mnist[:4]
    reference = sample_indices(y_train, settings.train_count, settings.seed) if reference_indices is None else np.asarray(reference_indices)
    evaluation = sample_indices(y_evaluation, settings.evaluation_count, settings.seed) if evaluation_indices is None else np.asarray(evaluation_indices)
    return {
        "X_train": take_rows(X_train.T, reference), "y_train": y_train[reference],
        "X_evaluation": take_rows(X_evaluation.T, evaluation), "y_evaluation": y_evaluation[evaluation],
        "reference_indices": reference, "evaluation_indices": evaluation,
        "classes": np.arange(10), "class_names": list(map(str, range(10))),
        "dataset_id": MNIST_DATASET, "evaluation_offset": len(y_train),
    }


class LearningLab:
    def __init__(self, manager_provider, output_dir):
        self.manager_provider = manager_provider
        self.output_dir = Path(output_dir)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.phase = "idle"
        self.kind = None
        self.run_id = None
        self.started_at = None
        self.duration = 0
        self.progress = {"completed": 0, "total": 0, "message": "Listo para comenzar"}
        self.logs = []
        self.error = None
        self.records = {}
        self.cached_model = None
        for path in self.output_dir.glob("*/run.json"):
            try:
                record = json.loads(path.read_text())
                if record["id"] == path.parent.name and record["kind"] in ("knn", "comparison"):
                    self.records[record["id"]] = record
            except (OSError, ValueError, KeyError):
                continue

    @property
    def busy(self):
        return self.phase in ("running", "stopping")

    def snapshot(self):
        with self.lock:
            return {
                "phase": self.phase, "kind": self.kind, "run_id": self.run_id,
                "progress": dict(self.progress), "logs": list(self.logs), "error": self.error,
                "duration": time.monotonic() - self.started_at if self.busy else self.duration,
                "dataset_ready": self.manager_provider().dataset is not None,
                "network_busy": self.manager_provider().phase in ("training", "stopping"),
            }

    def list_records(self, kind=None):
        with self.lock:
            records = sorted(self.records.values(), key=lambda row: row["created_at"], reverse=True)
            return [{key: record[key] for key in ("id", "kind", "name", "created_at", "settings", "summary")} for record in records if kind is None or record["kind"] == kind]

    def get_record(self, run_id):
        with self.lock:
            if run_id not in self.records:
                raise HTTPException(404, "No se encontró esa ejecución")
            return self.records[run_id]

    def log(self, completed, total, message):
        with self.lock:
            self.progress = {"completed": completed, "total": total, "message": message}
            self.logs.append(message)
            self.logs = self.logs[-2000:]

    def start(self, kind, settings):
        with self.lock:
            if self.busy or self.manager_provider().phase in ("training", "stopping"):
                raise HTTPException(409, "Espera a que termine la ejecución actual o detenla primero")
            if kind == "knn":
                try:
                    data = prepare_data(settings, self.manager_provider().dataset)
                    if max(settings.k_values) > len(data["y_train"]):
                        raise ValueError("K no puede superar la cantidad de ejemplos de referencia")
                except ValueError as error:
                    raise HTTPException(422, str(error)) from error
            else:
                self.validate_comparison(settings)
            self.phase = "running"
            self.kind = kind
            self.run_id = uuid.uuid4().hex
            self.started_at = time.monotonic()
            self.duration = 0
            self.logs = []
            self.progress = {"completed": 0, "total": 0, "message": "Preparando los datos"}
            self.error = None
            self.stop_event.clear()
            threading.Thread(target=self.run, args=(kind, settings, self.run_id), daemon=True).start()
            return self.snapshot()

    def validate_comparison(self, settings):
        manager = self.manager_provider()
        if manager.dataset is None:
            raise HTTPException(409, "MNIST todavía está cargando")
        network = manager.get_experiment(settings.network_experiment_id)
        record = self.get_record(settings.knn_run_id)
        if record["kind"] != "knn" or record["settings"]["dataset"] != "mnist":
            raise HTTPException(422, "Selecciona una ejecución KNN de MNIST, no de Iris")
        if network.get("dataset") != MNIST_DATASET or record["dataset_id"] != MNIST_DATASET:
            raise HTTPException(422, "Ambos modelos deben utilizar la misma partición de MNIST")
        references = record["reference_indices"]
        if references != list(range(len(manager.dataset[1]))):
            raise HTTPException(422, "Para comparar en igualdad de condiciones, prepara KNN con todos los ejemplos de entrenamiento (50.000 en MNIST)")
        if settings.k > len(references) or settings.evaluation_count > len(manager.dataset[3]):
            raise HTTPException(422, "K o la cantidad de imágenes superan los datos disponibles")

    def run(self, kind, settings, run_id):
        outcome = "completed"
        try:
            record = self.run_knn(settings, run_id) if kind == "knn" else self.run_comparison(settings, run_id)
            if self.stop_event.is_set():
                raise EvaluationCancelled()
            record.update(id=run_id, kind=kind, created_at=datetime.now(timezone.utc).isoformat(), settings=settings.model_dump())
            directory = self.output_dir / run_id
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "run.tmp.json"
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
            temporary.replace(directory / "run.json")
            with self.lock:
                self.records[run_id] = record
            self.log(self.progress["total"], self.progress["total"], "Evaluación completa. Resultado guardado en tu PC")
        except EvaluationCancelled:
            outcome = "stopped"
            self.log(self.progress["completed"], self.progress["total"], "Evaluación detenida. No se guardó un resultado parcial")
        except Exception as error:
            outcome = "error"
            with self.lock:
                self.error = f"No se pudo completar la ejecución: {error}"
        finally:
            with self.lock:
                self.duration = time.monotonic() - self.started_at
                self.phase = outcome

    def run_knn(self, settings, run_id):
        data = prepare_data(settings, self.manager_provider().dataset)
        self.log(0, len(data["y_evaluation"]), f"Referencia: {len(data['y_train']):,} ejemplos · Validación: {len(data['y_evaluation']):,} · K: {settings.k_values}")
        started = time.perf_counter()
        model = KNNClassifier(max(settings.k_values), standardize=settings.standardize).fit(data["X_train"], data["y_train"])
        fit_seconds = time.perf_counter() - started
        results, evaluation_seconds = evaluate_knn(
            model, data["X_evaluation"], data["y_evaluation"], settings.k_values, data["classes"],
            on_progress=lambda done, total: self.log(done, total, f"Evaluadas {done:,}/{total:,} imágenes o muestras"),
            should_stop=self.stop_event.is_set,
        )
        best = max(results, key=lambda row: (row["accuracy"], -row["k"]))
        model.k = best["k"]
        with self.lock:
            self.cached_model = (run_id, model, data)
        dataset_name = "MNIST" if settings.dataset == "mnist" else "Iris"
        return {
            "name": f"KNN {dataset_name} · K {','.join(map(str, settings.k_values))} · {len(data['y_train'])} referencias · #{run_id[:8]}",
            "dataset_id": data["dataset_id"], "class_names": data["class_names"],
            "reference_indices": data["reference_indices"].tolist(), "evaluation_indices": data["evaluation_indices"].tolist(),
            "evaluation_offset": data["evaluation_offset"], "expected": data["y_evaluation"].tolist(),
            "results": results, "fit_seconds": fit_seconds, "evaluation_seconds": evaluation_seconds,
            "model_bytes": model.model_bytes,
            "summary": {"best_k": best["k"], "accuracy": best["accuracy"], "reference_count": len(data["y_train"]), "evaluation_count": len(data["y_evaluation"]), "evaluation_seconds": evaluation_seconds},
        }

    def model_for_record(self, record):
        with self.lock:
            if self.cached_model is not None and self.cached_model[0] == record["id"]:
                return self.cached_model[1:]
        settings = KNNSettings(**record["settings"])
        data = prepare_data(settings, self.manager_provider().dataset, record["reference_indices"], record["evaluation_indices"])
        model = KNNClassifier(record["summary"]["best_k"], standardize=settings.standardize).fit(data["X_train"], data["y_train"])
        with self.lock:
            self.cached_model = (record["id"], model, data)
        return model, data

    def run_comparison(self, settings, run_id):
        manager = self.manager_provider()
        network_record = manager.get_experiment(settings.network_experiment_id)
        knn_record = self.get_record(settings.knn_run_id)
        params = load_network(manager.output_dir / "experiments" / network_record["id"] / "model.npz")
        fitted, _ = self.model_for_record(knn_record)
        model = copy.copy(fitted)
        model.k = settings.k
        indices = sample_indices(manager.dataset[3], settings.evaluation_count, settings.seed)
        self.log(0, len(indices), f"Ambos métodos: {len(manager.dataset[1]):,} ejemplos de entrenamiento y las mismas {len(indices):,} imágenes reservadas")
        result = compare_models(
            params, model, take_rows(manager.dataset[2].T, indices), manager.dataset[3][indices],
            on_progress=lambda done, total: self.log(done, total, f"Comparadas {done:,}/{total:,} imágenes en ambos métodos"),
            should_stop=self.stop_event.is_set,
        )
        result["models"]["network"].update(name=network_record["name"], preparation_seconds=network_record.get("duration"), model_bytes=sum(a.nbytes for layer in params for a in layer))
        result["models"]["knn"].update(name=f"KNN · K {settings.k}", preparation_seconds=knn_record["fit_seconds"], model_bytes=model.model_bytes)
        offset = len(manager.dataset[1])
        examples = []
        for position in result.pop("disagreement_positions"):
            index = int(indices[position])
            examples.append({"index": index + offset, "label": int(manager.dataset[3][index]), "pixels": (manager.dataset[2][:, index].reshape(28, 28) * 255).round().astype(int).tolist(), "network": result["models"]["network"]["predictions"][position], "knn": result["models"]["knn"]["predictions"][position]})
        return {
            **result, "name": f"Red vs KNN · K {settings.k} · {len(indices)} imágenes · #{run_id[:8]}",
            "dataset_id": MNIST_DATASET, "class_names": list(map(str, range(10))),
            "reference_count": len(manager.dataset[1]), "evaluation_indices": (indices + offset).tolist(),
            "expected": manager.dataset[3][indices].tolist(), "examples": examples,
            "summary": {"network_accuracy": result["models"]["network"]["accuracy"], "knn_accuracy": result["models"]["knn"]["accuracy"], "evaluation_count": len(indices)},
        }

    def predict(self, settings):
        record = self.get_record(settings.run_id)
        if record["kind"] != "knn":
            raise HTTPException(422, "Selecciona una ejecución de KNN")
        model, data = self.model_for_record(record)
        k = settings.k or record["summary"]["best_k"]
        if k > len(model.y_train):
            raise HTTPException(422, "K supera la cantidad de referencias")
        if record["settings"]["dataset"] == "iris":
            if settings.features is None:
                raise HTTPException(422, "Ingresa las cuatro medidas de la flor")
            query = np.array(settings.features, dtype=np.float32)[None, :]
            pixels = None
            label = None
        else:
            manager = self.manager_provider()
            if manager.dataset is None:
                raise HTTPException(409, "MNIST todavía está cargando")
            pixels = manager.dataset[4][settings.index]
            label = int(manager.dataset[5][settings.index])
            query = pixels.reshape(1, 784).astype(np.float32) / 255
        distances, indices = model.kneighbors(query, k)
        labels = model.y_train[indices]
        prediction = int(vote_labels(labels)[0])
        votes = {str(int(value)): int(np.sum(labels[0] == value)) for value in data["classes"]}
        neighbors = []
        for distance, index in zip(distances[0][:12], indices[0][:12]):
            item = {"label": int(model.y_train[index]), "distance": float(distance), "reference_index": int(data["reference_indices"][index])}
            if pixels is not None:
                item["pixels"] = self.manager_provider().dataset[4][item["reference_index"]].tolist()
            else:
                item["features"] = data["X_train"][index].tolist()
            neighbors.append(item)
        return {"prediction": prediction, "prediction_name": data["class_names"][prediction], "label": label, "pixels": pixels.tolist() if pixels is not None else None, "votes": votes, "k": k, "class_names": data["class_names"], "neighbors": neighbors}

    def delete(self, run_id=None, kind=None):
        with self.lock:
            if self.busy:
                raise HTTPException(409, "Detén la evaluación antes de borrar resultados")
            if run_id is not None:
                self.get_record(run_id)
            ids = [key for key, record in self.records.items() if (run_id is None or key == run_id) and (kind is None or record["kind"] == kind)]
            for key in ids:
                directory = self.output_dir / key
                if directory.exists():
                    shutil.rmtree(directory)
                del self.records[key]
                if self.cached_model is not None and self.cached_model[0] == key:
                    self.cached_model = None
            return len(ids)


def create_lab_router(lab):
    router = APIRouter(prefix="/api/lab")

    @router.get("/state")
    def state():
        return lab.snapshot()

    @router.post("/knn", status_code=202)
    def start_knn(settings: KNNSettings):
        return lab.start("knn", settings)

    @router.post("/compare", status_code=202)
    def start_comparison(settings: ComparisonSettings):
        return lab.start("comparison", settings)

    @router.post("/stop")
    def stop():
        with lab.lock:
            if lab.busy:
                lab.stop_event.set()
                lab.phase = "stopping"
        return lab.snapshot()

    @router.get("/runs")
    def runs(kind: Literal["knn", "comparison"] | None = None):
        return {"runs": lab.list_records(kind)}

    @router.delete("/runs")
    def delete_runs(kind: Literal["knn", "comparison"]):
        return {"deleted": lab.delete(kind=kind)}

    @router.get("/runs/{run_id}")
    def run(run_id: str):
        return lab.get_record(run_id)

    @router.delete("/runs/{run_id}")
    def delete_run(run_id: str):
        return {"deleted": lab.delete(run_id)}

    @router.get("/runs/{run_id}/download")
    def download(run_id: str):
        record = lab.get_record(run_id)
        return FileResponse(lab.output_dir / record["id"] / "run.json", filename=f"{record['kind']}-{record['id'][:8]}.json")

    @router.post("/predict")
    def predict(settings: PredictionSettings):
        if lab.busy:
            raise HTTPException(409, "Espera a que termine la evaluación para buscar vecinos")
        try:
            return lab.predict(settings)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return router
