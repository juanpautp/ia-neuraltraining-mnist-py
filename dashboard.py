import argparse
import io
import json
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from mnist import PROJECT_DIR, load_dataset
from network import architecture, evaluate_network, forward_network, load_network, parameter_count, save_network, train_network


class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    iterations: int = Field(default=500, ge=1, le=5000)
    alpha: float = Field(default=0.5, gt=0, le=10)
    seed: int = Field(default=0, ge=0)
    log_every: int = Field(default=1, ge=1, le=5000)
    hidden_neurons: int = Field(default=10, ge=1, le=256)
    second_hidden_enabled: bool = False
    second_hidden_neurons: int = Field(default=16, ge=1, le=256)
    validation_every: int = Field(default=50, ge=1, le=5000)
    experiment_name: str = Field(default="", max_length=60)

    @property
    def hidden_layers(self):
        return (self.hidden_neurons, self.second_hidden_neurons) if self.second_hidden_enabled else (self.hidden_neurons,)

    @property
    def architecture(self):
        return [784, *self.hidden_layers, 10]

    @property
    def parameter_count(self):
        sizes = self.architecture
        return sum((input_size + 1) * output_size for input_size, output_size in zip(sizes, sizes[1:]))

    def display_name(self, experiment_id, status="completed"):
        hidden = " → ".join(map(str, self.hidden_layers))
        prefix = "Modelo anterior · " if status == "imported" else ""
        return f"{prefix}Alpha {self.alpha:g} · {hidden} neuronas · {self.iterations} iteraciones · semilla {self.seed} · #{experiment_id[:8]}"


class TrainingManager:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir or PROJECT_DIR / "outputs" / "web")
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.dataset = None
        self.params = None
        self.phase = "loading"
        self.error = None
        self.events = []
        self.settings = TrainingSettings()
        self.run_id = "saved"
        self.model_version = 0
        self.model_experiment_id = None
        self.held_out_accuracy = None
        self.validation_loss = None
        self.validation_iteration = None
        self.started_at = None
        self.duration = 0
        self.experiments = {}
        self.experiment_models = {}
        self.experiment_revision = 0
        for path in sorted((self.output_dir / "experiments").glob("*/run.json")):
            try:
                record = json.loads(path.read_text())
                if record["id"] == path.parent.name:
                    settings = TrainingSettings(**record["settings"])
                    name = settings.display_name(record["id"], record["status"])
                    if record["name"] != name:
                        record["name"] = name
                        temporary = path.with_name("run.tmp.json")
                        temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
                        temporary.replace(path)
                    self.experiments[record["id"]] = record
            except (OSError, ValueError, KeyError):
                continue
        self.experiment_revision = len(self.experiments)
        model_path = self.output_dir / "model.npz"
        if not model_path.exists() and not (self.output_dir / "reset.json").exists():
            model_path = PROJECT_DIR / "outputs" / "model.npz"
        if model_path.exists():
            try:
                self.params = load_network(model_path)
                self.model_version = 1
                sizes = architecture(self.params)
                metadata_path = model_path.parent / "run.json"
                metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
                self.settings = TrainingSettings(**metadata.get("settings", {}))
                self.settings.hidden_neurons = sizes[1]
                self.settings.second_hidden_enabled = len(sizes) == 4
                if len(sizes) == 4:
                    self.settings.second_hidden_neurons = sizes[2]
                self.model_experiment_id = metadata.get("experiment_id")
                existing = self.experiments.get(self.model_experiment_id)
                if existing:
                    self.run_id = existing["id"]
                    self.events = [dict(event) for event in existing["history"]]
                    self.duration = existing.get("duration") or 0
                else:
                    history_path = model_path.parent / "history.csv"
                    if history_path.exists():
                        for values in np.atleast_2d(np.loadtxt(history_path, delimiter=",", skiprows=1)):
                            self.record({"iteration": int(values[0]), "accuracy": float(values[1]), "loss": float(values[2]),
                                         "validation_accuracy": None, "validation_loss": None})
            except (OSError, ValueError, KeyError) as error:
                self.params = None
                self.model_version = 0
                self.events = []
                self.error = f"No se pudo cargar el modelo guardado: {error}. Puedes entrenar uno nuevo."

    def prepare(self):
        try:
            dataset = load_dataset(PROJECT_DIR / "data")
            accuracy, loss = (None, None)
            if self.params is not None:
                accuracy, loss = evaluate_network(self.params, dataset[2], dataset[3])
            if self.params is not None and self.events and not self.experiments:
                self.events[-1].update(validation_accuracy=accuracy, validation_loss=loss)
                record = self.build_record("baseline", self.settings, self.params, self.events, "imported", None)
                self.save_experiment(record, self.params)
                self.save_current_model(record, self.params)
                self.model_experiment_id = "baseline"
            with self.lock:
                self.dataset = dataset
                self.held_out_accuracy = accuracy
                self.validation_loss = loss
                self.validation_iteration = self.events[-1]["iteration"] if self.events else None
                self.phase = "ready"
        except Exception as error:
            with self.lock:
                self.phase = "error"
                self.error = f"No se pudo preparar el laboratorio: {error}. Revisa la conexión y reinicia la página local."

    def record(self, values):
        event = dict(values)
        event["message"] = f"Iteration {event['iteration']}/{self.settings.iterations}: accuracy={event['accuracy']:.2%}, loss={event['loss']:.4f}"
        if event.get("validation_accuracy") is not None:
            event["message"] += f" | val_accuracy={event['validation_accuracy']:.2%}, val_loss={event['validation_loss']:.4f}"
        with self.lock:
            self.events.append(event)
            if event.get("validation_accuracy") is not None:
                self.held_out_accuracy = event["validation_accuracy"]
                self.validation_loss = event["validation_loss"]
                self.validation_iteration = event["iteration"]

    def snapshot(self, since=0, run_id=None):
        with self.lock:
            cursor = since if run_id == self.run_id and since <= len(self.events) else 0
            return {
                "phase": self.phase, "observed_at": time.time(), "error": self.error,
                "dataset_ready": self.dataset is not None, "has_model": self.params is not None,
                "model_version": self.model_version, "model_experiment_id": self.model_experiment_id,
                "model_architecture": architecture(self.params) if self.params is not None else None,
                "run_id": self.run_id, "settings": self.settings.model_dump(),
                "architecture": self.settings.architecture, "parameter_count": self.settings.parameter_count,
                "latest": dict(self.events[-1]) if self.events else None,
                "events": [dict(event) for event in self.events[cursor:]], "cursor": len(self.events),
                "held_out_accuracy": self.held_out_accuracy, "validation_loss": self.validation_loss,
                "validation_iteration": self.validation_iteration, "experiment_revision": self.experiment_revision,
                "elapsed": time.monotonic() - self.started_at if self.phase in ("training", "stopping") else self.duration,
            }

    def start(self, settings):
        with self.lock:
            if self.dataset is None:
                raise HTTPException(409, "MNIST todavía no está disponible. Espera a que termine de cargar.")
            if self.phase in ("training", "stopping"):
                raise HTTPException(409, "Ya hay un entrenamiento en curso.")
            self.settings = settings
            self.events = []
            self.run_id = uuid.uuid4().hex
            self.phase = "training"
            self.error = None
            self.held_out_accuracy = None
            self.validation_loss = None
            self.validation_iteration = None
            self.stop_event.clear()
            self.started_at = time.monotonic()
        threading.Thread(target=self.train, args=(settings, self.run_id), daemon=True).start()

    def build_record(self, experiment_id, settings, params, history, status, duration):
        last = history[-1]
        sizes = architecture(params)
        return {
            "id": experiment_id,
            "name": settings.display_name(experiment_id, status),
            "settings": settings.model_dump(), "architecture": sizes, "parameter_count": parameter_count(params),
            "actual_iterations": last["iteration"], "accuracy": last["accuracy"], "loss": last["loss"],
            "validation_accuracy": last.get("validation_accuracy"), "validation_loss": last.get("validation_loss"),
            "duration": duration, "status": status, "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "mnist-train-first-50000-validation-last-10000", "activation": "sigmoid", "initialization_scale": 0.01,
            "history": [dict(event) for event in history],
        }

    def save_experiment(self, record, params):
        directory = self.output_dir / "experiments" / record["id"]
        directory.mkdir(parents=True, exist_ok=True)
        save_network(directory / "model.npz", params)
        temporary = directory / "run.tmp.json"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
        temporary.replace(directory / "run.json")
        with self.lock:
            self.experiments[record["id"]] = record
            self.experiment_models[record["id"]] = params
            self.experiment_revision += 1

    def train(self, settings, experiment_id):
        try:
            X_train, Y_train, X_validation, Y_validation, _, _ = self.dataset
            params, history = train_network(
                X_train, Y_train, settings.iterations, settings.alpha, settings.seed, settings.log_every,
                hidden_layers=settings.hidden_layers, validation=(X_validation, Y_validation),
                validation_every=settings.validation_every, on_progress=self.record, should_stop=self.stop_event.is_set,
            )
            duration = time.monotonic() - self.started_at
            status = "stopped" if self.stop_event.is_set() else "completed"
            record = self.build_record(experiment_id, settings, params, self.events, status, duration)
            self.save_experiment(record, params)
            self.save_current_model(record, params)
            with self.lock:
                self.params = params
                self.model_experiment_id = experiment_id
                self.model_version += 1
                self.held_out_accuracy = record["validation_accuracy"]
                self.validation_loss = record["validation_loss"]
                self.phase = status
                self.duration = duration
        except Exception as error:
            with self.lock:
                self.phase = "error"
                self.error = f"El entrenamiento falló: {error}. Revisa los valores y prueba un alpha menor."
                self.duration = time.monotonic() - self.started_at

    def save_current_model(self, record, params):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_network(self.output_dir / "model.npz", params)
        values = [[event["iteration"], event["accuracy"], event["loss"], event.get("validation_accuracy") if event.get("validation_accuracy") is not None else np.nan,
                   event.get("validation_loss") if event.get("validation_loss") is not None else np.nan] for event in record["history"]]
        np.savetxt(self.output_dir / "history.csv", values, delimiter=",", header="iteration,accuracy,loss,validation_accuracy,validation_loss", comments="")
        metadata = {"experiment_id": record["id"], "settings": record["settings"], "held_out_accuracy": record["validation_accuracy"], "stopped": record["status"] == "stopped"}
        temporary = self.output_dir / "run.tmp.json"
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False))
        temporary.replace(self.output_dir / "run.json")

    def delete_experiments(self, experiment_id=None):
        with self.lock:
            if self.phase in ("loading", "training", "stopping"):
                raise HTTPException(409, "Espera a que termine la carga o detén el entrenamiento antes de borrar experimentos.")
            if experiment_id is not None and experiment_id not in self.experiments:
                raise HTTPException(404, "No se encontró ese experimento.")
            deleted = [experiment_id] if experiment_id is not None else list(self.experiments)
            remaining = [record for key, record in self.experiments.items() if key not in deleted]
            replace_current = experiment_id is None or self.model_experiment_id in deleted or not remaining
            replacement = max(remaining, key=lambda record: record["created_at"]) if remaining else None
            params = None
            if replace_current:
                if replacement:
                    params = self.experiment_models.get(replacement["id"])
                    if params is None:
                        params = load_network(self.output_dir / "experiments" / replacement["id"] / "model.npz")
                    self.save_current_model(replacement, params)
                else:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    (self.output_dir / "reset.json").write_text(json.dumps({"cleared_at": datetime.now(timezone.utc).isoformat()}))
                    for name in ("model.npz", "history.csv", "run.json", "model.npz.tmp.npz", "run.tmp.json"):
                        (self.output_dir / name).unlink(missing_ok=True)
            for key in deleted:
                directory = self.output_dir / "experiments" / key
                if directory.exists():
                    shutil.rmtree(directory)
                self.experiments.pop(key)
                self.experiment_models.pop(key, None)
            self.experiment_revision += 1
            if replace_current:
                self.params = params
                self.model_experiment_id = replacement["id"] if replacement else None
                self.model_version += 1
                self.events = [dict(event) for event in replacement["history"]] if replacement else []
                self.settings = TrainingSettings(**replacement["settings"]) if replacement else TrainingSettings()
                self.run_id = replacement["id"] if replacement else uuid.uuid4().hex
                self.held_out_accuracy = replacement["validation_accuracy"] if replacement else None
                self.validation_loss = replacement["validation_loss"] if replacement else None
                self.validation_iteration = replacement["actual_iterations"] if replacement else None
                self.duration = (replacement.get("duration") or 0) if replacement else 0
                self.started_at = None
                self.phase = "ready"
                self.error = None
            return len(deleted)

    def list_experiments(self):
        with self.lock:
            records = sorted(self.experiments.values(), key=lambda record: record["created_at"], reverse=True)
            return [{key: value for key, value in record.items() if key != "history"} for record in records]

    def get_experiment(self, experiment_id):
        with self.lock:
            record = self.experiments.get(experiment_id)
        if record is None:
            raise HTTPException(404, "No se encontró ese experimento.")
        return record

    def predict(self, image, label=None, index=None, experiment_id=None):
        with self.lock:
            params = self.params
            version = self.model_version
            model_id = self.model_experiment_id
            if experiment_id is not None:
                if experiment_id not in self.experiments:
                    raise HTTPException(404, "No se encontró ese experimento.")
                params = self.experiment_models.get(experiment_id)
                if params is None:
                    params = load_network(self.output_dir / "experiments" / experiment_id / "model.npz")
                    self.experiment_models[experiment_id] = params
                model_id = experiment_id
        if params is None:
            raise HTTPException(409, "Entrena un modelo antes de probar imágenes.")
        X = np.asarray(image, dtype=np.float32).reshape(784, 1) / 255.0
        probabilities = forward_network(params, X)[-1][:, 0]
        prediction = int(np.argmax(probabilities))
        return {
            "pixels": np.asarray(image, dtype=np.uint8).tolist(), "probabilities": probabilities.tolist(),
            "prediction": prediction, "confidence": float(probabilities[prediction]),
            "label": label, "index": index, "model_version": version,
            "architecture": architecture(params), "experiment_id": model_id,
        }

manager = TrainingManager()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=manager.prepare, daemon=True).start()
    yield
    manager.stop_event.set()


app = FastAPI(title="MNIST Lab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PROJECT_DIR / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(PROJECT_DIR / "static" / "index.html")


@app.get("/api/state")
def state(since: int = Query(default=0, ge=0), run_id: str | None = None):
    return manager.snapshot(since, run_id)


@app.post("/api/train", status_code=202)
def start_training(settings: TrainingSettings):
    manager.start(settings)
    return manager.snapshot()


@app.get("/api/experiments")
def experiments():
    return {"experiments": manager.list_experiments(), "revision": manager.experiment_revision}


@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: str):
    return manager.get_experiment(experiment_id)


@app.delete("/api/experiments")
def delete_all_experiments():
    deleted = manager.delete_experiments()
    return {"deleted": deleted, "state": manager.snapshot()}


@app.delete("/api/experiments/{experiment_id}")
def delete_experiment(experiment_id: str):
    deleted = manager.delete_experiments(experiment_id)
    return {"deleted": deleted, "state": manager.snapshot()}


@app.post("/api/stop")
def stop_training():
    with manager.lock:
        if manager.phase in ("training", "stopping"):
            manager.stop_event.set()
            manager.phase = "stopping"
    return manager.snapshot()


@app.get("/api/predict")
def predict(index: int = Query(default=50000, ge=0, lt=60000), experiment_id: str | None = None):
    if manager.dataset is None:
        raise HTTPException(409, "MNIST todavía está cargando.")
    images, labels = manager.dataset[4:]
    return manager.predict(images[index], int(labels[index]), index, experiment_id)


@app.get("/api/sample")
def sample(digit: int | None = Query(default=None, ge=0, le=9), experiment_id: str | None = None):
    if manager.dataset is None:
        raise HTTPException(409, "MNIST todavía está cargando.")
    images, labels = manager.dataset[4:]
    candidates = np.arange(50000, 60000)
    if digit is not None:
        candidates = candidates[labels[candidates] == digit]
    index = int(np.random.default_rng().choice(candidates))
    return manager.predict(images[index], int(labels[index]), index, experiment_id)


@app.post("/api/predict-image")
async def predict_image(request: Request, experiment_id: str | None = None):
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(413, "La imagen debe pesar menos de 5 MB.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in ("PNG", "JPEG", "WEBP") or source.width * source.height > 16_000_000:
                raise ValueError("Usa una imagen PNG, JPG o WebP de hasta 16 megapíxeles.")
            source = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", source.size, "white")
            image = Image.alpha_composite(background, source).convert("L")
            pixels = np.asarray(image)
            border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
            if border.mean() > 127:
                image = ImageOps.invert(image)
            mask = np.asarray(image) > max(20, np.asarray(image).max() * 0.2)
            rows, columns = np.where(mask)
            if not len(rows):
                raise ValueError("No se encontró un dígito. Usa un número con contraste sobre un fondo uniforme.")
            image = image.crop((int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1))
            image.thumbnail((20, 20), Image.Resampling.LANCZOS)
            normalized = Image.new("L", (28, 28), 0)
            normalized.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
            return manager.predict(np.asarray(normalized), experiment_id=experiment_id)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(422, f"No se pudo leer la imagen: {error}") from error


def main():
    parser = argparse.ArgumentParser(description="Open the local MNIST training dashboard.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port} in your browser", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
