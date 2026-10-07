const $ = (id) => document.getElementById(id);
const ui = {
  runId: null, cursor: 0, events: [], connected: false, hydrated: false,
  modelVersion: null, datasetReady: false, hasModel: false, busy: false,
  source: "dataset", file: null, predictionRequest: 0, lastSnapshot: 0,
  experimentRevision: -1, catalogLoading: false, experiments: [], selectedExperiments: [],
  selectionTouched: false, experimentDetails: new Map(), comparisonRequest: 0, lastPredictionArchitecture: null,
  catalogMutating: false, pendingDelete: null,
};
const percent = (value) => new Intl.NumberFormat("es-CO", { style: "percent", maximumFractionDigits: 2 }).format(value);
const decimal = (value) => new Intl.NumberFormat("es-CO", { minimumFractionDigits: 4, maximumFractionDigits: 4 }).format(value);

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, signal: AbortSignal.timeout(15000) });
  const data = await response.json();
  if (!response.ok) {
    const message = typeof data.detail === "string" ? data.detail : "Revisa los valores ingresados e inténtalo de nuevo.";
    throw new Error(message);
  }
  return data;
}

function showError(message) {
  $("error-banner").textContent = message;
  $("error-banner").hidden = !message;
}

function syncAlphaSlider() {
  const alpha = Number($("alpha").value);
  if (alpha > 0) $("alpha-slider").value = Math.log10(alpha);
  renderExperimentName();
}

function renderExperimentName() {
  const hidden = [$("hidden-neurons").value];
  if ($("second-hidden-enabled").checked) hidden.push($("second-hidden-neurons").value);
  $("experiment-name").textContent = `Alpha ${$("alpha").value} · ${hidden.join(" → ")} neuronas · ${$("iterations").value} iteraciones · semilla ${$("seed").value}`;
}

function drawChart(id, series, metric, total, width = 600, height = 158) {
  const chart = $(id);
  const left = 42, right = 12, top = 14, bottom = 28;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const values = series.flatMap((item) => item.events.filter((event) => Number.isFinite(event[item.key])).map((event) => event[item.key]));
  const largest = values.reduce((maximum, value) => Math.max(maximum, value), 0);
  const maxValue = metric === "accuracy" ? 1 : Math.max(1, Math.ceil(largest * 11) / 10);
  const maxStep = Math.max(1, total);
  let content = "";
  for (let tick = 0; tick <= 2; tick++) {
    const value = maxValue * tick / 2;
    const y = top + plotHeight * (1 - tick / 2);
    const label = metric === "accuracy" ? `${Math.round(value * 100)}%` : value.toFixed(1);
    content += `<line x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" stroke="#e4ebf4"/><text x="${left - 9}" y="${y + 3}" text-anchor="end" font-size="10" fill="#61728d">${label}</text>`;
  }
  for (let tick = 0; tick <= 4; tick++) {
    const x = left + plotWidth * tick / 4;
    content += `<text x="${x}" y="${height - 9}" text-anchor="middle" font-size="10" fill="#61728d">${Math.round(maxStep * tick / 4)}</text>`;
  }
  for (const item of series) {
    const points = item.events.filter((event) => Number.isFinite(event[item.key])).map((event) => [left + plotWidth * event.iteration / maxStep, top + plotHeight * (1 - event[item.key] / maxValue)]);
    if (!points.length) continue;
    content += `<polyline points="${points.map((point) => point.join(",")).join(" ")}" fill="none" stroke="${item.color}" stroke-width="2.5" ${item.dashed ? 'stroke-dasharray="6 4"' : ""} stroke-linecap="round" stroke-linejoin="round"/>`;
    const last = points.at(-1);
    content += `<circle cx="${last[0]}" cy="${last[1]}" r="3" fill="${item.color}"/>`;
  }
  if (!values.length) content += `<text x="${width / 2}" y="${height / 2}" text-anchor="middle" font-size="12" fill="#7b8ca7">${id === "comparison-chart" ? "Selecciona experimentos para comparar" : "La curva aparecerá al entrenar"}</text>`;
  chart.innerHTML = content;
}

function renderChart(id, key, color, total) {
  drawChart(id, [
    { events: ui.events, key, color, dashed: false },
    { events: ui.events, key: `validation_${key}`, color, dashed: true },
  ], key, total);
}

function setSelectValue(id, value) {
  if (!Array.from($(id).options).some((option) => option.value === String(value))) $(id).add(new Option(`${value} iteraciones`, value));
  $(id).value = value;
}

function prepareSettings(settings) {
  $("alpha").value = settings.alpha;
  $("iterations").value = settings.iterations;
  $("seed").value = settings.seed;
  $("hidden-neurons").value = settings.hidden_neurons ?? 10;
  $("second-hidden-enabled").checked = settings.second_hidden_enabled ?? false;
  $("second-hidden-neurons").value = settings.second_hidden_neurons ?? 16;
  setSelectValue("log-every", settings.log_every ?? 1);
  setSelectValue("validation-every", settings.validation_every ?? 50);
  syncAlphaSlider();
  renderArchitecture();
}

function renderArchitecture() {
  const second = $("second-hidden-enabled").checked;
  $("second-hidden-field").hidden = !second;
  $("second-hidden-neurons").disabled = !second;
  const hidden = [Number($("hidden-neurons").value) || 10];
  if (second) hidden.push(Number($("second-hidden-neurons").value) || 16);
  const sizes = [784, ...hidden, 10];
  const names = ["Píxeles", ...hidden.map((_, index) => `Oculta ${index + 1}`), "Dígitos"];
  const diagram = $("network-diagram");
  diagram.classList.toggle("two-hidden", second);
  diagram.setAttribute("aria-label", `Arquitectura seleccionada: ${sizes.join(" → ")}`);
  diagram.replaceChildren();
  sizes.forEach((size, index) => {
    if (index) {
      const arrow = document.createElement("span");
      arrow.textContent = "→";
      arrow.setAttribute("aria-hidden", "true");
      diagram.append(arrow);
    }
    const layer = document.createElement("div");
    const count = document.createElement("strong");
    count.textContent = size;
    const label = document.createElement("span");
    label.textContent = names[index];
    layer.append(count, label);
    diagram.append(layer);
  });
  const parameters = sizes.slice(1).reduce((sum, size, index) => sum + (sizes[index] + 1) * size, 0);
  $("parameter-count").textContent = `${new Intl.NumberFormat("es-CO").format(parameters)} parámetros`;
  renderExperimentName();
}
function refreshControls() {
  $("train-button").disabled = !ui.connected || !ui.datasetReady || ui.busy || ui.catalogMutating;
  $("settings-fields").disabled = ui.busy || ui.catalogMutating;
  $("stop-button").hidden = !ui.busy;
  for (const button of document.querySelectorAll(".reuse-settings")) button.disabled = ui.busy || ui.catalogMutating;
  for (const button of document.querySelectorAll(".delete-experiment")) button.disabled = ui.busy || ui.catalogMutating || !ui.connected || !ui.datasetReady;
  $("delete-all-experiments").disabled = ui.busy || ui.catalogMutating || !ui.connected || !ui.datasetReady || !ui.experiments.length;
  for (const id of ["predict-button", "random-button", "image-file", "prediction-model"]) $(id).disabled = !ui.connected || !ui.hasModel || !ui.datasetReady;
}

function renderState(data) {
  if (data.observed_at < ui.lastSnapshot) return;
  ui.lastSnapshot = data.observed_at;
  const changedRun = data.run_id !== ui.runId;
  if (changedRun) {
    ui.runId = data.run_id;
    ui.events = [];
    ui.cursor = 0;
    $("logs").replaceChildren();
  }
  ui.connected = true;
  ui.busy = ["training", "stopping"].includes(data.phase);
  ui.datasetReady = data.dataset_ready;
  ui.hasModel = data.has_model;
  if (!data.has_model && ui.modelVersion !== data.model_version) resetPrediction();
  if (!data.has_model) ui.modelVersion = data.model_version;
  if (!ui.hydrated) {
    prepareSettings(data.settings);
    ui.hydrated = true;
  }
  const logs = $("logs");
  const follow = changedRun || logs.scrollHeight - logs.scrollTop - logs.clientHeight < 45;
  if (data.events.length) {
    logs.querySelector(".log-empty")?.remove();
    const fragment = document.createDocumentFragment();
    for (const event of data.events) {
      if (ui.events.length && event.iteration <= ui.events.at(-1).iteration) continue;
      ui.events.push(event);
      const line = document.createElement("div");
      line.className = "log-line";
      line.textContent = event.message;
      fragment.append(line);
    }
    logs.append(fragment);
    if (follow) logs.scrollTop = logs.scrollHeight;
  }
  if (!ui.events.length && !logs.children.length) {
    const empty = document.createElement("p");
    empty.className = "log-empty";
    empty.textContent = "Los registros aparecerán aquí cuando inicies el entrenamiento.";
    logs.append(empty);
  }
  ui.cursor = data.cursor;
  const labels = { loading: "Cargando MNIST", ready: data.has_model ? "Modelo cargado" : "Listo para entrenar", training: "Entrenando", stopping: "Deteniendo", stopped: "Detenido", completed: "Completado", error: "Revisa el error" };
  $("run-status").textContent = labels[data.phase];
  $("run-status").className = `badge ${ui.busy ? "running" : data.has_model ? "complete" : ""}`;
  $("connection").className = "connection connected";
  $("connection").replaceChildren();
  const dot = document.createElement("span");
  dot.className = "status-dot";
  $("connection").append(dot, "Conectado a tu PC");
  const latest = data.latest;
  const progress = latest ? Math.min(100, latest.iteration / data.settings.iterations * 100) : 0;
  $("progress-fill").style.width = `${progress}%`;
  $("progress").setAttribute("aria-valuenow", Math.round(progress));
  $("iteration-count").textContent = latest ? `Iteración ${latest.iteration} / ${data.settings.iterations} · ${data.architecture.join(" → ")}` : "Sin entrenamiento";
  $("accuracy-value").textContent = latest ? percent(latest.accuracy) : "—";
  $("loss-value").textContent = latest ? decimal(latest.loss) : "—";
  $("held-out-value").textContent = data.held_out_accuracy !== null ? percent(data.held_out_accuracy) : "—";
  $("validation-loss-value").textContent = data.validation_loss !== null ? decimal(data.validation_loss) : "—";
  $("validation-note").textContent = data.validation_iteration !== null ? `Última validación: iteración ${data.validation_iteration}. Se mide cada ${data.settings.validation_every} iteraciones y al finalizar.` : "La validación se mide al inicio, a intervalos y al finalizar.";
  $("elapsed-value").textContent = data.elapsed ? `${data.elapsed.toFixed(1)} s` : "—";
  $("log-count").textContent = `${data.cursor} registros`;
  document.querySelector(".logs-panel").classList.toggle("running", ui.busy);
  $("model-status").textContent = data.has_model ? (ui.lastPredictionArchitecture || data.model_architecture).join(" → ") : "Sin modelo";
  if (data.experiment_revision !== ui.experimentRevision && !ui.catalogLoading) refreshExperiments();
  $("form-message").textContent = ui.busy ? `Entrenando con alpha ${data.settings.alpha}. Puedes detenerlo y empezar con otros valores.` : data.dataset_ready ? "Los cambios se aplican al iniciar un nuevo entrenamiento, desde cero." : "Cargando el dataset…";
  $("prediction-note").textContent = ui.busy ? "Mientras entrena, las predicciones usan el último modelo terminado." : "Las imágenes propias se convierten a 28 × 28 píxeles. La red fue entrenada con dígitos manuscritos.";
  if (data.error) showError(data.error);
  if (changedRun || data.events.length) {
    renderChart("accuracy-chart", "accuracy", "#173fbd", data.settings.iterations);
    renderChart("loss-chart", "loss", "#b6670b", data.settings.iterations);
  }
  refreshControls();
  if (data.has_model && data.dataset_ready && ui.modelVersion !== data.model_version) {
    ui.modelVersion = data.model_version;
    refreshPrediction();
  }
}

async function poll() {
  try {
    const data = await api(`/api/state?since=${ui.cursor}&run_id=${encodeURIComponent(ui.runId || "")}`);
    renderState(data);
  } catch (error) {
    ui.connected = false;
    $("connection").className = "connection disconnected";
    $("connection").textContent = "Reconectando…";
    $("form-message").textContent = "No hay conexión. Mantén abierto el servidor local; intentaremos reconectar.";
    refreshControls();
  }
  setTimeout(poll, ui.connected ? 500 : 1500);
}

function renderPrediction(data) {
  const context = $("digit-canvas").getContext("2d");
  const image = context.createImageData(28, 28);
  data.pixels.flat().forEach((value, index) => {
    image.data.set([value, value, value, 255], index * 4);
  });
  context.putImageData(image, 0, 0);
  ui.lastPredictionArchitecture = data.architecture;
  $("model-status").textContent = data.architecture.join(" → ");
  $("predicted-digit").textContent = data.prediction;
  $("confidence-value").textContent = `${percent(data.confidence)} de probabilidad`;
  $("image-caption").textContent = data.index === null ? "Tu imagen, normalizada a 28 × 28" : `MNIST · imagen ${data.index}`;
  const expected = $("expected-label");
  expected.className = "expected-label";
  if (data.label !== null) {
    expected.textContent = `${data.prediction === data.label ? "Acertó" : "No acertó"} · Etiqueta real: ${data.label}`;
    expected.classList.add(data.prediction === data.label ? "correct" : "incorrect");
  } else expected.textContent = "Imagen propia · etiqueta desconocida";
  data.probabilities.forEach((value, digit) => {
    const column = $("probability-bars").children[digit];
    column.classList.toggle("winner", digit === data.prediction);
    column.querySelector(".bar-fill").style.height = `${value * 100}%`;
    column.querySelector(".bar-track").style.setProperty("--probability", `${Math.min(85, value * 100)}%`);
    column.querySelector(".bar-value").textContent = `${Math.round(value * 100)}%`;
    column.title = `${digit}: ${percent(value)}`;
  });
  $("probability-bars").setAttribute("aria-label", data.probabilities.map((value, digit) => `Dígito ${digit}: ${percent(value)}`).join(", "));
}

function resetPrediction() {
  ui.predictionRequest += 1;
  ui.lastPredictionArchitecture = null;
  $("digit-canvas").getContext("2d").clearRect(0, 0, 28, 28);
  $("image-caption").textContent = "Esperando imagen";
  $("predicted-digit").textContent = "?";
  $("confidence-value").textContent = "Sin predicción";
  $("expected-label").textContent = "Entrena un modelo para empezar.";
  $("expected-label").className = "expected-label";
  for (const column of $("probability-bars").children) {
    column.classList.remove("winner");
    column.querySelector(".bar-fill").style.height = "0%";
    column.querySelector(".bar-track").style.setProperty("--probability", "0%");
    column.querySelector(".bar-value").textContent = "0%";
    column.removeAttribute("title");
  }
  $("probability-bars").setAttribute("aria-label", "Probabilidades para los dígitos del cero al nueve");
}

async function requestPrediction(path, options) {
  if (!ui.hasModel || !ui.datasetReady) return;
  const request = ++ui.predictionRequest;
  try {
    const url = new URL(path, window.location.origin);
    if ($("prediction-model").value) url.searchParams.set("experiment_id", $("prediction-model").value);
    const data = await api(url.pathname + url.search, options);
    if (request !== ui.predictionRequest) return;
    renderPrediction(data);
    if (data.index !== null) $("image-index").value = data.index;
  } catch (error) {
    if (request === ui.predictionRequest) showError(error.message);
  }
}

function refreshPrediction() {
  if (ui.source === "upload") {
    if (ui.file) requestPrediction("/api/predict-image", { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: ui.file });
  } else if ($("image-index").reportValidity()) {
    requestPrediction(`/api/predict?index=${Number($("image-index").value)}`);
  }
}

function selectSource(source) {
  ui.source = source;
  for (const mode of ["dataset", "upload"]) {
    $(mode + "-tab").classList.toggle("active", mode === source);
    $(mode + "-tab").setAttribute("aria-selected", mode === source);
    $(mode + "-controls").hidden = mode !== source;
  }
  if (source === "upload" && !ui.file) {
    ui.predictionRequest++;
    $("digit-canvas").getContext("2d").clearRect(0, 0, 28, 28);
    $("predicted-digit").textContent = "?";
    $("confidence-value").textContent = "Selecciona una imagen";
    $("image-caption").textContent = "Tu imagen aparecerá aquí";
    $("expected-label").className = "expected-label";
    $("expected-label").textContent = "Un dígito manuscrito sobre fondo uniforme";
    for (const column of $("probability-bars").children) {
      column.classList.remove("winner");
      column.querySelector(".bar-fill").style.height = "0%";
      column.querySelector(".bar-value").textContent = "0%";
      column.querySelector(".bar-track").style.setProperty("--probability", "0%");
    }
    $("probability-bars").setAttribute("aria-label", "Sin predicción; selecciona una imagen");
  }
  refreshPrediction();
}

$("training-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");
  $("train-button").disabled = true;
  const settings = {
    iterations: Number($("iterations").value), alpha: Number($("alpha").value), seed: Number($("seed").value),
    log_every: Number($("log-every").value), hidden_neurons: Number($("hidden-neurons").value),
    second_hidden_enabled: $("second-hidden-enabled").checked, second_hidden_neurons: Number($("second-hidden-neurons").value),
    validation_every: Number($("validation-every").value),
  };
  try {
    const data = await api("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settings) });
    renderState(data);
  } catch (error) {
    showError(error.message);
    refreshControls();
  }
});
$("stop-button").addEventListener("click", async () => {
  $("stop-button").disabled = true;
  try { renderState(await api("/api/stop", { method: "POST" })); }
  catch (error) { showError(error.message); }
  finally { $("stop-button").disabled = false; }
});
$("alpha").addEventListener("input", syncAlphaSlider);
$("alpha-slider").addEventListener("input", () => { $("alpha").value = Number((10 ** Number($("alpha-slider").value)).toPrecision(3)); renderExperimentName(); });
for (const id of ["iterations", "seed"]) $(id).addEventListener("input", renderExperimentName);
$("predict-button").addEventListener("click", () => { showError(""); refreshPrediction(); });
$("image-index").addEventListener("change", refreshPrediction);
$("random-button").addEventListener("click", () => { showError(""); const digit = $("digit-filter").value; requestPrediction(`/api/sample${digit === "" ? "" : `?digit=${digit}`}`); });
$("dataset-tab").addEventListener("click", () => selectSource("dataset"));
$("upload-tab").addEventListener("click", () => selectSource("upload"));
for (const source of ["dataset", "upload"]) {
  $(source + "-tab").addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const next = source === "dataset" ? "upload" : "dataset";
    selectSource(next);
    $(next + "-tab").focus();
  });
}
$("image-file").addEventListener("change", () => {
  const file = $("image-file").files[0];
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) { showError("La imagen debe pesar menos de 5 MB."); return; }
  ui.file = file;
  showError("");
  refreshPrediction();
});
for (let digit = 0; digit < 10; digit++) {
  const column = document.createElement("div");
  column.className = "probability-column";
  column.innerHTML = `<div class="bar-track"><div class="bar-fill"></div><span class="bar-value">0%</span></div><span>${digit}</span>`;
  $("probability-bars").append(column);
}

const experimentColors = ["#173fbd", "#087a69", "#aa5a14", "#8058a6"];

async function refreshExperiments() {
  ui.catalogLoading = true;
  try {
    const data = await api("/api/experiments");
    ui.experiments = data.experiments;
    ui.experimentRevision = data.revision;
    for (const id of ui.experimentDetails.keys()) if (!ui.experiments.some((record) => record.id === id)) ui.experimentDetails.delete(id);
    if (!ui.selectionTouched) ui.selectedExperiments = ui.experiments.slice(0, 2).map((record) => record.id);
    else ui.selectedExperiments = ui.selectedExperiments.filter((id) => ui.experiments.some((record) => record.id === id));
    const picker = $("prediction-model");
    const selectedModel = picker.value;
    picker.replaceChildren(new Option("Último entrenamiento", ""));
    for (const record of ui.experiments) picker.add(new Option(`${record.name} (${record.architecture.join(" → ")})`, record.id));
    picker.value = ui.experiments.some((record) => record.id === selectedModel) ? selectedModel : "";
    if (selectedModel && !picker.value) {
      ui.predictionRequest += 1;
      refreshPrediction();
    }
    $("experiment-count").textContent = `${ui.experiments.length} ${ui.experiments.length === 1 ? "experimento" : "experimentos"}`;
    renderExperimentRows();
    refreshControls();
    await renderComparison();
  } catch (error) {
    showError(`No se pudieron cargar los experimentos: ${error.message}`);
  } finally {
    ui.catalogLoading = false;
  }
}

function renderExperimentRows() {
  const body = $("experiment-rows");
  body.replaceChildren();
  if (!ui.experiments.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 11;
    cell.className = "empty-experiments";
    cell.textContent = "Entrena tu primera configuración para comenzar a comparar.";
    return;
  }
  for (const record of ui.experiments) {
    const row = body.insertRow();
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `compare-${record.id}`;
    checkbox.checked = ui.selectedExperiments.includes(record.id);
    checkbox.disabled = ui.selectedExperiments.length >= 4 && !checkbox.checked;
    checkbox.setAttribute("aria-label", `Comparar ${record.name}`);
    checkbox.addEventListener("change", () => {
      ui.selectionTouched = true;
      if (checkbox.checked) ui.selectedExperiments.push(record.id);
      else ui.selectedExperiments = ui.selectedExperiments.filter((id) => id !== record.id);
      renderExperimentRows();
      $(checkbox.id)?.focus();
      renderComparison();
    });
    row.insertCell().append(checkbox);
    const nameCell = row.insertCell();
    const name = document.createElement("span");
    name.className = "experiment-name";
    name.textContent = record.name;
    const detail = document.createElement("span");
    detail.className = "experiment-detail";
    detail.textContent = record.architecture.join(" → ");
    const state = document.createElement("span");
    state.className = "experiment-state";
    const status = { completed: "Completo", stopped: "Detenido", imported: "Modelo anterior" };
    state.textContent = `${status[record.status] || record.status} · ${new Intl.DateTimeFormat("es-CO", { hour: "2-digit", minute: "2-digit", timeZone: "America/Bogota" }).format(new Date(record.created_at))}`;
    const selectedIndex = ui.selectedExperiments.indexOf(record.id);
    if (selectedIndex >= 0) {
      const swatch = document.createElement("i");
      swatch.className = "experiment-swatch";
      swatch.style.background = experimentColors[selectedIndex];
      name.prepend(swatch, " ");
    }
    nameCell.append(name, detail, state);
    const values = [record.settings.alpha, `${record.actual_iterations} / ${record.settings.iterations}`, record.settings.seed,
      new Intl.NumberFormat("es-CO").format(record.parameter_count), percent(record.accuracy),
      record.validation_accuracy === null ? "—" : percent(record.validation_accuracy),
      record.validation_loss === null ? "—" : decimal(record.validation_loss),
      record.duration === null ? "—" : `${record.duration.toFixed(1)} s`];
    for (const value of values) row.insertCell().textContent = value;
    const reuse = document.createElement("button");
    reuse.type = "button";
    reuse.className = "button secondary compact reuse-settings";
    reuse.textContent = "Usar valores";
    reuse.disabled = ui.busy || ui.catalogMutating;
    reuse.setAttribute("aria-label", `Usar configuración de ${record.name}`);
    reuse.addEventListener("click", () => {
      prepareSettings(record.settings);
      $("experiment-preset").value = "custom";
      $("form-message").textContent = "Configuración cargada. Cambia una variable y pulsa Iniciar entrenamiento para comparar.";
      $("alpha").focus();
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "button danger compact delete-experiment";
    remove.textContent = "Borrar";
    remove.disabled = ui.busy || ui.catalogMutating || !ui.connected || !ui.datasetReady;
    remove.setAttribute("aria-label", `Borrar experimento ${record.name}`);
    remove.addEventListener("click", () => openDeleteDialog(record.id));
    const actions = document.createElement("div");
    actions.className = "experiment-actions";
    actions.append(reuse, remove);
    row.insertCell().append(actions);
  }
}

async function renderComparison() {
  const request = ++ui.comparisonRequest;
  try {
    await Promise.all(ui.selectedExperiments.map(async (id) => {
      if (!ui.experimentDetails.has(id)) ui.experimentDetails.set(id, await api(`/api/experiments/${encodeURIComponent(id)}`));
    }));
    if (request !== ui.comparisonRequest) return;
    const metric = $("comparison-metric").value;
    const series = [];
    let total = 1;
    $("comparison-legend").replaceChildren();
    ui.selectedExperiments.forEach((id, index) => {
      const record = ui.experimentDetails.get(id);
      if (!record) return;
      const color = experimentColors[index];
      series.push({ events: record.history, key: metric, color, dashed: false });
      series.push({ events: record.history, key: `validation_${metric}`, color, dashed: true });
      total = Math.max(total, record.actual_iterations);
      const legend = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.className = "experiment-swatch";
      swatch.style.background = color;
      legend.append(swatch, record.name);
      $("comparison-legend").append(legend);
    });
    drawChart("comparison-chart", series, metric, total, 900, 230);
  } catch (error) {
    if (request === ui.comparisonRequest) showError(`No se pudieron comparar las curvas: ${error.message}`);
  }
}

const experimentPresets = {
  baseline: { alpha: 0.5, hidden_neurons: 10 },
  "small-alpha": { alpha: 0.1, hidden_neurons: 10 },
  "large-alpha": { alpha: 1, hidden_neurons: 10 },
  wide: { alpha: 0.5, hidden_neurons: 32 },
  wider: { alpha: 0.5, hidden_neurons: 64 },
  deep: { alpha: 0.5, hidden_neurons: 32, second_hidden_enabled: true },
};
$("experiment-preset").addEventListener("change", () => {
  const preset = experimentPresets[$("experiment-preset").value];
  if (preset) prepareSettings({ iterations: 500, seed: 0, log_every: 1, validation_every: 50, second_hidden_enabled: false, second_hidden_neurons: 16, ...preset });
});
for (const id of ["hidden-neurons", "second-hidden-neurons", "second-hidden-enabled"]) $(id).addEventListener("input", renderArchitecture);
$("comparison-metric").addEventListener("change", renderComparison);
$("prediction-model").addEventListener("change", () => { showError(""); refreshPrediction(); });

function openDeleteDialog(id = null) {
  if (ui.busy || ui.catalogMutating) return;
  const record = id === null ? null : ui.experiments.find((item) => item.id === id);
  if (id !== null && !record) return;
  ui.pendingDelete = { id };
  $("delete-dialog-title").textContent = record ? "Borrar experimento" : "Borrar todos los experimentos";
  $("delete-dialog-description").textContent = record
    ? `Se borrarán «${record.name}», sus pesos, métricas y logs. Si es el modelo actual, se usará el experimento más reciente que quede disponible.`
    : `Se borrarán los ${ui.experiments.length} experimentos guardados, sus modelos, métricas y logs. La página quedará lista para entrenar desde cero.`;
  $("delete-dialog-confirm").textContent = record ? "Borrar experimento" : "Borrar todos";
  $("delete-dialog").showModal();
}

$("delete-all-experiments").addEventListener("click", () => openDeleteDialog());
$("delete-dialog-cancel").addEventListener("click", () => $("delete-dialog").close());
$("delete-dialog").addEventListener("cancel", (event) => { if (ui.catalogMutating) event.preventDefault(); });
$("delete-dialog").addEventListener("close", () => { ui.pendingDelete = null; });
$("delete-dialog-confirm").addEventListener("click", async () => {
  const pending = ui.pendingDelete;
  if (!pending || ui.catalogMutating) return;
  ui.catalogMutating = true;
  $("delete-dialog-confirm").disabled = true;
  $("delete-dialog-cancel").disabled = true;
  refreshControls();
  try {
    const path = pending.id === null ? "/api/experiments" : `/api/experiments/${encodeURIComponent(pending.id)}`;
    const data = await api(path, { method: "DELETE" });
    if (data.state.run_id !== ui.runId) ui.hydrated = false;
    renderState(data.state);
    await refreshExperiments();
    $("delete-dialog").close();
    showError("");
  } catch (error) {
    $("delete-dialog").close();
    showError(`No se pudieron borrar los experimentos: ${error.message}`);
  } finally {
    ui.catalogMutating = false;
    $("delete-dialog-confirm").disabled = false;
    $("delete-dialog-cancel").disabled = false;
    refreshControls();
  }
});

poll();
