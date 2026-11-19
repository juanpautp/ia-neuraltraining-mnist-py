const byId = (id) => document.getElementById(id);
const moduleKind = document.body.dataset.module;
const labUI = { records: [], selected: null, busy: false, connected: false, ready: false, networkBusy: false, networks: [], knnRuns: [], handledRun: null, detailRequest: 0, catalogSignature: "", recordSignature: "", polling: false, ticks: 0 };
const number = (value) => new Intl.NumberFormat("es-CO", { maximumFractionDigits: 2 }).format(value);
const percentage = (value) => new Intl.NumberFormat("es-CO", { style: "percent", maximumFractionDigits: 2 }).format(value);
const seconds = (value) => Number.isFinite(value) ? `${new Intl.NumberFormat("es-CO", { maximumFractionDigits: 4 }).format(value)} s` : "Sin registro";
const milliseconds = (value) => `${new Intl.NumberFormat("es-CO", { maximumFractionDigits: 4 }).format(value)} ms`;
const memory = (value) => value < 1024 * 1024 ? `${number(value / 1024)} KiB` : `${number(value / 1024 / 1024)} MiB`;

async function request(path, options = {}) {
  const response = await fetch(path, { ...options, signal: AbortSignal.timeout(30000) });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Revisa los valores ingresados e inténtalo de nuevo.");
  return result;
}

function errorMessage(message = "") {
  byId("lab-error").textContent = message;
  byId("lab-error").hidden = !message;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function row(values, tag = "td") {
  const result = element("tr");
  for (const value of values) {
    const cell = element(tag);
    if (value instanceof Node) cell.append(value);
    else cell.textContent = value;
    result.append(cell);
  }
  return result;
}

function button(text, action, className = "secondary") {
  const result = element("button", text, `button compact ${className}`);
  result.type = "button";
  result.addEventListener("click", async () => {
    result.disabled = true;
    try { await action(); } catch (error) { errorMessage(error.message); }
    finally { result.disabled = false; }
  });
  return result;
}

function metric(label, value) {
  const result = element("div", undefined, "lab-metric");
  result.append(element("span", label), element("strong", value));
  return result;
}

function thumbnail(pixels, description) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 28;
  canvas.className = "digit-thumbnail";
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", description);
  const context = canvas.getContext("2d");
  const image = context.createImageData(28, 28);
  pixels.flat().forEach((value, index) => {
    image.data[index * 4] = image.data[index * 4 + 1] = image.data[index * 4 + 2] = value;
    image.data[index * 4 + 3] = 255;
  });
  context.putImageData(image, 0, 0);
  return canvas;
}

function svgNode(tag, attributes = {}, text) {
  const result = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) result.setAttribute(key, value);
  if (text !== undefined) result.textContent = text;
  return result;
}

function chartAxes(chart, labels) {
  chart.replaceChildren();
  for (let tick = 0; tick <= 4; tick++) {
    const y = 205 - tick * 44;
    chart.append(svgNode("line", { x1: 55, x2: 770, y1: y, y2: y, stroke: "#e4ebf4" }));
    chart.append(svgNode("text", { x: 45, y: y + 4, "text-anchor": "end", fill: "#61728d", "font-size": 11 }, `${tick * 25}%`));
  }
  labels.forEach(({ text, x }) => chart.append(svgNode("text", { x, y: 228, "text-anchor": "middle", fill: "#61728d", "font-size": 11 }, text)));
}

function renderKChart(results) {
  const chart = byId("knn-chart");
  const low = Math.min(...results.map((item) => item.k));
  const high = Math.max(...results.map((item) => item.k));
  const xFor = (k) => high === low ? 412 : 80 + (k - low) / (high - low) * 665;
  chartAxes(chart, results.map((item) => ({ text: `K ${item.k}`, x: xFor(item.k) })));
  chart.append(svgNode("polyline", { points: results.map((item) => `${xFor(item.k)},${205 - item.accuracy * 176}`).join(" "), fill: "none", stroke: "#173fbd", "stroke-width": 3 }));
  results.forEach((item) => {
    const point = svgNode("circle", { cx: xFor(item.k), cy: 205 - item.accuracy * 176, r: 4, fill: "#173fbd" });
    point.append(svgNode("title", {}, `K ${item.k}: ${percentage(item.accuracy)}`));
    chart.append(point);
    if (results.length <= 10) chart.append(svgNode("text", { x: xFor(item.k), y: 195 - item.accuracy * 176, "text-anchor": "middle", fill: "#173fbd", "font-size": 11 }, percentage(item.accuracy)));
  });
}

function renderMethodChart(models) {
  const chart = byId("method-chart");
  chartAxes(chart, [{ text: "Red neuronal", x: 260 }, { text: "KNN", x: 565 }]);
  ["network", "knn"].forEach((method, index) => {
    const value = models[method].accuracy;
    const x = index === 0 ? 260 : 565;
    chart.append(svgNode("rect", { x: x - 70, y: 205 - value * 176, width: 140, height: value * 176, rx: 5, fill: index === 0 ? "#173fbd" : "#268faf" }));
    chart.append(svgNode("text", { x, y: 195 - value * 176, "text-anchor": "middle", fill: "#172744", "font-size": 12 }, percentage(value)));
  });
}

function renderConfusion(id, matrix, names) {
  const table = element("table", undefined, "confusion-table");
  table.append(element("caption", "Real ↓ / Predicción →"));
  const head = element("thead");
  head.append(row(["", ...names], "th"));
  table.append(head);
  const body = element("tbody");
  const maxValue = Math.max(1, ...matrix.flat());
  matrix.forEach((values, index) => {
    const tr = element("tr");
    const label = element("th", names[index]);
    label.scope = "row";
    tr.append(label);
    values.forEach((value, column) => {
      const td = element("td", String(value));
      td.style.background = `rgba(23,63,189,${0.04 + value / maxValue * 0.86})`;
      td.style.color = value / maxValue > 0.55 ? "white" : "#172744";
      td.title = `Real ${names[index]}, predicción ${names[column]}: ${value}`;
      tr.append(td);
    });
    body.append(tr);
  });
  table.append(body);
  byId(id).replaceChildren(table);
}

function renderReport(id, report, names) {
  const table = element("table", undefined, "lab-table");
  const head = element("thead");
  head.append(row(["Clase", "Precisión", "Recall", "F1", "Ejemplos"], "th"));
  const body = element("tbody");
  report.forEach((item) => body.append(row([names[item.label], percentage(item.precision), percentage(item.recall), percentage(item.f1), number(item.support)])));
  table.append(head, body);
  byId(id).replaceChildren(table);
}

function renderKDetail() {
  const record = labUI.selected;
  if (!record) return;
  const result = record.results.find((item) => item.k === Number(byId("result-k").value));
  byId("knn-metrics").replaceChildren(
    metric("Aciertos de validación", percentage(result.accuracy)), metric("Errores", number(result.errors)),
    metric("K seleccionado", number(result.k)), metric("Tiempo del barrido completo", seconds(record.evaluation_seconds)),
  );
  renderConfusion("knn-confusion", result.confusion_matrix, record.class_names);
  renderReport("knn-report", result.report, record.class_names);
  byId("knn-prediction").replaceChildren();
}

function renderComparison(record) {
  const models = record.models;
  const difference = models.network.correct - models.knn.correct;
  byId("comparison-summary").textContent = difference === 0
    ? `Ambos acertaron ${number(models.network.correct)} de las mismas ${number(record.summary.evaluation_count)} imágenes. Pueden haber acertado imágenes diferentes.`
    : `${difference > 0 ? "La red neuronal" : "KNN"} acertó ${number(Math.abs(difference))} imágenes más en esta selección de validación. Revisa también tiempo, memoria y los ejemplos de desacuerdo.`;
  byId("method-rows").replaceChildren(...["network", "knn"].map((method) => {
    const model = models[method];
    return row([method === "network" ? "Red neuronal" : `KNN · K ${record.settings.k}`, percentage(model.accuracy), number(model.errors), percentage(model.macro_f1), seconds(model.prediction_seconds), milliseconds(model.milliseconds_per_image), memory(model.model_bytes), seconds(model.preparation_seconds)]);
  }));
  const agreement = record.agreements;
  byId("agreement-metrics").replaceChildren(metric("Ambos acertaron", number(agreement.both_correct)), metric("Solo acertó la red", number(agreement.network_only)), metric("Solo acertó KNN", number(agreement.knn_only)), metric("Ambos fallaron", number(agreement.both_wrong)));
  renderMethodChart(models);
  renderConfusion("network-confusion", models.network.confusion_matrix, record.class_names);
  renderConfusion("comparison-knn-confusion", models.knn.confusion_matrix, record.class_names);
  renderReport("network-report", models.network.report, record.class_names);
  renderReport("comparison-knn-report", models.knn.report, record.class_names);
  const examples = byId("disagreement-examples");
  examples.replaceChildren();
  record.examples.forEach((item) => {
    const card = element("div", undefined, "disagreement-card");
    card.append(thumbnail(item.pixels, `Dígito real ${item.label}`), element("p", `Imagen ${item.index} · Real: ${item.label}`));
    card.append(element("p", `Red: ${item.network}`, item.network === item.label ? "correct-label" : "wrong-label"), element("p", `KNN: ${item.knn}`, item.knn === item.label ? "correct-label" : "wrong-label"));
    examples.append(card);
  });
  if (!record.examples.length) examples.append(element("p", "Ambos métodos dieron las mismas respuestas en esta selección.", "lab-empty"));
}

async function openRecord(id) {
  const token = ++labUI.detailRequest;
  const record = await request(`/api/lab/runs/${encodeURIComponent(id)}`);
  if (token !== labUI.detailRequest) return;
  labUI.selected = record;
  byId("lab-result").hidden = false;
  byId("lab-download").href = `/api/lab/runs/${encodeURIComponent(id)}/download`;
  if (moduleKind === "knn") {
    byId("result-subtitle").textContent = `${record.settings.dataset.toUpperCase()} · ${number(record.summary.reference_count)} referencias · ${number(record.summary.evaluation_count)} ejemplos de validación · Preparación: ${seconds(record.fit_seconds)} · ${memory(record.model_bytes)} de arrays numéricos`;
    byId("result-k").replaceChildren(...record.results.map((item) => new Option(String(item.k), String(item.k))));
    byId("result-k").value = record.summary.best_k;
    const iris = record.settings.dataset === "iris";
    byId("iris-predict-controls").hidden = !iris;
    byId("mnist-predict-controls").hidden = iris;
    renderKChart(record.results);
    renderKDetail();
  } else {
    byId("result-subtitle").textContent = `${number(record.reference_count)} referencias para ambos · ${number(record.summary.evaluation_count)} imágenes reservadas · ${record.models.network.name}`;
    renderComparison(record);
  }
  renderHistory();
}

async function deleteRecords(id) {
  const description = id ? "esta ejecución y sus resultados" : moduleKind === "knn" ? "todas las evaluaciones KNN" : "todas las comparaciones";
  if (!window.confirm(`¿Quieres borrar ${description}? Las imágenes MNIST y los modelos neuronales se conservan.`)) return;
  await request(id ? `/api/lab/runs/${encodeURIComponent(id)}` : `/api/lab/runs?kind=${moduleKind}`, { method: "DELETE" });
  if (!id || labUI.selected?.id === id) {
    labUI.selected = null;
    labUI.detailRequest++;
    byId("lab-result").hidden = true;
  }
  await refreshRecords();
  if (!labUI.selected && labUI.records.length) await openRecord(labUI.records[0].id);
  await refreshCatalog();
}

function renderHistory() {
  const body = byId("lab-history");
  body.replaceChildren();
  labUI.records.forEach((record) => {
    const name = element("div", record.name, "lab-run-name");
    name.append(element("span", new Date(record.created_at).toLocaleString("es-CO"), "lab-run-date"));
    const actions = element("div", undefined, "lab-row-actions");
    actions.append(button("Ver", () => openRecord(record.id)), button("Borrar", () => deleteRecords(record.id), "danger"));
    actions.querySelectorAll("button").forEach((item) => { if (item.textContent === "Borrar") item.disabled = labUI.busy; });
    const summary = record.summary;
    const values = moduleKind === "knn"
      ? [name, summary.best_k, percentage(summary.accuracy), number(summary.evaluation_count), seconds(summary.evaluation_seconds), actions]
      : [name, percentage(summary.network_accuracy), percentage(summary.knn_accuracy), number(summary.evaluation_count), actions];
    const tr = row(values);
    if (record.id === labUI.selected?.id) tr.className = "lab-history-selected";
    body.append(tr);
  });
  if (!labUI.records.length) {
    const cell = element("td", moduleKind === "knn" ? "Todavía no hay evaluaciones KNN. Prepara la primera con el formulario." : "Todavía no hay comparaciones. Elige una red y una ejecución KNN.");
    cell.colSpan = moduleKind === "knn" ? 6 : 5;
    const tr = element("tr");
    tr.append(cell);
    body.append(tr);
  }
  byId("lab-delete-all").disabled = labUI.busy || !labUI.records.length;
}

async function refreshRecords() {
  labUI.records = (await request(`/api/lab/runs?kind=${moduleKind}`)).runs;
  const signature = JSON.stringify([labUI.records, labUI.busy]);
  if (signature !== labUI.recordSignature) {
    labUI.recordSignature = signature;
    renderHistory();
  }
}

function selectOptions(id, records, emptyText) {
  const select = byId(id);
  const previous = select.value;
  select.replaceChildren(...records.map((record) => new Option(record.name, record.id)));
  if (!records.length) select.append(new Option(emptyText, ""));
  if (records.some((record) => record.id === previous)) select.value = previous;
}

async function refreshCatalog() {
  if (moduleKind !== "comparison") return;
  const [networkCatalog, knnCatalog] = await Promise.all([request("/api/experiments"), request("/api/lab/runs?kind=knn")]);
  const networks = networkCatalog.experiments.filter((record) => record.dataset === "mnist-train-first-50000-validation-last-10000").sort((a, b) => (b.validation_accuracy ?? -1) - (a.validation_accuracy ?? -1));
  const knnRuns = knnCatalog.runs.filter((record) => record.settings.dataset === "mnist" && record.summary.reference_count === 50000);
  const signature = JSON.stringify([networks.map((record) => record.id), knnRuns.map((record) => record.id)]);
  labUI.networks = networks;
  labUI.knnRuns = knnRuns;
  if (signature !== labUI.catalogSignature) {
    const previous = byId("comparison-knn").value;
    selectOptions("comparison-network", networks, "Entrena primero una red neuronal");
    selectOptions("comparison-knn", knnRuns, "Prepara KNN con 50.000 referencias");
    if (previous !== byId("comparison-knn").value && knnRuns.length) byId("comparison-k").value = knnRuns.find((record) => record.id === byId("comparison-knn").value).summary.best_k;
    labUI.catalogSignature = signature;
  }
}

function refreshControls() {
  const needsMNIST = moduleKind === "comparison" || byId("knn-dataset")?.value === "mnist";
  const hasModels = moduleKind === "knn" || (labUI.networks.length > 0 && labUI.knnRuns.length > 0);
  byId("lab-fields").disabled = labUI.busy;
  byId("lab-start").disabled = !labUI.connected || labUI.busy || labUI.networkBusy || (needsMNIST && !labUI.ready) || !hasModels;
  byId("lab-stop").hidden = !labUI.busy;
  byId("lab-delete-all").disabled = labUI.busy || !labUI.records.length;
  if (byId("knn-predict-button")) byId("knn-predict-button").disabled = labUI.busy;
  byId("lab-form-note").textContent = labUI.busy ? "Hay una evaluación en curso. Puedes detenerla sin borrar los resultados anteriores."
    : labUI.networkBusy ? "La red neuronal está entrenando. Espera a que termine para medir los métodos."
    : needsMNIST && !labUI.ready ? "Cargando MNIST. Iris se puede explorar mientras tanto."
    : !hasModels ? "Necesitas una red guardada y una evaluación KNN de MNIST con 50.000 referencias."
    : "Los resultados se guardan automáticamente con un nombre distinto para cada ejecución.";
}

async function poll() {
  if (labUI.polling) return;
  labUI.polling = true;
  try {
    const state = await request("/api/lab/state");
    labUI.connected = true;
    labUI.ready = state.dataset_ready;
    labUI.networkBusy = state.network_busy;
    const wasBusy = labUI.busy;
    labUI.busy = ["running", "stopping"].includes(state.phase);
    const relevant = state.kind === moduleKind;
    const names = { idle: "Listo", running: "Evaluando", stopping: "Deteniendo", completed: "Completo", stopped: "Detenido", error: "Error" };
    byId("lab-status").textContent = relevant ? names[state.phase] || state.phase : labUI.busy ? "Otro módulo en curso" : "Listo";
    byId("lab-status").className = `badge ${labUI.busy ? "running" : relevant && state.phase === "completed" ? "complete" : ""}`;
    byId("lab-connection").textContent = "Conectado";
    const progress = relevant && state.progress.total ? Math.min(100, state.progress.completed / state.progress.total * 100) : 0;
    byId("lab-progress-fill").style.width = `${progress}%`;
    byId("lab-progress").setAttribute("aria-valuenow", progress.toFixed(1));
    byId("lab-progress-message").textContent = relevant ? `${state.progress.message}${state.duration ? ` · ${seconds(state.duration)}` : ""}` : labUI.busy ? "Hay una evaluación en curso en el otro módulo." : "Listo para iniciar una nueva evaluación. Puedes consultar los resultados guardados abajo.";
    const logs = byId("lab-logs");
    const nearBottom = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 40;
    logs.textContent = relevant && state.logs.length ? state.logs.join("\n") : "Los registros de este módulo aparecerán al comenzar.";
    if (nearBottom) logs.scrollTop = logs.scrollHeight;
    if (relevant && state.error) errorMessage(state.error);
    if (labUI.ticks++ % 4 === 0 || wasBusy !== labUI.busy) {
      await refreshRecords();
      await refreshCatalog();
    }
    if (state.phase === "completed" && state.kind === moduleKind && state.run_id !== labUI.handledRun) {
      labUI.handledRun = state.run_id;
      await refreshRecords();
      if (labUI.records.some((record) => record.id === state.run_id)) await openRecord(state.run_id);
    }
    if (!labUI.selected && labUI.records.length) await openRecord(labUI.records[0].id);
    refreshControls();
  } catch (error) {
    labUI.connected = false;
    byId("lab-connection").textContent = "Sin conexión";
    byId("lab-start").disabled = true;
    errorMessage(error.message);
  } finally { labUI.polling = false; }
}

function updateDatasetFields() {
  const iris = byId("knn-dataset").value === "iris";
  byId("iris-settings").hidden = !iris;
  byId("mnist-settings").hidden = iris;
  refreshControls();
}

byId("lab-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage();
  byId("lab-start").disabled = true;
  try {
    let settings;
    if (moduleKind === "knn") {
      const tokens = byId("knn-values").value.split(",").map((value) => value.trim());
      const values = tokens.map(Number);
      if (tokens.some((value) => !/^\d+$/.test(value)) || values.some((value) => !Number.isInteger(value) || value < 1 || value > 50) || values.length > 20) throw new Error("Ingresa entre uno y veinte valores enteros de K, del 1 al 50, separados por comas.");
      const dataset = byId("knn-dataset").value;
      settings = { dataset, k_values: values, train_count: Number(byId("reference-count").value), evaluation_count: Number(byId("evaluation-count").value), seed: Number(byId("lab-seed").value), validation_fraction: Number(byId("validation-fraction").value), standardize: dataset === "iris" && byId("standardize").checked };
    } else settings = { network_experiment_id: byId("comparison-network").value, knn_run_id: byId("comparison-knn").value, k: Number(byId("comparison-k").value), evaluation_count: Number(byId("evaluation-count").value), seed: Number(byId("lab-seed").value) };
    await request(`/api/lab/${moduleKind === "knn" ? "knn" : "compare"}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settings) });
    labUI.busy = true;
    refreshControls();
    await poll();
  } catch (error) { errorMessage(error.message); refreshControls(); }
});

byId("lab-stop").addEventListener("click", async () => {
  try { await request("/api/lab/stop", { method: "POST" }); await poll(); }
  catch (error) { errorMessage(error.message); }
});
byId("lab-delete-all").addEventListener("click", async () => {
  try { await deleteRecords(); } catch (error) { errorMessage(error.message); }
});

if (moduleKind === "knn") {
  byId("knn-dataset").addEventListener("change", updateDatasetFields);
  byId("result-k").addEventListener("change", renderKDetail);
  byId("knn-predict-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!labUI.selected) return;
    errorMessage();
    byId("knn-predict-button").disabled = true;
    const selectedId = labUI.selected.id;
    const k = Number(byId("result-k").value);
    try {
      const iris = labUI.selected.settings.dataset === "iris";
      const settings = { run_id: selectedId, k, index: Number(byId("knn-image-index").value), features: iris ? ["iris-sl", "iris-sw", "iris-pl", "iris-pw"].map((id) => Number(byId(id).value)) : null };
      const prediction = await request("/api/lab/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settings) });
      if (labUI.selected?.id !== selectedId || Number(byId("result-k").value) !== k) return;
      const result = byId("knn-prediction");
      result.replaceChildren();
      const heading = element("div", undefined, "neighbor-heading");
      if (prediction.pixels) heading.append(thumbnail(prediction.pixels, "Imagen que recibe KNN"));
      const text = element("div");
      text.append(element("p", "KNN predice"), element("strong", prediction.prediction_name));
      if (prediction.label !== null) text.append(element("p", `Número real: ${prediction.label} · ${prediction.prediction === prediction.label ? "Acertó" : "No acertó"}`));
      text.append(element("p", `${prediction.votes[prediction.prediction]} de ${prediction.k} vecinos votaron por esta clase.`));
      heading.append(text);
      result.append(heading);
      const votes = element("div", undefined, "vote-list");
      Object.entries(prediction.votes).forEach(([label, count]) => votes.append(element("span", `${prediction.class_names[Number(label)]}: ${count} votos`)));
      result.append(votes);
      const neighbors = element("div", undefined, "neighbors-grid");
      prediction.neighbors.forEach((item, index) => {
        const card = element("div", undefined, "neighbor-card");
        if (item.pixels) card.append(thumbnail(item.pixels, `Vecino ${index + 1}, dígito ${item.label}`));
        card.append(element("p", `#${index + 1} · ${prediction.class_names[item.label]} · Distancia ${number(item.distance)}`));
        if (item.features) card.append(element("p", item.features.map(number).join(" · ") + " cm"));
        neighbors.append(card);
      });
      result.append(neighbors);
    } catch (error) { errorMessage(error.message); }
    finally { byId("knn-predict-button").disabled = labUI.busy; }
  });
  updateDatasetFields();
} else {
  byId("comparison-knn").addEventListener("change", () => {
    const record = labUI.knnRuns.find((item) => item.id === byId("comparison-knn").value);
    if (record) byId("comparison-k").value = record.summary.best_k;
  });
}

poll();
setInterval(poll, 1000);
