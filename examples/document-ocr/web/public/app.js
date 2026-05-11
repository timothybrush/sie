const els = {
  badge: document.getElementById("badge"),
  models: document.getElementById("models"),
  sieState: document.getElementById("sie-state"),
  events: document.getElementById("events"),
  selectRecognition: document.getElementById("select-recognition"),
  selectStructured: document.getElementById("select-structured"),
  selectNer: document.getElementById("select-ner"),
  recognition: document.getElementById("recognition"),
  recognitionMeta: document.getElementById("recognition-meta"),
  extraction: document.getElementById("extraction"),
  extractionMeta: document.getElementById("extraction-meta"),
  footer: document.getElementById("footer"),
  sieUrl: document.getElementById("sie-url"),
  timings: document.getElementById("timings"),
  snippetRecognition: document.getElementById("snippet-recognition"),
  snippetStructured: document.getElementById("snippet-structured"),
  snippetNer: document.getElementById("snippet-ner"),
};

let activeSampleId = null;
let activeSample = null;
let timings = { recognitionMs: 0, donutMs: 0, glinerMs: 0 };
let donutBuf = { entities: [], data: null };
let glinerBuf = [];
let modelConfig = null;
let registeredSet = new Set();
let cudaAvailable = false;

function setBadge(text, cls) {
  els.badge.textContent = text;
  els.badge.className = "badge" + (cls ? " " + cls : "");
}
function shortModel(id) {
  if (!id) return "";
  const slash = id.indexOf("/");
  return slash === -1 ? id : id.slice(slash + 1);
}
function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

function findModel(list, id) {
  return list ? list.find((m) => m.id === id) : null;
}

function snippetRecognition(modelId) {
  if (!modelConfig) return "";
  const opt = findModel(modelConfig.recognition, modelId);
  const hasOpts = opt && opt.options && Object.keys(opt.options).length > 0;
  const lines = [
    'client.extract(',
    `    <span class="str">"${escapeHtml(modelId)}"</span>,`,
    '    Item(images=[image_bytes]),',
  ];
  if (hasOpts) {
    const optsJson = JSON.stringify(opt.options).replace(/"/g, '"');
    lines.push(`    <span class="arg">options</span>=${escapeHtml(optsJson)},`);
  }
  lines.push(')');
  return `<span class="com"># Recognition: one call against SIE</span>\n` + lines.join("\n");
}

function snippetStructured(modelId) {
  return (
    `<span class="com"># Structured: same client.extract, different model_id</span>\n` +
    `client.extract(\n` +
    `    <span class="str">"${escapeHtml(modelId)}"</span>,\n` +
    `    Item(images=[image_bytes]),\n` +
    `)`
  );
}

function snippetNer(modelId, sample) {
  const labels = sample ? sample.labels : ["merchant", "total", "date"];
  const labelsStr = "[" + labels.map((l) => `<span class="str">"${escapeHtml(l)}"</span>`).join(", ") + "]";
  return (
    `<span class="com"># NER: text input this time, declared labels</span>\n` +
    `client.extract(\n` +
    `    <span class="str">"${escapeHtml(modelId)}"</span>,\n` +
    `    Item(text=recognized_markdown),\n` +
    `    <span class="arg">labels</span>=${labelsStr},\n` +
    `)`
  );
}

function updateSnippets() {
  if (!els.snippetRecognition) return;
  els.snippetRecognition.innerHTML = snippetRecognition(els.selectRecognition.value);
  els.snippetStructured.innerHTML = snippetStructured(els.selectStructured.value);
  els.snippetNer.innerHTML = snippetNer(els.selectNer.value, activeSample);
}

function populateDropdown(selectEl, options, defaultId) {
  selectEl.innerHTML = "";
  for (const opt of options) {
    const node = document.createElement("option");
    node.value = opt.id;
    const inCatalog = registeredSet.size === 0 || registeredSet.has(opt.id);
    const blockedByCuda = opt.gpuRequired && !cudaAvailable;
    const available = inCatalog && !blockedByCuda;
    const labelSuffix = !available
      ? blockedByCuda
        ? " (GPU image needed)"
        : opt.gpuRequired
          ? " (GPU image needed)"
          : " (not registered)"
      : "";
    node.textContent = opt.label + labelSuffix;
    if (!available) node.disabled = true;
    if (opt.id === defaultId) node.selected = true;
    node.title = opt.description;
    selectEl.appendChild(node);
  }
  selectEl.addEventListener("change", updateSnippets);
}

function renderSamples(samples, onClick) {
  if (!samples || samples.length === 0) {
    els.events.innerHTML = '<p class="hint">no samples</p>';
    return;
  }
  els.events.innerHTML = samples
    .map(
      (s) => `<div class="event" data-id="${escapeHtml(s.id)}">
        <img src="/samples/${encodeURIComponent(s.filename)}" alt="${escapeHtml(s.label)}" />
        <div>
          <div class="label">${escapeHtml(s.label)}</div>
          <div class="desc">${escapeHtml(s.description)}</div>
        </div>
      </div>`,
    )
    .join("");
  for (const node of els.events.querySelectorAll(".event")) {
    node.addEventListener("click", () => {
      for (const n of els.events.querySelectorAll(".event")) n.classList.remove("active");
      node.classList.add("active");
      activeSample = samples.find((s) => s.id === node.dataset.id) || null;
      updateSnippets();
      onClick(node.dataset.id);
    });
  }
}

function updateTimings() {
  const total = timings.recognitionMs + timings.donutMs + timings.glinerMs;
  els.timings.textContent =
    total > 0
      ? `recognition ${timings.recognitionMs}ms · structured ${timings.donutMs}ms · ner ${timings.glinerMs}ms · total ${total}ms`
      : "";
}

function renderExtraction() {
  let html = "";

  if (glinerBuf.length > 0) {
    html += `<div class="section"><h3>NER (${escapeHtml(shortModel(els.selectNer.value))})</h3>`;
    for (const f of glinerBuf) {
      html += `<div class="field">
        <span class="label-name">${escapeHtml(f.label)}</span>
        <span class="text">${escapeHtml(f.text)}</span>
        <span class="score">${f.score.toFixed(2)}</span>
      </div>`;
    }
    html += "</div>";
  }

  if (donutBuf.entities.length > 0) {
    html += `<div class="section"><h3>Structured (${escapeHtml(shortModel(els.selectStructured.value))})</h3>`;
    for (const e of donutBuf.entities.slice(0, 25)) {
      html += `<div class="donut-row">
        <span class="key">${escapeHtml(e.label)}</span>
        <span class="val">${escapeHtml(e.text)}</span>
      </div>`;
    }
    html += "</div>";
  }

  if (!html) html = '<p class="hint">running...</p>';
  els.extraction.innerHTML = html;
}

function runSample(sampleId) {
  activeSampleId = sampleId;
  setBadge("running", "running");
  els.recognition.innerHTML = '<p class="hint">running recognition...</p>';
  els.extraction.innerHTML = '<p class="hint">waiting...</p>';
  els.recognitionMeta.textContent = "";
  els.extractionMeta.textContent = "";
  timings = { recognitionMs: 0, donutMs: 0, glinerMs: 0 };
  donutBuf = { entities: [], data: null };
  glinerBuf = [];
  updateTimings();

  const recognition = els.selectRecognition.value;
  const structured = els.selectStructured.value;
  const ner = els.selectNer.value;
  const url = `/api/run?id=${encodeURIComponent(sampleId)}&recognition=${encodeURIComponent(recognition)}&structured=${encodeURIComponent(structured)}&ner=${encodeURIComponent(ner)}`;
  const es = new EventSource(url);

  es.addEventListener("models", (e) => {
    const d = JSON.parse(e.data);
    els.models.innerHTML = `recognition: <code>${shortModel(d.recognition)}</code> · structured: <code>${shortModel(d.structured)}</code> · ner: <code>${shortModel(d.extractor)}</code>`;
  });
  es.addEventListener("recognition_start", () => {
    els.recognitionMeta.textContent = "loading model + generating...";
  });
  es.addEventListener("recognition_done", (e) => {
    const d = JSON.parse(e.data);
    timings.recognitionMs = d.ms;
    els.recognitionMeta.textContent = `${d.markdown.length} chars in ${d.ms}ms`;
    els.recognition.textContent = d.markdown;
    updateTimings();
  });
  es.addEventListener("donut_start", () => {
    els.extractionMeta.textContent = "running structured...";
  });
  es.addEventListener("donut_done", (e) => {
    const d = JSON.parse(e.data);
    timings.donutMs = d.ms;
    donutBuf = { entities: d.entities, data: d.rawData };
    els.extractionMeta.textContent = `structured ${d.ms}ms`;
    renderExtraction();
    updateTimings();
  });
  es.addEventListener("gliner_start", () => {
    els.extractionMeta.textContent = "running NER...";
  });
  es.addEventListener("gliner_done", (e) => {
    const d = JSON.parse(e.data);
    timings.glinerMs = d.ms;
    glinerBuf = d.fields;
    els.extractionMeta.textContent = `ner ${d.ms}ms · ${d.fields.length} fields`;
    renderExtraction();
    updateTimings();
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    setBadge(`done ${d.totalMs}ms`, "green");
    es.close();
  });
  es.addEventListener("error", (e) => {
    setBadge("error", "red");
    if (e.data) {
      try {
        const m = JSON.parse(e.data);
        els.recognitionMeta.textContent = `${m.stage}: ${m.message}`;
      } catch {
        /* */
      }
    }
    es.close();
  });
}

async function init() {
  // Fetch SIE health (and registered models)
  let registered = [];
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    els.sieUrl.textContent = j.sieUrl;
    if (!j.sie) {
      els.sieState.textContent = "SIE not reachable yet (still preloading models?)";
    } else {
      els.sieState.textContent = `SIE healthy · ${j.registeredModels} models registered`;
      registered = j.registered ?? [];
      cudaAvailable = !!j.cuda;
    }
  } catch {
    els.sieState.textContent = "could not reach the local server";
  }
  registeredSet = new Set(registered);

  // Fetch model menus (config-side)
  try {
    const r = await fetch("/api/models");
    modelConfig = await r.json();
    populateDropdown(els.selectRecognition, modelConfig.recognition, modelConfig.defaults.recognition);
    populateDropdown(els.selectStructured, modelConfig.structured, modelConfig.defaults.structured);
    populateDropdown(els.selectNer, modelConfig.ner, modelConfig.defaults.ner);
    updateSnippets();
  } catch (e) {
    console.error("failed to load model config", e);
  }

  // Fetch sample documents
  try {
    const r = await fetch("/api/samples");
    const samples = await r.json();
    renderSamples(samples, runSample);
  } catch {
    els.events.innerHTML = '<p class="hint">failed to load samples</p>';
  }
}

init();
