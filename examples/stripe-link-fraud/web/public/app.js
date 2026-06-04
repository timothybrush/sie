// stripe-link-fraud frontend: SSE-driven risk panel + Stripe Elements (Link)
// checkout. No build step; vanilla JS.

const els = {
  badge: document.getElementById("badge"),
  sieState: document.getElementById("sie-state"),
  stripeState: document.getElementById("stripe-state"),
  sieUrl: document.getElementById("sie-url"),
  orders: document.getElementById("orders"),
  risk: document.getElementById("risk"),
  riskMeta: document.getElementById("risk-meta"),
  checkout: document.getElementById("checkout"),
  checkoutMeta: document.getElementById("checkout-meta"),
  timings: document.getElementById("timings"),
};

let stripe = null;
let elements = null;
let activeOrder = null;
let currentRisk = null;
let stripeMode = "mock";
let stripePk = null;

function setBadge(text, cls) {
  els.badge.textContent = text;
  els.badge.className = "badge" + (cls ? " " + cls : "");
}

function fmtMs(ms) {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function fmtUsd(n) {
  return `$${n.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 2 })}`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderOrders(samples) {
  if (!samples?.length) {
    els.orders.innerHTML = '<p class="hint">no samples</p>';
    return;
  }
  els.orders.innerHTML = samples
    .map((s) => {
      const total = s.cart.reduce((acc, i) => acc + i.qty * i.unit_price_usd, 0);
      return `
      <div class="order" data-id="${s.id}">
        <div class="order-head">
          <span class="order-label">${escapeHtml(s.label)}</span>
          <span class="order-total">${fmtUsd(total)}</span>
        </div>
        <div class="order-desc">${escapeHtml(s.description)}</div>
        <div class="order-meta">
          <span class="pill">${escapeHtml(s.customer.email)}</span>
          <span class="pill">${s.customer.link_returning ? "Link returning" : "Link new"}</span>
          <span class="pill">${escapeHtml(s.customer.billing_country)}/${escapeHtml(s.customer.ip_country)}</span>
        </div>
      </div>`;
    })
    .join("");
  for (const node of els.orders.querySelectorAll(".order")) {
    node.addEventListener("click", () => {
      for (const o of els.orders.querySelectorAll(".order")) o.classList.remove("active");
      node.classList.add("active");
      const id = node.dataset.id;
      activeOrder = samples.find((s) => s.id === id);
      runRisk(id);
    });
  }
}

function renderRiskPanelStart() {
  els.risk.innerHTML = `
    <div class="risk-stages">
      <div class="stage" data-key="extract"><span class="stage-dot"></span> extract</div>
      <div class="stage" data-key="encode"><span class="stage-dot"></span> encode</div>
      <div class="stage" data-key="score"><span class="stage-dot"></span> score</div>
    </div>
    <div class="risk-body"></div>`;
  els.riskMeta.textContent = "";
  els.timings.textContent = "";
}

function setStage(key, state) {
  const node = els.risk.querySelector(`.stage[data-key="${key}"]`);
  if (node) node.dataset.state = state;
}

function renderEntities(entities) {
  if (!entities?.length) return '<p class="hint">no entities extracted</p>';
  return entities
    .map(
      (e) => `
      <span class="entity-group" title="GLiNER confidence: ${e.score.toFixed(3)}">
        <span class="ekey">${escapeHtml(e.label)}</span>
        <span class="eval">${escapeHtml(e.text)}</span>
        <span class="econf">${e.score.toFixed(2)}</span>
      </span>`,
    )
    .join("");
}

function renderHits(hits, topScore, band) {
  const bandWhy = {
    low: "below the block threshold (cosine &lt; 0.47). Proceed with normal Link checkout.",
    medium: "between block and review thresholds (0.47 - 0.52). Authorize but flag for analyst review.",
    high: "above the review threshold (cosine &gt; 0.52). Hold for review before clearing.",
  }[band];
  return `
    <div class="risk-band band-${band}">
      <div class="band-label">SIE risk band</div>
      <div class="band-value">${band.toUpperCase()}</div>
      <div class="band-score" title="Cosine similarity of the order embedding to the closest fraud pattern in the corpus.">
        top cosine ${topScore.toFixed(3)}
      </div>
    </div>
    <p class="band-why">${bandWhy}</p>
    <div class="score-legend">
      Each row below shows a candidate fraud pattern. The number on the right is the
      <strong>cross-encoder reranker score</strong> for this order against that pattern
      (higher = more semantically similar). The top-3 cosine candidates are reranked
      and reordered by this score.
    </div>
    <div class="hits">
      ${hits
        .map(
          (h) => `
        <div class="hit">
          <div class="hit-head">
            <span class="hit-label">${escapeHtml(h.label)}</span>
            <span class="hit-score" title="Cross-encoder relevance score (BGE-reranker-base), 0-1 range. Higher = more semantically similar to this fraud pattern.">${h.score.toFixed(3)}</span>
          </div>
          <div class="hit-summary">${escapeHtml(h.summary)}</div>
          <div class="hit-meta">
            <span class="pill outcome-${h.outcome}">${escapeHtml(h.outcome)}</span>
            <span class="pill">${fmtUsd(h.loss_usd)} historical loss</span>
          </div>
        </div>`,
        )
        .join("")}
    </div>`;
}

async function runRisk(id) {
  setBadge("running", "running");
  renderRiskPanelStart();
  els.checkout.innerHTML = '<p class="hint">Risk pipeline running...</p>';
  const evt = new EventSource(`/api/run?id=${encodeURIComponent(id)}`);
  const body = els.risk.querySelector(".risk-body");
  const ts = { extract: 0, encode: 0, score: 0, total: 0 };

  evt.addEventListener("extracting", () => setStage("extract", "running"));
  evt.addEventListener("extracted", (e) => {
    setStage("extract", "done");
    const data = JSON.parse(e.data);
    ts.extract = data.ms;
    body.insertAdjacentHTML(
      "beforeend",
      `<div class="risk-section">
         <div class="risk-section-head">Extracted signals · ${fmtMs(data.ms)}</div>
         <div class="hint" style="margin-bottom:8px">
           GLiNER zero-shot NER pulls typed entities out of the order summary.
           The number next to each label is the model's confidence (0-1) for that span.
         </div>
         <div class="entities">${renderEntities(data.entities)}</div>
       </div>`,
    );
  });
  evt.addEventListener("encoding", () => setStage("encode", "running"));
  evt.addEventListener("encoded", (e) => {
    setStage("encode", "done");
    const data = JSON.parse(e.data);
    ts.encode = data.ms;
    body.insertAdjacentHTML(
      "beforeend",
      `<div class="risk-section">
         <div class="risk-section-head">Encoded order context · ${fmtMs(data.ms)}</div>
         <div class="hint">
           MiniLM-L6 turns the order summary into a dense ${data.dim}-dimensional vector.
           Cosine similarity against pre-encoded fraud-pattern vectors picks the top-3 candidates for reranking.
         </div>
       </div>`,
    );
  });
  evt.addEventListener("scoring", () => setStage("score", "running"));
  evt.addEventListener("scored", (e) => {
    setStage("score", "done");
    const data = JSON.parse(e.data);
    ts.score = data.ms;
    body.insertAdjacentHTML(
      "beforeend",
      `<div class="risk-section">
         <div class="risk-section-head">Reranked top matches · ${fmtMs(data.ms)}</div>
         ${renderHits(data.hits, data.topScore, data.band)}
       </div>`,
    );
    currentRisk = { band: data.band, topScore: data.topScore };
    mountCheckout();
  });
  evt.addEventListener("done", (e) => {
    const data = JSON.parse(e.data);
    ts.total = data.totalMs;
    setBadge("done", "green");
    els.timings.textContent = `extract ${fmtMs(ts.extract)} · encode ${fmtMs(ts.encode)} · score ${fmtMs(ts.score)} · total ${fmtMs(ts.total)}`;
    evt.close();
  });
  evt.addEventListener("error", (e) => {
    setBadge("error", "red");
    let msg = "stream error";
    try {
      const data = JSON.parse(e.data);
      msg = `${data.stage}: ${data.message}`;
    } catch {
      /* network error event has no payload */
    }
    body.insertAdjacentHTML("beforeend", `<div class="error">${escapeHtml(msg)}</div>`);
    evt.close();
  });
}

async function mountCheckout() {
  if (!activeOrder) return;
  const total = activeOrder.cart.reduce((s, i) => s + i.qty * i.unit_price_usd, 0);
  if (stripeMode === "mock") {
    els.checkout.innerHTML = `
      <div class="mock-checkout">
        <div class="mock-banner">
          Stripe keys not set. The risk pipeline above runs against real SIE;
          set <code>STRIPE_PUBLISHABLE_KEY</code> and <code>STRIPE_SECRET_KEY</code>
          in <code>.env</code> (test mode) to mount a real Link payment form.
        </div>
        <div class="mock-cart">
          <div class="mock-cart-head">
            <span>Cart</span>
            <span>${fmtUsd(total)}</span>
          </div>
          ${activeOrder.cart
            .map(
              (i) => `
            <div class="mock-cart-row">
              <span>${escapeHtml(i.name)} × ${i.qty}</span>
              <span>${fmtUsd(i.qty * i.unit_price_usd)}</span>
            </div>`,
            )
            .join("")}
        </div>
        <div class="mock-link-button" data-band="${currentRisk?.band ?? "low"}">
          ${currentRisk?.band === "high" ? "Hold for review" : "Pay with Link"}
        </div>
      </div>`;
    els.checkoutMeta.textContent = `mock · risk ${currentRisk?.band ?? ""}`;
    return;
  }

  els.checkout.innerHTML = '<p class="hint">Creating PaymentIntent...</p>';
  try {
    const r = await fetch("/api/payment-intent", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        cart: activeOrder.cart,
        customer: activeOrder.customer,
        riskBand: currentRisk?.band ?? "unknown",
      }),
    });
    const j = await r.json();
    if (j.mode !== "live") throw new Error("server returned mock mode");
    els.checkout.innerHTML = `
      <div class="cart-summary">
        <div class="cart-head"><span>Cart</span><span>${fmtUsd(total)}</span></div>
        ${activeOrder.cart
          .map(
            (i) => `
          <div class="cart-row">
            <span>${escapeHtml(i.name)} × ${i.qty}</span>
            <span>${fmtUsd(i.qty * i.unit_price_usd)}</span>
          </div>`,
          )
          .join("")}
      </div>
      <div id="payment-element"></div>
      <button id="pay-btn" class="pay-btn band-${currentRisk?.band ?? "low"}">
        ${currentRisk?.band === "high" ? "Hold for review" : "Pay with Link"}
      </button>
      <div id="pay-msg" class="pay-msg"></div>`;
    stripe = stripe ?? Stripe(j.publishableKey ?? stripePk);
    elements = stripe.elements({ clientSecret: j.clientSecret });
    const payEl = elements.create("payment", { layout: "tabs" });
    payEl.mount("#payment-element");
    const btn = document.getElementById("pay-btn");
    const msg = document.getElementById("pay-msg");
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      msg.textContent = "";
      const { error } = await stripe.confirmPayment({
        elements,
        confirmParams: { return_url: window.location.href },
      });
      if (error) {
        msg.textContent = error.message ?? "payment failed";
        msg.className = "pay-msg error";
        btn.disabled = false;
      }
    });
    els.checkoutMeta.textContent = `live · risk ${currentRisk?.band ?? ""}`;
  } catch (e) {
    els.checkout.innerHTML = `<div class="error">${escapeHtml(e?.message ?? String(e))}</div>`;
  }
}

async function init() {
  els.sieUrl.textContent = "...";
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    els.sieUrl.textContent = j.sieUrl;
    els.sieState.textContent = j.sie
      ? `SIE healthy · ${j.registeredModels} models`
      : "SIE not reachable yet";
    stripeMode = j.stripe;
    stripePk = j.publishableKey;
    els.stripeState.textContent =
      stripeMode === "live"
        ? "Stripe: live (test keys detected)"
        : "Stripe: mock (no keys set)";
  } catch {
    els.sieState.textContent = "could not reach the local server";
  }
  try {
    const r = await fetch("/api/samples");
    const samples = await r.json();
    renderOrders(samples);
  } catch {
    els.orders.innerHTML = '<p class="hint">failed to load samples</p>';
  }
}

init();
