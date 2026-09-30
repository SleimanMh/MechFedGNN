/* MechFedGNN dashboard - presentation only.
 *
 * Every number shown here is produced by the backend, which calls the research
 * code. This file computes no scores, no weights and no statistics; it formats
 * what it is given. The one thing it does generate is the REPLAY ANIMATION, and
 * that is labelled everywhere as a reconstruction of the protocol's fixed order,
 * because historical runs recorded no events and no timestamps.
 */
"use strict";

const S = {          // shared UI state
  run: null, runPath: null, seed: null, arm: null, receiver: null,
  view: "overview", log: [], seq: 0, replaying: false,
};
const $ = (id) => document.getElementById(id);
const el = (tag, cls, txt) => { const n = document.createElement(tag);
  if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };

async function api(name, params) {
  const q = new URLSearchParams(params || {}).toString();
  const r = await fetch(`/api/${name}${q ? "?" + q : ""}`);
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}
function toast(msg, ms = 2600) {
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => { t.hidden = true; }, ms);
}
const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v))
  ? "n/a" : Number(v).toFixed(d);
const pct = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? "n/a" : `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(d)}%`;
const naCell = (v, d = 4) => { const td = el("td", v == null ? "na" : null, fmt(v, d)); return td; };

function fill(sel, values, current) {
  sel.innerHTML = "";
  values.forEach((v) => {
    const o = el("option", null, String(v.label !== undefined ? v.label : v));
    o.value = String(v.value !== undefined ? v.value : v);
    sel.appendChild(o);
  });
  if (current !== undefined && current !== null &&
      values.some((v) => String(v.value !== undefined ? v.value : v) === String(current))) {
    sel.value = String(current);
  }
  return sel.value;
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  const { runs } = await api("runs");
  if (!runs.length) { document.querySelector("main").prepend(
    banner("No completed runs were found under the results directory.")); return; }
  const sel = $("run-select");
  sel.innerHTML = "";
  runs.forEach((r) => {
    const o = el("option", null,
      `${r.experiment} — ${r.dataset}${r.condition ? " · " + r.condition : ""}` +
      `${r.s_config ? " · " + r.s_config : ""}  (${r.run_id})`);
    o.value = r.path; sel.appendChild(o);
  });
  sel.onchange = () => loadRun(sel.value);
  await loadRun(sel.value);
}
function banner(text, cls = "callout warn") {
  const d = el("div", cls); d.textContent = text; return d;
}

async function loadRun(path) {
  S.runPath = path;
  S.run = await api("run", { path });
  S.log = []; S.seq = 0;
  const armsWithScore = S.run.method.arms;
  const seeds = S.run.seeds;
  S.seed = seeds[0];
  S.arm = armsWithScore.includes("combined-Q") ? "combined-Q"
        : armsWithScore[armsWithScore.length - 1];
  S.receiver = S.run.clients[0].id;

  ["ov-seed", "ag-seed", "w-seed"].forEach((id) => { fill($(id), seeds, S.seed); });
  ["ov-arm", "ag-arm", "w-arm", "e-arm"].forEach((id) => { fill($(id), armsWithScore, S.arm); });
  const ids = S.run.clients.map((c) => c.id);
  ["ag-receiver", "w-receiver", "e-receiver"].forEach((id) => { fill($(id), ids, S.receiver); });
  fill($("r-fold"), S.run.axes.folds, S.run.axes.folds.includes("test") ? "test" : undefined);
  fill($("r-tp"), S.run.axes.timepoints, S.run.axes.timepoints.includes("t2") ? "t2" : undefined);
  fill($("r-ref"), armsWithScore, armsWithScore.includes("uniform-donor")
    ? "uniform-donor" : armsWithScore[0]);

  logLine("run", `opened ${S.run.experiment} / ${S.run.dataset} (${S.run.run_id})`);
  await renderAll();
}

async function renderAll() {
  await Promise.all([renderOverview(), renderAggregation(), renderWeights(),
                     renderResults(), renderTimeline()]);
  renderGlossary();
}

/* -------------------------------------------------------------- overview */
async function renderOverview() {
  const r = S.run, m = r.method;
  const kv = $("round-kv"); kv.innerHTML = "";
  const a = r.availability;
  addKV(kv, "Dataset", r.dataset);
  addKV(kv, "Experiment", r.experiment);
  addKV(kv, "Clients", String(r.clients.length));
  addKV(kv, "Federated rounds", "1" + tagHTML("single round", "unavailable"));
  addKV(kv, "Seeds (repetitions)", String(r.seeds.length));
  addKV(kv, "Participation", `${r.clients.length} of ${r.clients.length} clients in every round`);
  addKV(kv, "Messages per round", `${r.clients.length} signature uploads, ` +
    `${r.clients.length} parameter uploads, ${r.clients.length} personalised models returned`);
  $("round-note").textContent = a.federated_rounds.note +
    " — participation was complete by construction, so there is no drop-out to show.";

  const mk = $("method-kv"); mk.innerHTML = "";
  addKV(mk, "Selected method (arm)", S.arm);
  addKV(mk, "alpha (score vs size)", fmt(m.alpha, 2));
  addKV(mk, "beta (sharpening)", fmt(m.beta, 2));
  addKV(mk, "gamma (self-weight)", fmt(m.gamma, 2));
  addKV(mk, "lambda_pop", fmt(m.lambda_pop, 2));
  addKV(mk, "Local steps", String(m.local_steps));
  addKV(mk, "Adaptation steps", String(m.adapt_budget));

  await drawMap();
}
function addKV(dl, k, v) {
  dl.appendChild(el("dt", null, k));
  const dd = el("dd"); dd.innerHTML = v; dl.appendChild(dd);
}
function tagHTML(text, cls) { return ` <span class="tag ${cls}">${text}</span>`; }

async function drawMap() {
  const svg = $("map");
  const ids = S.run.clients.map((c) => c.id);
  let agg = null;
  try { agg = await api("aggregation", { path: S.runPath, seed: S.seed,
                                         receiver: S.receiver, arm: S.arm }); }
  catch (e) { /* local-only has no donors */ }

  const W = 900, H = 460, cx = W / 2, cy = H / 2, R = 165;
  const pos = {}; const n = ids.length;
  ids.forEach((id, i) => {
    const th = -Math.PI / 2 + (2 * Math.PI * i) / n;
    pos[id] = { x: cx + R * Math.cos(th), y: cy + R * Math.sin(th) };
  });
  const contrib = {};
  if (agg) agg.donors.forEach((d) => { contrib[d.donor] = d.effective_contribution; });

  const parts = [];
  parts.push(`<g id="edges">`);
  ids.forEach((id) => {
    const p = pos[id];
    const c = contrib[id];
    const wdt = c == null ? 1.2 : 1.2 + 16 * c;
    const dim = (id === S.receiver || c == null) ? " dim" : "";
    parts.push(`<path class="edge${dim}" data-client="${id}" stroke-width="${wdt.toFixed(2)}"
      d="M ${p.x.toFixed(1)} ${p.y.toFixed(1)} Q ${((p.x + cx) / 2).toFixed(1)}
         ${((p.y + cy) / 2).toFixed(1)} ${cx} ${cy}"/>`);
  });
  parts.push(`</g><g id="packets"></g>`);

  parts.push(`<g class="node server" transform="translate(${cx},${cy})">
    <circle class="disc" r="52"/>
    <text text-anchor="middle" y="-6" font-size="13" font-weight="700">Server</text>
    <text text-anchor="middle" y="11" font-size="10.5">coordinator</text>
    <text text-anchor="middle" y="25" font-size="9.5" opacity=".8">no client rows</text></g>`);

  S.run.clients.forEach((c) => {
    const p = pos[c.id];
    const isRec = c.id === S.receiver;
    const share = contrib[c.id];
    const line = isRec ? `receiver · γ=${fmt(agg ? agg.gamma : S.run.method.gamma, 2)}`
      : (share == null ? "not weighted" : `gives ${(100 * share).toFixed(1)}%`);
    parts.push(`<g class="node ${isRec ? "receiver selected" : ""}" data-client="${c.id}"
        transform="translate(${p.x.toFixed(1)},${p.y.toFixed(1)})">
      <circle class="disc" r="43"/>
      <text text-anchor="middle" y="-9" font-size="13" font-weight="700">${c.id}</text>
      <text text-anchor="middle" y="6" font-size="9.5">n=${c.n_train}</text>
      <text text-anchor="middle" y="19" font-size="8.5" opacity=".85">${line}</text></g>`);
  });
  svg.innerHTML = parts.join("");

  svg.querySelectorAll(".node[data-client]").forEach((g) => {
    g.addEventListener("click", () => openClient(g.dataset.client));
  });
  $("map-note").textContent =
    `Line thickness is each donor's effective contribution to ${S.receiver} under ` +
    `${S.arm}. Click any client to inspect what it holds and what it sends.`;
}

async function openClient(id) {
  S.receiver = id;
  ["ag-receiver", "w-receiver", "e-receiver"].forEach((s) => { $(s).value = id; });
  const c = await api("client", { path: S.runPath, id });
  $("client-card").hidden = false;
  $("client-title").textContent = `Client ${id}`;
  const b = $("client-body"); b.innerHTML = "";

  const dl = el("dl", "kv");
  addKV(dl, "Training rows", String(c.n_train));
  addKV(dl, "Validation rows", `<span class="na">not recorded</span>${tagHTML("unavailable", "unavailable")}`);
  addKV(dl, "Test rows", `<span class="na">not recorded</span>${tagHTML("unavailable", "unavailable")}`);
  addKV(dl, "Missingness group", String(c.group));
  b.appendChild(dl);

  b.appendChild(banner(
    "What this client sends: aggregate mask summaries (per-feature missing rates, joint-absence " +
    "and joint-observation counts, the association matrix), coarse histograms, and its model " +
    "parameters. What never leaves it: its rows, its labels, its per-row masks and its " +
    "per-example predictions.", "callout"));

  const h = el("h3", null, "Per-feature missingness (this client)"); b.appendChild(h);
  const t = el("table");
  t.innerHTML = "<thead><tr><th>Feature</th><th>Missing rate</th><th>Observed</th>" +
    "<th>Maskable</th></tr></thead>";
  const tb = el("tbody");
  c.feature_rows.forEach((f) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, f.feature));
    tr.appendChild(el("td", null, fmt(f.missing_rate, 3)));
    const bar = el("td"); const b2 = el("span", "bar");
    b2.style.width = `${Math.max(1, 100 * (1 - (f.missing_rate || 0))).toFixed(1)}%`;
    bar.appendChild(b2); tr.appendChild(bar);
    tr.appendChild(el("td", null, f.maskable ? "yes" : "no"));
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  const wrap = el("div", "table-wrap"); wrap.appendChild(t); b.appendChild(wrap);

  b.appendChild(el("p", "note",
    "'Maskable' features are the ones the injector was allowed to hide for this experiment; " +
    "the rest are always observed. A high missing rate means this client rarely sees that column."));

  await drawMap();
  await renderAggregation();
}
$("client-close").onclick = () => { $("client-card").hidden = true; };

/* ----------------------------------------------------------- aggregation */
async function renderAggregation() {
  const seed = Number($("ag-seed").value || S.seed);
  const rec = $("ag-receiver").value || S.receiver;
  const arm = $("ag-arm").value || S.arm;
  let d;
  try { d = await api("aggregation", { path: S.runPath, seed, receiver: rec, arm }); }
  catch (e) {
    $("ag-steps").innerHTML = ""; $("ag-table").innerHTML = "";
    $("ag-source").textContent = "";
    $("ag-fallback").innerHTML = "";
    $("ag-balance").className = "balance";
    $("ag-balance").textContent = String(e.message);
    return;
  }

  $("ag-source").innerHTML =
    `Donor weights for this run were <b>${d.weights_source}</b>.` +
    (d.weights_source.startsWith("recomputed")
      ? tagHTML("recomputed", "recomputed") +
        " The research runs never wrote a weights file, so the dashboard calls the same " +
        "weighting function the experiment used rather than approximating it."
      : "");

  $("ag-fallback").innerHTML = d.fallback
    ? `<div class="callout warn"><b>Declared fallback active (${d.fallback}).</b> ` +
      `${d.fallback_explained} This is the pre-declared rule, not an error.</div>` : "";

  const ol = $("ag-steps"); ol.innerHTML = "";
  d.steps.forEach((s) => {
    const li = el("li");
    if (!d.uses_score && s.n === 1) li.className = "inactive";
    li.appendChild(el("div", "t", s.title));
    li.appendChild(el("div", "d", s.detail));
    const vals = Object.entries(s.values || {});
    if (vals.length) {
      li.appendChild(el("div", "v",
        vals.map(([k, v]) => `${k} = ${v == null ? "undefined" : fmt(v, 5)}`).join("   ")));
    }
    ol.appendChild(li);
  });

  const t = $("ag-table");
  t.innerHTML = `<thead><tr><th>Donor</th>
    <th>${d.uses_score ? "Score (" + (d.donors[0] ? d.donors[0].score_name : "-") + ")" : "Score"}</th>
    <th>Sample share p</th><th>Donor weight w</th><th>Effective (1−γ)·w</th>
    <th>Share of the aggregate</th></tr></thead>`;
  const tb = el("tbody");
  d.donors.forEach((x) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, x.donor));
    tr.appendChild(x.score == null ? el("td", "na", d.uses_score ? "undefined" : "not used")
                                   : el("td", null, fmt(x.score, 5)));
    tr.appendChild(naCell(x.sample_share, 4));
    tr.appendChild(naCell(x.donor_weight, 4));
    tr.appendChild(naCell(x.effective_contribution, 4));
    const bc = el("td"); const bar = el("span", "bar");
    bar.style.width = `${Math.max(1, 100 * (x.effective_contribution || 0)).toFixed(1)}%`;
    bc.appendChild(bar); tr.appendChild(bc);
    tb.appendChild(tr);
  });
  const self = el("tr");
  self.appendChild(el("td", null, `${d.receiver} (itself)`));
  self.appendChild(el("td", "na", "—"));
  self.appendChild(el("td", "na", "—"));
  self.appendChild(el("td", null, `γ = ${fmt(d.gamma, 4)}`));
  self.appendChild(el("td", null, fmt(d.receiver_contribution, 4)));
  const sc = el("td"); const sb = el("span", "bar self");
  sb.style.width = `${Math.max(1, 100 * d.receiver_contribution).toFixed(1)}%`;
  sc.appendChild(sb); self.appendChild(sc);
  tb.appendChild(self);
  t.appendChild(tb);

  const bal = $("ag-balance");
  bal.className = "balance " + (d.total_ok ? "ok" : "bad");
  bal.textContent = `γ (${fmt(d.receiver_contribution, 6)}) + Σ donors ` +
    `(${fmt(d.effective_donor_sum, 6)}) = ${fmt(d.total_contribution, 6)}` +
    (d.total_ok ? "  ✓ the aggregate is a proper weighted average"
                : "  ✗ contributions do not sum to 1");
}

function renderGlossary() {
  const g = $("glossary"); g.innerHTML = "";
  const terms = [
    ["Receiver", "The client whose personalised model is being built right now. Every client is " +
      "a receiver in its own turn — there is no single global model."],
    ["Donor", "Any other client whose parameters may be mixed into the receiver's model. A client " +
      "never donates to itself."],
    ["Score (q)", "A number describing how useful a donor looks for this receiver, computed only " +
      "from aggregate missingness summaries and coarse histograms — never from the donor's data."],
    ["alpha", "How much the score is trusted relative to plain sample size: " +
      "base = α·q + (1−α)·p. α = 0 ignores the score entirely."],
    ["beta", "Sharpening. Raising base to the power β widens the gap between strong and weak " +
      "donors; β = 1 leaves it unchanged."],
    ["gamma (γ)", "The fraction of its OWN model the receiver keeps. This is what makes the " +
      "result personalised rather than a single shared model."],
    ["Donor weight (w)", "The normalised split of the remaining 1−γ across donors. The w column " +
      "sums to 1; the effective contributions sum to 1−γ."],
    ["Sample share (p)", "The donor's training rows as a fraction of all clients' training rows."],
    ["Declared fallback", "If the score is undefined for any donor, or all donors tie to within " +
      "1e-9, the score cannot rank donors, so the arm falls back to sample-size weighting. " +
      "This was declared before the experiments ran and is recorded whenever it fires."],
    ["FedAvg", "The standard baseline: no score, weights proportional to sample size, and the " +
      "receiver's self-weight equal to its own sample share."],
    ["uniform-donor", "Equal weight to every donor. This is the reference the scored methods " +
      "have to beat to be worth anything."],
    ["t1 / t2", "t1 is the aggregated model as received; t2 is after the receiver adapts it on " +
      "its own training rows."],
  ];
  terms.forEach(([k, v]) => { g.appendChild(el("dt", null, k)); g.appendChild(el("dd", null, v)); });
}

/* --------------------------------------------------------------- weights */
async function renderWeights() {
  const seed = Number($("w-seed").value || S.seed);
  const arm = $("w-arm").value || S.arm;
  const t = $("w-matrix");
  try {
    const m = await api("matrix", { path: S.runPath, seed, arm });
    const head = ["Receiver ↓ / contributor →", ...m.clients];
    t.innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`;
    const tb = el("tbody");
    m.matrix.forEach((row, i) => {
      const tr = el("tr");
      tr.appendChild(el("td", null, m.clients[i] +
        (m.fallbacks[m.clients[i]] ? " (fallback)" : "")));
      row.forEach((v, j) => {
        const td = naCell(v, 3);
        if (i === j) td.className = "diag";
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    $("w-matrix-note").textContent = m.note +
      (Object.keys(m.fallbacks).length
        ? ` Rows marked (fallback) used the declared sample-size fallback.` : "");
  } catch (e) {
    t.innerHTML = ""; $("w-matrix-note").textContent = e.message;
  }

  const rec = $("w-receiver").value || S.receiver;
  try {
    const h = await api("history", { path: S.runPath, receiver: rec, arm });
    drawHistory(h);
    $("w-history-note").textContent =
      (h.constant_note || "Donor weights vary across seeds because the partition and the injected " +
       "masks are redrawn for each seed.") +
      "  A research run is a single federated round, so this is a comparison across repetitions, " +
      "not a within-run trajectory.";
  } catch (e) {
    $("w-history").textContent = e.message; $("w-history-note").textContent = "";
  }
}

function drawHistory(h) {
  const box = $("w-history"); box.innerHTML = "";
  const names = Object.keys(h.normalised_weights);
  if (!names.length) { box.appendChild(el("p", "note", "This arm assigns no donor weights.")); return; }
  const W = 760, H = 170, pad = { l: 44, r: 12, t: 12, b: 26 };
  const xs = h.x, n = xs.length;
  const X = (i) => pad.l + (n === 1 ? (W - pad.l - pad.r) / 2
    : (i * (W - pad.l - pad.r)) / (n - 1));
  const all = names.flatMap((k) => h.normalised_weights[k]).filter((v) => v != null);
  const lo = Math.min(0, ...all), hi = Math.max(...all, 0.001);
  const Y = (v) => H - pad.b - ((v - lo) / (hi - lo || 1)) * (H - pad.t - pad.b);
  const colors = ["var(--teal)", "var(--burgundy)", "var(--navy-2)", "var(--teal-2)",
                  "var(--burgundy-2)"];
  const p = [`<svg class="spark" viewBox="0 0 ${W} ${H}">`];
  p.push(`<line class="axis" x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}"/>`);
  p.push(`<line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H - pad.b}"/>`);
  p.push(`<text class="lbl" x="4" y="${Y(hi) + 4}">${hi.toFixed(2)}</text>`);
  p.push(`<text class="lbl" x="4" y="${Y(lo) + 4}">${lo.toFixed(2)}</text>`);
  xs.forEach((s, i) => p.push(
    `<text class="lbl" x="${X(i)}" y="${H - 8}" text-anchor="middle">${s}</text>`));
  names.forEach((k, ci) => {
    const vals = h.normalised_weights[k];
    const d = vals.map((v, i) => (v == null ? null : `${i === 0 ? "M" : "L"} ${X(i)} ${Y(v)}`))
      .filter(Boolean).join(" ");
    p.push(`<path d="${d}" stroke="${colors[ci % colors.length]}"/>`);
    vals.forEach((v, i) => { if (v != null)
      p.push(`<circle cx="${X(i)}" cy="${Y(v)}" r="2.5" fill="${colors[ci % colors.length]}"/>`); });
  });
  p.push(`</svg>`);
  box.innerHTML = p.join("");
  const legend = el("div");
  names.forEach((k, ci) => {
    const c = el("span", "chip");
    const i = el("i"); i.style.background = colors[ci % colors.length];
    c.appendChild(i); c.appendChild(document.createTextNode(k));
    legend.appendChild(c);
  });
  legend.appendChild(el("span", "note", ` — x axis: ${h.x_label}`));
  box.appendChild(legend);
}

/* --------------------------------------------------------------- results */
async function renderResults() {
  const fold = $("r-fold").value, tp = $("r-tp").value, ref = $("r-ref").value;
  const d = await api("results", { path: S.runPath, fold, timepoint: tp, reference: ref });

  $("r-caution").innerHTML =
    "<b>Reading this table.</b> The seed is the unit of replication: each arm is compared with " +
    "the reference within a seed, averaged over receivers, and then summarised over seeds with a " +
    "95% interval. A single run is never a finding. The ±2% threshold was fixed before these " +
    "experiments ran and is not adjustable here.";

  const pr = d.per_receiver;
  $("r-sub").textContent = `(${pr.metric}, ${fold}, ${tp}; mean over ${pr.seeds} seeds)`;
  const t = $("r-table");
  t.innerHTML = `<thead><tr><th>Receiver</th>${pr.arms.map((a) =>
    `<th>${a}</th>`).join("")}</tr></thead>`;
  const tb = el("tbody");
  pr.receivers.forEach((rname, i) => {
    const row = pr.mean[i];
    const best = Math.min(...row.filter((v) => v != null));
    const tr = el("tr");
    tr.appendChild(el("td", null, rname));
    row.forEach((v) => {
      const td = naCell(v, 5);                 // 5dp: 4dp made distinct values look tied
      if (v != null && v === best) td.classList.add("best");
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  t.appendChild(tb);

  const ct = $("r-contrast");
  if (d.contrasts.error) { ct.innerHTML = ""; $("r-note").textContent = d.contrasts.error; }
  else {
    ct.innerHTML = `<thead><tr><th>Arm vs ${d.contrasts.reference}</th><th>Δ%</th>
      <th>95% interval</th><th>Seeds</th><th>Decision</th></tr></thead>`;
    const b = el("tbody");
    d.contrasts.rows.forEach((r) => {
      const tr = el("tr");
      tr.appendChild(el("td", null, r.arm));
      tr.appendChild(el("td", null, pct(r.mean_pct)));
      tr.appendChild(el("td", null, `[${pct(r.lo)}, ${pct(r.hi)}]`));
      tr.appendChild(el("td", null, String(r.seeds)));
      const dec = el("td");
      const cls = r.decision.startsWith("meaningful") ? "meaningful"
        : r.decision === "negligible" ? "negligible" : "unresolved";
      dec.innerHTML = `<span class="tag ${cls}">${r.decision}</span>`;
      tr.appendChild(dec);
      b.appendChild(tr);
    });
    ct.appendChild(b);
    $("r-note").textContent = d.contrasts.note;
  }
  await renderExplain();
}

async function renderExplain() {
  const d = await api("explain", { path: S.runPath, fold: $("r-fold").value,
    timepoint: $("r-tp").value, receiver: $("e-receiver").value, arm: $("e-arm").value });
  const b = $("explain-body"); b.innerHTML = "";
  b.appendChild(el("p", null,
    `Mean RMSE for ${d.receiver} under ${d.arm} on the ${d.fold} fold at ${d.timepoint}: ` +
    `${fmt(d.mean_rmse, 5)} (over ${d.per_seed.length} seeds).`));

  const ol = el("ol", "steps");
  d.chain.forEach((c) => {
    const li = el("li");
    li.appendChild(el("div", "t", c.step));
    li.appendChild(el("div", "d", c.detail));
    ol.appendChild(li);
  });
  b.appendChild(ol);

  const h = el("h3", null, "Per-seed values"); b.appendChild(h);
  const t = el("table");
  t.innerHTML = "<thead><tr><th>Seed</th><th>RMSE</th><th>Fallback</th></tr></thead>";
  const tb = el("tbody");
  d.per_seed.forEach((r) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, String(r.seed)));
    tr.appendChild(el("td", null, fmt(r.rmse, 5)));
    tr.appendChild(r.fallback ? el("td", null, r.fallback) : el("td", "na", "—"));
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  const w = el("div", "table-wrap"); w.appendChild(t); b.appendChild(w);

  const un = el("div", "callout warn");
  un.innerHTML = "<b>Not shown, because it was never saved:</b><br>" +
    Object.entries(d.unavailable).map(([k, v]) => `<code>${k}</code> — ${v}`).join("<br>");
  b.appendChild(un);
  b.appendChild(banner(d.caution, "callout"));
}

/* -------------------------------------------------------------- timeline */
async function renderTimeline() {
  const d = await api("timeline", { path: S.runPath, seed: S.seed, arm: S.arm });
  $("tl-warning").innerHTML = `<b>This is not an execution trace.</b> ${d.note}. ` +
    `${d.rounds_note}`;
  const ol = $("tl-stages"); ol.innerHTML = "";
  d.stages.forEach((s) => {
    const li = el("li"); li.dataset.stage = s.id;
    li.appendChild(el("div", "t", s.label));
    li.appendChild(el("div", "d", s.detail));
    ol.appendChild(li);
  });
  S.stages = d.stages;
  renderLog();
}

function logLine(kind, text) {
  S.seq += 1;
  S.log.push({ seq: S.seq, kind, text });
  renderLog();
}
function renderLog() {
  const box = $("event-log");
  box.innerHTML = "";
  if (!S.log.length) {
    box.appendChild(el("div", "note",
      "No entries yet. Press Replay to step through the protocol order for this run."));
  }
  S.log.slice(-400).forEach((e) => {
    const d = el("div");
    d.innerHTML = `<span class="seq">#${String(e.seq).padStart(3, "0")}</span>` +
      `<span class="k">${e.kind}</span> ${e.text}`;
    box.appendChild(d);
  });
  box.scrollTop = box.scrollHeight;
  $("log-source").textContent = S.replaying
    ? "reconstructed from the protocol order — this run saved no timestamps"
    : "entries are dashboard actions and reconstructed protocol steps, not recorded server events";
}
$("log-clear").onclick = () => { S.log = []; S.seq = 0; renderLog(); };

/* ------------------------------------------------------- replay animation */
async function replayRound() {
  if (S.replaying) return;
  S.replaying = true;
  $("replay-btn").disabled = true; $("log-replay").disabled = true;
  const ids = S.run.clients.map((c) => c.id);
  let agg = null;
  try { agg = await api("aggregation", { path: S.runPath, seed: S.seed,
                                         receiver: S.receiver, arm: S.arm }); } catch (e) {}

  logLine("replay", `reconstructing seed ${S.seed}, arm ${S.arm}, receiver ${S.receiver}` +
    " — order is the protocol's, not a measured timeline");

  const stages = S.stages || [];
  for (const st of stages) {
    markStage(st.id);
    logLine("stage", `${st.label} — ${st.detail}`);
    if (st.id === "signature") {
      for (const id of ids) { await flyPacket(id, "in", "sig");
        logLine("upload", `${id} → server: aggregate mask summary (no rows, no labels)`); }
    } else if (st.id === "aggregation") {
      for (const d of (agg ? agg.donors : [])) {
        await flyPacket(d.donor, "in", "model");
        logLine("upload", `${d.donor} → server: parameters, effective contribution ` +
          `${fmt(d.effective_contribution, 4)}`);
      }
      await flyPacket(S.receiver, "out", "back");
      logLine("aggregate", `server → ${S.receiver}: personalised model, γ=` +
        `${fmt(agg ? agg.gamma : S.run.method.gamma, 4)} of its own model retained`);
    } else {
      await sleep(420);
    }
  }
  document.querySelectorAll("#tl-stages li").forEach((li) => {
    li.classList.remove("active"); li.classList.add("done");
  });
  logLine("replay", "reconstruction complete");
  S.replaying = false;
  $("replay-btn").disabled = false; $("log-replay").disabled = false;
  renderLog();
}
function markStage(id) {
  document.querySelectorAll("#tl-stages li").forEach((li) => {
    li.classList.toggle("active", li.dataset.stage === id);
    if (id && li.dataset.stage !== id) li.classList.toggle("done",
      (S.stages || []).findIndex((s) => s.id === li.dataset.stage) <
      (S.stages || []).findIndex((s) => s.id === id));
  });
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function flyPacket(clientId, dir, kind) {
  const svg = $("map");
  const edge = svg.querySelector(`.edge[data-client="${clientId}"]`);
  const layer = svg.querySelector("#packets");
  if (!edge || !layer) return sleep(120);
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return sleep(80);
  const len = edge.getTotalLength();
  const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  dot.setAttribute("r", "6");
  dot.setAttribute("class", `packet ${kind === "sig" ? "sig" : kind === "back" ? "back" : ""}`);
  layer.appendChild(dot);
  const dur = 520;
  return new Promise((res) => {
    const t0 = performance.now();
    const tick = (t) => {
      const u = Math.min(1, (t - t0) / dur);
      const p = edge.getPointAtLength((dir === "in" ? u : 1 - u) * len);
      dot.setAttribute("cx", p.x); dot.setAttribute("cy", p.y);
      if (u < 1) requestAnimationFrame(tick);
      else { dot.remove(); res(); }
    };
    requestAnimationFrame(tick);
  });
}

/* ------------------------------------------------------------ guided tour */
const TOUR = [
  ["overview", "Each circle is a client holding its own rows. They differ in WHICH columns are " +
    "missing, not only in how much. The server in the middle never receives a single record."],
  ["overview", "Click a client. You see its per-feature missing rates and, explicitly, the list " +
    "of what it transmits — aggregate summaries and parameters only."],
  ["aggregation", "This is the whole method in seven steps, with this run's actual numbers: " +
    "a score per donor, a blend with sample size, sharpening, normalisation, the receiver's " +
    "self-weight, the effective contributions, and the parameter mix."],
  ["aggregation", "Note the balance line: γ plus the donor contributions equals exactly 1. " +
    "That is the check that the result is a proper weighted average."],
  ["weights", "Who contributes to whom. The diagonal is each client's self-weight; every row " +
    "sums to one. Weights differ per receiver — this is personalisation, not one global model."],
  ["results", "Now the outcome. Compare each method with equal donor weighting. The honest " +
    "result across these experiments is that the scored methods sit inside ±2% — no meaningful " +
    "advantage. The dashboard shows that rather than hiding it."],
  ["timeline", "Finally, the protocol order, and an explicit statement that this run saved no " +
    "timestamps, so nothing here is presented as a measured execution trace."],
];
let tourAt = -1;
function tour(step) {
  tourAt = step;
  if (step < 0 || step >= TOUR.length) { $("guide").hidden = true; tourAt = -1; return; }
  const [view, text] = TOUR[step];
  showView(view);
  $("guide").hidden = false;
  $("guide-step").textContent = `Step ${step + 1} of ${TOUR.length}`;
  $("guide-text").textContent = text;
  $("guide-prev").disabled = step === 0;
  $("guide-next").textContent = step === TOUR.length - 1 ? "Finish" : "Next";
}
$("guide-btn").onclick = () => tour(0);
$("guide-next").onclick = () => tour(tourAt + 1);
$("guide-prev").onclick = () => tour(tourAt - 1);
$("guide-exit").onclick = () => tour(-1);

/* ---------------------------------------------------------------- wiring */
function showView(name) {
  S.view = name;
  document.querySelectorAll(".view").forEach((v) =>
    v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === name));
}
document.querySelectorAll("#tabs button").forEach((b) => {
  b.onclick = () => showView(b.dataset.view);
});

$("ov-seed").onchange = async (e) => { S.seed = Number(e.target.value);
  $("ag-seed").value = e.target.value; $("w-seed").value = e.target.value;
  await renderOverview(); await renderAggregation(); await renderWeights(); await renderTimeline(); };
$("ov-arm").onchange = async (e) => { S.arm = e.target.value;
  $("ag-arm").value = S.arm; $("w-arm").value = S.arm;
  await renderOverview(); await renderAggregation(); await renderWeights(); await renderTimeline(); };
["ag-seed", "ag-receiver", "ag-arm"].forEach((id) => {
  $(id).onchange = async () => {
    if (id === "ag-receiver") { S.receiver = $(id).value; await drawMap(); }
    await renderAggregation();
  };
});
["w-seed", "w-arm", "w-receiver"].forEach((id) => { $(id).onchange = renderWeights; });
["r-fold", "r-tp", "r-ref"].forEach((id) => { $(id).onchange = renderResults; });
["e-receiver", "e-arm"].forEach((id) => { $(id).onchange = renderExplain; });
$("replay-btn").onclick = replayRound;
$("log-replay").onclick = async () => { showView("timeline"); await replayRound(); };

$("theme-btn").onclick = () => {
  const dark = document.documentElement.dataset.theme === "dark";
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  $("theme-btn").textContent = dark ? "Dark" : "Light";
  try { localStorage.setItem("mfg-theme", dark ? "light" : "dark"); } catch (e) {}
};
try {
  const saved = localStorage.getItem("mfg-theme");
  if (saved) { document.documentElement.dataset.theme = saved;
    $("theme-btn").textContent = saved === "dark" ? "Light" : "Dark"; }
} catch (e) {}

boot().catch((e) => {
  document.querySelector("main").prepend(banner(`Could not start: ${e.message}`, "callout bad"));
});
