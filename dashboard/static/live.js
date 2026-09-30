/* Live monitoring and launch controls.
 *
 * The animation here is driven by events the BACKEND recorded as they happened:
 * a packet moves because a `transfer` or `aggregate` event with that client and
 * direction arrived, never on a timer. If the backend emits nothing, nothing
 * moves — an idle picture is the correct picture.
 */
"use strict";
(function () {
  const L = { since: 0, timer: null, state: "idle", clients: [], seen: [], busy: false };
  const $l = (id) => document.getElementById(id);
  const mk = (tag, cls, txt) => { const n = document.createElement(tag);
    if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
  const f4 = (v) => (v == null ? "n/a" : Number(v).toFixed(4));

  function layout() {
    $l("live-root").innerHTML = `
      <div class="card">
        <div class="card-head">
          <h2>Launch a small run</h2>
          <span id="lv-enabled" class="pill idle">checking…</span>
        </div>
        <p class="note" id="lv-frozen"></p>
        <fieldset class="fieldset"><legend>Frozen configuration</legend>
          <div class="row">
            <label class="field"><span>Clients</span>
              <input type="number" id="lv-clients" value="3" min="2" max="4"></label>
            <label class="field"><span>Rounds</span>
              <input type="number" id="lv-rounds" value="2" min="1" max="3"></label>
            <label class="field"><span>Local steps</span>
              <input type="number" id="lv-steps" value="25" min="5" max="50"></label>
            <label class="field"><span>Seed</span>
              <input type="number" id="lv-seed" value="7" min="0" max="9999"></label>
            <label class="field"><span>Method</span><select id="lv-arm"></select></label>
            <label class="field"><span>Evaluate on</span>
              <select id="lv-fold"><option value="val">val fold</option>
                <option value="test">test fold</option></select></label>
            <button class="btn" id="lv-start">Start run</button>
            <button class="btn ghost" id="lv-stop">Stop</button>
          </div>
        </fieldset>
        <div id="lv-error"></div>
        <p class="note">Data is synthetic and generated for the run, so nothing depends on a
          downloaded dataset. Clients are started as separate OS processes and speak to the
          coordinator over mutual TLS with per-client certificates.</p>
      </div>

      <div class="card">
        <div class="card-head"><h2>Live federation</h2><span id="lv-state" class="pill idle">idle</span></div>
        <dl class="kv" id="lv-kv"></dl>
        <div id="lv-map-wrap"><svg id="lv-map" viewBox="0 0 900 420"></svg></div>
        <div class="legend">
          <span><i class="sw model"></i> model update received from a client</span>
          <span><i class="sw back"></i> model sent in response to a client's request</span>
          <span><i class="sw sig"></i> model computed by the server (no transfer)</span>
        </div>
        <p class="note">Four stages, and only three are observable from the server:
          the server <b>computes</b> a model, a client <b>requests</b> it, the server
          <b>sends</b> the response. Receipt by the client is not acknowledged by this
          protocol, so it is never shown. A model that is computed and never requested
          shows a pulse on the coordinator and no packet.</p>
      </div>

      <div class="card">
        <h2>Realised weights (this run)</h2>
        <div class="table-wrap"><table id="lv-weights"></table></div>
        <p class="note" id="lv-weights-note"></p>
      </div>

      <div class="card">
        <h2>Prediction error, reported by the clients</h2>
        <div class="table-wrap"><table id="lv-rmse"></table></div>
        <p class="note" id="lv-rmse-note"></p>
      </div>

      <div class="card">
        <h2>Recorded events</h2>
        <p class="note">Timestamps are real, not simulated. For <code>transfer</code>,
          <code>aggregate</code>, <code>round</code> and <code>fallback</code> lines they are the
          moment the coordinator handled that event. <code>client-log</code> lines are the client
          processes' own output, collected when each process exits, so their timestamp is the
          collection time and is marked as such.</p>
        <div class="log" id="lv-log"></div>
      </div>

      <div class="card">
        <h2>Not available for a live run</h2>
        <dl class="glossary" id="lv-unavailable"></dl>
        <div class="callout warn">A live run is one run. It is a demonstration that the system
          works end to end, not evidence about which method is better. Nothing here feeds back
          into method selection or tuning.</div>
      </div>`;

    $l("lv-start").onclick = start;
    $l("lv-stop").onclick = () => post("live/stop", {});
  }

  async function post(name, body) {
    const r = await fetch(`/api/${name}`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const out = await r.json();
    if (!r.ok) throw new Error(out.error || `HTTP ${r.status}`);
    return out;
  }

  async function start() {
    $l("lv-error").innerHTML = "";
    const cfg = {
      clients: Number($l("lv-clients").value), rounds: Number($l("lv-rounds").value),
      local_steps: Number($l("lv-steps").value), seed: Number($l("lv-seed").value),
      arm: $l("lv-arm").value, evaluate_fold: $l("lv-fold").value,
    };
    try {
      L.since = 0; L.seen = [];
      $l("lv-log").innerHTML = "";
      await post("live/start", cfg);
    } catch (e) {
      $l("lv-error").innerHTML = `<div class="callout bad">${e.message}</div>`;
    }
  }

  async function poll() {
    let s;
    try { s = await (await fetch(`/api/live/status?since=${L.since}`)).json(); }
    catch (e) { return; }
    if (s.error && s.state === "failed") {
      $l("lv-error").innerHTML = `<div class="callout bad">${s.error}</div>`;
    }

    if (!$l("lv-arm").options.length) {
      s.arms.forEach((a) => { const o = mk("option", null, a); o.value = a;
        $l("lv-arm").appendChild(o); });
      $l("lv-arm").value = s.defaults.arm;
      const p = $l("lv-enabled");
      p.textContent = s.launch_enabled ? "launching enabled" : "launching disabled";
      p.className = "pill " + (s.launch_enabled ? "" : "off");
      $l("lv-start").disabled = !s.launch_enabled;
      $l("lv-frozen").textContent = s.launch_enabled
        ? "Only these options can be set, each within a fixed range. There is no command or path " +
          "field: the dashboard starts the bundled demonstration and nothing else."
        : "Launching is disabled. Restart the dashboard with --allow-launch to enable it. " +
          "Monitoring still works for a run started elsewhere in this process.";
      const dl = $l("lv-unavailable"); dl.innerHTML = "";
      Object.entries(s.unavailable || {}).forEach(([k, v]) => {
        dl.appendChild(mk("dt", null, k.replace(/_/g, " ")));
        dl.appendChild(mk("dd", null, v));
      });
    }

    L.state = s.state;
    const badge = $l("lv-state");
    badge.textContent = s.state + (s.state === "running"
      ? ` · round ${s.rounds_done + (s.rounds_done < s.rounds_total ? 1 : 0)} of ${s.rounds_total}`
      : "");
    badge.className = "pill " + (s.state === "running" || s.state === "preparing" ? ""
      : s.state === "failed" ? "off" : "idle");

    if (s.clients.length && s.clients.join() !== L.clients.join()) {
      L.clients = s.clients; drawLiveMap(s);
    }
    renderKV(s);
    renderWeights(s);
    renderRmse(s);

    (s.events || []).forEach((e) => {
      L.since = Math.max(L.since, e.seq);
      appendEvent(e);
      // Animate a packet ONLY for an observed transfer: an accepted update
      // arriving, or a parent model actually sent in response to a request.
      // A `computed` event is not a transfer - it pulses the server instead.
      if (e.kind === "transfer" || e.kind === "delivered") animate(e);
      else if (e.kind === "computed") pulseServer();
    });
  }

  function renderKV(s) {
    const dl = $l("lv-kv"); dl.innerHTML = "";
    const add = (k, v) => { dl.appendChild(mk("dt", null, k)); dl.appendChild(mk("dd", null, v)); };
    add("Coordinator", s.url || "not started");
    add("Transport", s.url ? "mutual TLS, per-client certificates" : "—");
    add("Method", s.cfg.arm || "—");
    add("Rounds completed", `${s.rounds_done} of ${s.rounds_total || 0}`);
    add("Client processes", s.processes.length
      ? s.processes.map((p) => `${p.client} (pid ${p.pid}${p.running ? ", running"
          : `, exited ${p.exit}`})`).join("; ")
      : "none");
    if (s.signature_clients && s.signature_clients.length) {
      add("Signatures received", s.signature_clients.join(", ") +
        " — aggregate mask summaries only");
    }
  }

  function drawLiveMap(s) {
    const ids = s.clients, n = ids.length;
    const W = 900, H = 420, cx = W / 2, cy = H / 2, R = 150;
    const p = [`<g id="lv-edges">`];
    const pos = {};
    ids.forEach((id, i) => {
      const th = -Math.PI / 2 + (2 * Math.PI * i) / n;
      pos[id] = { x: cx + R * Math.cos(th), y: cy + R * Math.sin(th) };
      p.push(`<path class="edge dim" data-client="${id}" stroke-width="2"
        d="M ${pos[id].x.toFixed(1)} ${pos[id].y.toFixed(1)}
           Q ${((pos[id].x + cx) / 2).toFixed(1)} ${((pos[id].y + cy) / 2).toFixed(1)}
           ${cx} ${cy}"/>`);
    });
    p.push(`</g><g id="lv-packets"></g>`);
    p.push(`<g class="node server" transform="translate(${cx},${cy})">
      <circle class="disc" r="50"/>
      <text text-anchor="middle" y="-4" font-size="13" font-weight="700">Coordinator</text>
      <text text-anchor="middle" y="13" font-size="9.5">holds no client data</text></g>`);
    ids.forEach((id) => {
      p.push(`<g class="node" data-client="${id}"
          transform="translate(${pos[id].x.toFixed(1)},${pos[id].y.toFixed(1)})">
        <circle class="disc" r="40"/>
        <text text-anchor="middle" y="-3" font-size="13" font-weight="700">${id}</text>
        <text text-anchor="middle" y="13" font-size="9" opacity=".8">separate process</text></g>`);
    });
    $l("lv-map").innerHTML = p.join("");
  }

  function pulseServer() {
    const svg = $l("lv-map");
    const disc = svg.querySelector(".server .disc");
    if (!disc || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    disc.style.transition = "none";
    disc.style.stroke = "var(--teal-2)";
    disc.style.strokeWidth = "7";
    setTimeout(() => { disc.style.transition = "stroke-width .5s, stroke .5s";
      disc.style.stroke = ""; disc.style.strokeWidth = ""; }, 60);
  }

  function animate(e) {
    const svg = $l("lv-map");
    const edge = svg.querySelector(`.edge[data-client="${e.client}"]`);
    const layer = svg.querySelector("#lv-packets");
    if (!edge || !layer) return;
    edge.classList.remove("dim");
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const len = edge.getTotalLength();
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("r", "6");
    dot.setAttribute("class", "packet " +
      (e.direction === "out" ? "back" : ""));
    layer.appendChild(dot);
    const inbound = e.direction === "in";
    const t0 = performance.now(), dur = 620;
    const tick = (t) => {
      const u = Math.min(1, (t - t0) / dur);
      const pt = edge.getPointAtLength((inbound ? u : 1 - u) * len);
      dot.setAttribute("cx", pt.x); dot.setAttribute("cy", pt.y);
      if (u < 1) requestAnimationFrame(tick); else dot.remove();
    };
    requestAnimationFrame(tick);
  }

  function appendEvent(e) {
    const box = $l("lv-log");
    const d = mk("div");
    const ts = new Date(e.t * 1000).toLocaleTimeString([], { hour12: false }) +
      "." + String(Math.floor((e.t % 1) * 1000)).padStart(3, "0");
    const caveat = e.kind === "client-log" ? ' <span class="tag unavailable">collected at exit</span>' : "";
    d.innerHTML = `<span class="seq">${ts}</span><span class="k">${e.kind}</span> ${e.text}${caveat}`;
    box.appendChild(d);
    box.scrollTop = box.scrollHeight;
  }

  function renderRmse(s) {
    const t = $l("lv-rmse");
    const rows = s.rmse || [];
    if (!rows.length) {
      t.innerHTML = "";
      $l("lv-rmse-note").textContent =
        "No client has reported yet. Error appears once a round closes: each client evaluates " +
        "on its own held-out fold and sends only the record count and the summed squared error.";
      return;
    }
    t.innerHTML = `<thead><tr><th>Round</th><th>Client</th><th>Fold</th>
      <th>Records</th><th>Summed sq. error<br>(received)</th><th>RMSE received</th>
      <th>Summed sq. error<br>(after local training)</th><th>RMSE local</th></tr></thead>`;
    const tb = mk("tbody");
    rows.forEach((r) => {
      const tr = mk("tr");
      if (String(r.client).startsWith("ALL")) tr.className = "diag";
      tr.appendChild(mk("td", null, String(r.round)));
      tr.appendChild(mk("td", null, r.client));
      tr.appendChild(mk("td", null, r.fold));
      tr.appendChild(mk("td", null, String(r.local_n)));
      tr.appendChild(mk("td", null, f4(r.received_sse)));
      tr.appendChild(mk("td", null, f4(r.received_rmse)));
      tr.appendChild(mk("td", null, f4(r.local_sse)));
      tr.appendChild(mk("td", null, f4(r.local_rmse)));
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    $l("lv-rmse-note").textContent = s.rmse_note || "";
  }

  function renderWeights(s) {
    const t = $l("lv-weights");
    if (!s.records.length) { t.innerHTML = "";
      $l("lv-weights-note").textContent =
        "No round has closed yet. Weights appear once the coordinator has actually aggregated.";
      return; }
    t.innerHTML = `<thead><tr><th>Round</th><th>Receiver</th><th>gamma (self)</th>
      <th>Donor weights</th><th>New version</th></tr></thead>`;
    const tb = mk("tbody");
    s.records.forEach((rec, ri) => rec.forEach((x) => {
      const tr = mk("tr");
      tr.appendChild(mk("td", null, String(ri + 1)));
      tr.appendChild(mk("td", null, x.receiver));
      tr.appendChild(mk("td", null, f4(x.gamma)));
      tr.appendChild(mk("td", null, Object.entries(x.weights)
        .map(([k, v]) => `${k}: ${f4(v)}`).join("   ")));
      tr.appendChild(mk("td", "mono", String(x.new_version || "").slice(0, 8)));
      tb.appendChild(tr);
    }));
    t.appendChild(tb);
    const sc = s.scores_seen || {};
    const rounds = Object.keys(sc);
    $l("lv-weights-note").textContent = rounds.length
      ? `Scores were computed on the server from the aggregate signatures the clients actually ` +
        `sent. They are identical across rounds when the clients' missingness does not change — ` +
        `that is expected, not a bug, and the dashboard does not vary them for effect.`
      : `This method uses no receiver-to-donor score, so weights come from the declared sample ` +
        `counts alone.`;
  }

  // Poll while the live tab is visible. Events are buffered server-side by seq,
  // so nothing is lost while the tab is hidden.
  setInterval(() => {
    const visible = document.getElementById("view-live").classList.contains("active");
    if (visible || L.state === "running" || L.state === "preparing") poll();
  }, 700);

  layout();
  poll();
})();
