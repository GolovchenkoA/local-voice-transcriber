(() => {
  const $ = (id) => document.getElementById(id);
  const BAR_COUNT = 56;

  const el = {
    btn: $("rec-btn"), label: $("rec-label"), icon: $("rec-icon"),
    status: $("status"), statusText: $("status-text"), timer: $("timer"),
    viz: $("viz"), count: $("count"), transcript: $("transcript"),
    stable: $("text-stable"), live: $("text-live"), placeholder: $("placeholder"),
    copy: $("copy-btn"), clear: $("clear-btn"),
    device: $("badge-device"), mode: $("badge-mode"),
    modelSelect: $("model-select"), langSelect: $("language-select"),
  };

  const ICON_DOT = '<circle cx="9" cy="9" r="8" fill="currentColor"/>';
  const ICON_SQUARE = '<rect x="2" y="2" width="14" height="14" rx="3" fill="currentColor"/>';

  let text = "";
  let session = null; // { stream, ctx, node, analyser, ws, pending, ready, startedAt }
  let finishing = false;
  let timerId = null;
  let info = null;        // last answer of /api/info
  let loadingModel = false;
  let streaming = true;   // false: the model transcribes when Stop is pressed

  function updateControls() {
    const locked = !!session || loadingModel;
    el.modelSelect.disabled = locked || !info;
    el.langSelect.disabled = locked || !info || !info.languages.length;
    el.btn.disabled = loadingModel || finishing || !info;
  }

  // ---------- visualizer ----------
  const bars = [];
  for (let i = 0; i < BAR_COUNT; i++) {
    const b = document.createElement("div");
    b.className = "bar";
    el.viz.appendChild(b);
    bars.push(b);
  }
  const levels = new Float32Array(BAR_COUNT);
  let freq = null;

  function drawBars(now) {
    const t = now / 1000;
    const mid = (BAR_COUNT - 1) / 2;
    const live = session && session.analyser;
    if (live) {
      if (!freq) freq = new Uint8Array(session.analyser.frequencyBinCount);
      session.analyser.getByteFrequencyData(freq);
      // voice lives roughly in the lowest 40% of the spectrum
      const usable = Math.floor(freq.length * 0.4);
      for (let i = 0; i < BAR_COUNT; i++) {
        const dist = Math.abs(i - mid) / mid; // 0 centre .. 1 edges
        const k = Math.min(usable - 1, Math.floor(dist * dist * usable));
        const target = Math.pow(freq[k] / 255, 1.4);
        levels[i] += (target - levels[i]) * 0.45;
      }
    } else {
      for (let i = 0; i < BAR_COUNT; i++) {
        const target = 0.05 + 0.03 * Math.sin(i * 0.4 + t * 1.5);
        levels[i] += (target - levels[i]) * 0.2;
      }
    }
    for (let i = 0; i < BAR_COUNT; i++) {
      bars[i].style.height = Math.round(6 + levels[i] * 150) + "px";
      bars[i].style.opacity = live ? 0.55 + levels[i] * 0.45 : 0.35;
    }
    requestAnimationFrame(drawBars);
  }
  requestAnimationFrame(drawBars);

  // ---------- transcript ----------
  function renderText() {
    const trimmed = text.replace(/\s+$/, "");
    const cut = trimmed.lastIndexOf(" ") + 1;
    const recording = !!session && !finishing;
    el.stable.textContent = text.slice(0, cut);
    el.live.textContent = text.slice(cut);
    el.transcript.classList.toggle("recording", recording);
    el.placeholder.hidden = text.length > 0;
    const words = trimmed ? trimmed.split(/\s+/).length : 0;
    el.count.textContent = words + (words === 1 ? " word" : " words");
    el.transcript.scrollTop = el.transcript.scrollHeight;
  }

  // ---------- status ----------
  function setStatus(state, message) {
    el.status.dataset.state = state;
    el.statusText.textContent = message;
  }

  function setRecordingUI(on) {
    el.btn.classList.toggle("live", on);
    el.btn.setAttribute("aria-pressed", String(on));
    el.label.textContent = on ? "Stop" : "Record";
    el.icon.innerHTML = on ? ICON_SQUARE : ICON_DOT;
    el.viz.classList.toggle("live", on);
  }

  function startTimer() {
    const t0 = Date.now();
    el.timer.textContent = "00:00";
    timerId = setInterval(() => {
      const s = Math.floor((Date.now() - t0) / 1000);
      el.timer.textContent = String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
    }, 250);
  }
  function stopTimer() { clearInterval(timerId); timerId = null; }

  // ---------- recording ----------
  async function start() {
    el.btn.disabled = true;
    el.modelSelect.disabled = true;
    el.langSelect.disabled = true;
    setStatus("idle", "Connecting…");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (err) {
      fail(err && err.name === "NotFoundError" ? "No microphone found." : "Microphone access was blocked.");
      return;
    }

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/transcribe`);
    ws.binaryType = "arraybuffer";
    const s = { stream, ws, pending: [], ready: false, ctx: null, node: null, analyser: null };
    session = s;

    ws.onopen = () => ws.send(JSON.stringify({
      type: "start",
      language: info && info.languages.length ? el.langSelect.value : null,
    }));
    ws.onerror = () => fail("Cannot reach the transcription server.");
    ws.onclose = () => { if (session === s && !finishing) fail("Connection to the server was lost."); };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "ready") {
        s.ready = true;
        streaming = msg.streaming !== false;
        s.pending.forEach((b) => ws.send(b));
        s.pending = [];
        beginCapture(s);
      } else if (msg.type === "text") {
        text += msg.delta;
        renderText();
      } else if (msg.type === "limit") {
        s.limitHit = true;
        stop(); // the model keeps what was recorded so far and transcribes it
      } else if (msg.type === "done") {
        endSession(s);
      } else if (msg.type === "error") {
        fail(msg.message);
      }
    };
  }

  async function beginCapture(s) {
    try {
      s.ctx = new AudioContext();
      await s.ctx.audioWorklet.addModule("/static/pcm-worklet.js");
      const source = s.ctx.createMediaStreamSource(s.stream);
      s.analyser = s.ctx.createAnalyser();
      s.analyser.fftSize = 1024;
      s.analyser.smoothingTimeConstant = 0.6;
      s.node = new AudioWorkletNode(s.ctx, "pcm-processor");
      s.node.port.onmessage = (e) => {
        if (s.ws.readyState === WebSocket.OPEN) s.ws.send(e.data);
      };
      source.connect(s.analyser);
      source.connect(s.node); // the node has no output, so nothing is played back
    } catch (err) {
      fail("Could not start audio capture: " + err.message);
      return;
    }
    if (text && !/\s$/.test(text)) text += " ";
    setRecordingUI(true);
    updateControls();
    setStatus("live", streaming ? "Listening and transcribing" : "Recording. Text appears when you press Stop");
    startTimer();
    el.btn.disabled = false;
    renderText();
  }

  function stop() {
    const s = session;
    if (!s || finishing) return;
    finishing = true;
    el.btn.disabled = true;
    updateControls();
    stopTimer();
    setRecordingUI(false);
    setStatus("idle", streaming ? "Finishing…" : "Transcribing…");
    s.stream.getTracks().forEach((t) => t.stop());
    if (s.node) s.node.disconnect();
    if (s.ws.readyState === WebSocket.OPEN) s.ws.send(JSON.stringify({ type: "stop" }));
    // safety net; an offline model may need a while for a long recording
    setTimeout(() => { if (session === s) endSession(s); }, streaming ? 8000 : 120000);
  }

  function endSession(s) {
    if (session !== s) return;
    cleanup(s);
    session = null;
    finishing = false;
    setStatus("idle", s.limitHit ? "Stopped at the recording length limit" : (text ? "Stopped" : "Ready"));
    updateControls();
    renderText();
  }

  function fail(message) {
    const s = session;
    if (s) { cleanup(s); session = null; }
    finishing = false;
    stopTimer();
    setRecordingUI(false);
    setStatus("error", message);
    updateControls();
    renderText();
  }

  function cleanup(s) {
    try { s.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
    try { if (s.ctx) s.ctx.close(); } catch (_) {}
    try { s.ws.onclose = null; s.ws.close(); } catch (_) {}
    stopTimer();
    setRecordingUI(false);
  }

  // ---------- buttons ----------
  el.btn.addEventListener("click", () => (session ? stop() : start()));
  el.clear.addEventListener("click", () => { text = ""; el.timer.textContent = "00:00"; renderText(); if (!session) setStatus("idle", "Ready"); });
  el.copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text.trim());
      const old = el.copy.textContent;
      el.copy.textContent = "Copied";
      setTimeout(() => (el.copy.textContent = old), 1200);
    } catch (_) {
      setStatus("error", "Copy is not allowed by the browser.");
    }
  });

  // ---------- engine info, model and language menus ----------
  function option(value, label, disabled) {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    o.disabled = !!disabled;
    return o;
  }

  function applyInfo(next) {
    info = next;
    streaming = info.streaming;
    el.device.textContent = info.cuda ? `${info.device} · CUDA` : "CPU (no CUDA)";
    el.device.classList.toggle("warn", !info.cuda);
    el.mode.textContent = info.streaming
      ? `Live · ${info.latency_ms} ms latency`
      : `Text appears when you press Stop · up to ${Math.round(info.max_seconds / 60)} min`;

    el.modelSelect.replaceChildren(...info.models.map((m) =>
      option(m.id, m.downloaded ? m.label : `${m.label} (not downloaded)`, !m.downloaded)));
    el.modelSelect.value = info.model;

    const previous = el.langSelect.value;
    if (info.languages.length) {
      el.langSelect.replaceChildren(...info.languages.map((l) => option(l.code, l.label)));
      const codes = info.languages.map((l) => l.code);
      el.langSelect.value = codes.includes(previous) ? previous : info.default_language;
    } else {
      el.langSelect.replaceChildren(option("", info.language_note || "Not available"));
    }
    updateControls();
  }

  el.modelSelect.addEventListener("change", async () => {
    const wanted = el.modelSelect.value;
    if (!info || wanted === info.model) return;
    loadingModel = true;
    updateControls();
    setStatus("idle", "Loading model…");
    try {
      const r = await fetch("/api/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: wanted }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      applyInfo(await r.json());
      setStatus("idle", text ? "Stopped" : "Ready");
    } catch (err) {
      el.modelSelect.value = info.model;
      setStatus("error", String(err.message || err));
    } finally {
      loadingModel = false;
      updateControls();
    }
  });

  updateControls();
  fetch("/api/info").then((r) => r.json()).then(applyInfo).catch(() => {
    el.device.textContent = "Server not reachable";
    el.device.classList.add("warn");
  });

  renderText();
})();
