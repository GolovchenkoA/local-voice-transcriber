(() => {
  const $ = (id) => document.getElementById(id);
  const BAR_COUNT = 56;
  // One colour per speaker; speakers are also named in text, so colour is never the only cue.
  const SPEAKER_COLORS = ["#4cc9f0", "#f4a261", "#b388ff", "#ff6b9a", "#7bd88f", "#f2d55c", "#6fa8ff", "#ff7a66"];

  const el = {
    btn: $("rec-btn"), label: $("rec-label"), icon: $("rec-icon"),
    status: $("status"), statusText: $("status-text"), timer: $("timer"),
    viz: $("viz"), count: $("count"), transcript: $("transcript"),
    stable: $("text-stable"), live: $("text-live"), placeholder: $("placeholder"),
    single: $("single-view"), dialogue: $("dialogue"),
    copy: $("copy-btn"), clear: $("clear-btn"),
    modelSelect: $("model-select"), langSelect: $("language-select"),
    diarField: $("diarizer-field"), diarSelect: $("diarizer-select"),
    modeBtn: $("mode-toggle"), modeText: $("mode-text"), modeKnob: $("mode-knob"),
    bDevice: $("badge-device"), bSpeakers: $("badge-speakers"), bModel: $("badge-model"),
    bDiar: $("badge-diarizer"), bMode: $("badge-mode"), bLang: $("badge-language"),
    inBtn: $("in-btn"), inMenu: $("in-menu"), inCount: $("in-count"),
  };

  const ICON_DOT = '<circle cx="9" cy="9" r="8" fill="currentColor"/>';
  const ICON_SQUARE = '<rect x="2" y="2" width="14" height="14" rx="3" fill="currentColor"/>';

  let mode = "single";    // what the user chose with the toggle
  let info = null;        // last answer of the server
  let text = "";          // single-speaker transcript
  let turns = [];         // dialogue: {speaker, start, end, text, rec}
  let recordings = 0;
  let session = null;     // { streams, ctx, node, analyser, ws, pending, ready, rec }
  let finishing = false;
  let timerId = null;
  let busyLoading = false;
  let streaming = true;   // false: the model transcribes when Stop is pressed

  const hasContent = () => (mode === "multi" ? turns.length > 0 : text.length > 0);

  // ---------- controls ----------
  function updateControls() {
    const locked = !!session || busyLoading;
    el.modeBtn.disabled = locked;
    el.modelSelect.disabled = locked || !info;
    el.langSelect.disabled = locked || !info || !info.languages.length;
    el.diarSelect.disabled = locked || !info || info.diarization !== "external";
    el.btn.disabled = busyLoading || finishing || !info;
    el.inBtn.disabled = locked;
    if (locked) closeInputMenu();
  }

  // ---------- audio inputs ----------
  const SYSTEM = "system"; // id of "computer audio" (picked in the browser's share dialog)
  let mics = [];           // {id, label}
  let chosen = new Set();  // ticked ids: microphone device ids and SYSTEM
  const stopStream = (st) => { try { st.getTracks().forEach((t) => t.stop()); } catch (_) {} };

  function saveInputs() {
    try { localStorage.setItem("audio-inputs", JSON.stringify([...chosen])); } catch (_) {}
    el.inCount.textContent = chosen.size;
    el.inBtn.title = `Audio inputs (${chosen.size} selected)`;
  }

  async function loadMics(askForNames) {
    try {
      let devs = await navigator.mediaDevices.enumerateDevices();
      if (askForNames && devs.some((d) => d.kind === "audioinput" && !d.label)) {
        // device names are hidden until the page has microphone access
        stopStream(await navigator.mediaDevices.getUserMedia({ audio: true }));
        devs = await navigator.mediaDevices.enumerateDevices();
      }
      mics = devs.filter((d) => d.kind === "audioinput" && d.deviceId !== "communications")
        .map((d, i) => ({ id: d.deviceId, label: d.label || `Microphone ${i + 1}` }));
    } catch (_) { /* no access yet: keep the list we have */ }
  }

  function renderInputMenu() {
    el.inMenu.textContent = "";
    const head = (t) => { const h = document.createElement("div"); h.className = "in-head"; h.textContent = t; el.inMenu.append(h); };
    const add = (id, label, hint) => {
      const row = document.createElement("label");
      row.className = "in-item";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = chosen.has(id);
      box.onchange = () => { if (box.checked) chosen.add(id); else chosen.delete(id); saveInputs(); };
      const name = document.createElement("span");
      name.textContent = label;
      if (hint) { const s = document.createElement("small"); s.textContent = hint; name.append(s); }
      row.append(box, name);
      el.inMenu.append(row);
    };
    head("Microphones");
    if (!mics.length) {
      const none = document.createElement("div");
      none.className = "in-empty";
      none.textContent = "No microphone found";
      el.inMenu.append(none);
    }
    mics.forEach((m) => add(m.id, m.label));
    head("Computer");
    add(SYSTEM, "System audio", "Video, music, calls. In the share dialog choose a tab or screen and tick “Share audio”.");
  }

  function closeInputMenu() {
    el.inMenu.hidden = true;
    el.inBtn.setAttribute("aria-expanded", "false");
  }

  el.inBtn.addEventListener("click", async () => {
    if (!el.inMenu.hidden) { closeInputMenu(); return; }
    el.inMenu.hidden = false;
    el.inBtn.setAttribute("aria-expanded", "true");
    renderInputMenu();
    await loadMics(true);
    if (!el.inMenu.hidden) renderInputMenu();
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".inputs")) closeInputMenu(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeInputMenu(); });

  async function initInputs() {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem("audio-inputs")); } catch (_) {}
    await loadMics(false);
    const known = new Set([SYSTEM, ...mics.map((m) => m.id)]);
    chosen = new Set(Array.isArray(saved) ? saved.filter((id) => known.has(id)) : []);
    // the microphone is ticked by default
    if (!chosen.size && !Array.isArray(saved)) {
      const def = mics.find((m) => m.id === "default") || mics[0];
      chosen.add(def ? def.id : "default");
    }
    saveInputs();
  }

  // Open every ticked input; system audio goes first because the share dialog needs the click.
  async function openInputs() {
    if (!chosen.size) throw new Error("Tick at least one audio input.");
    const ids = [...chosen].sort((a, b) => (b === SYSTEM) - (a === SYSTEM));
    const streams = [];
    for (const id of ids) {
      try {
        if (id === SYSTEM) {
          const disp = await navigator.mediaDevices.getDisplayMedia({
            video: true,
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
          });
          disp.getVideoTracks().forEach((t) => t.stop());
          if (!disp.getAudioTracks().length) {
            stopStream(disp);
            throw new Error("No audio was shared. Choose a tab or screen and tick “Share audio”.");
          }
          streams.push(new MediaStream(disp.getAudioTracks()));
        } else {
          streams.push(await navigator.mediaDevices.getUserMedia({
            audio: { deviceId: { exact: id }, channelCount: 1, echoCancellation: true, noiseSuppression: true },
          }));
        }
      } catch (err) {
        streams.forEach(stopStream);
        if (err.message.startsWith("No audio")) throw err;
        if (id === SYSTEM) throw new Error("System audio sharing was cancelled or blocked.");
        throw new Error(err.name === "NotFoundError" || err.name === "OverconstrainedError"
          ? "The chosen microphone was not found." : "Microphone access was blocked.");
      }
    }
    return streams;
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
    const maxH = Math.max(12, (el.viz.clientHeight || 52) - 4);
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
      bars[i].style.height = Math.round(4 + levels[i] * (maxH - 4)) + "px";
      bars[i].style.opacity = live ? 0.55 + levels[i] * 0.45 : 0.35;
    }
    requestAnimationFrame(drawBars);
  }
  requestAnimationFrame(drawBars);

  // ---------- transcript ----------
  const clock = (s) => {
    s = Math.max(0, Math.floor(s));
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  };
  const colorFor = (speaker) => SPEAKER_COLORS[speaker % SPEAKER_COLORS.length];

  function renderDialogue() {
    const rows = [];
    let lastRec = null;
    for (const t of turns) {
      if (lastRec !== null && t.rec !== lastRec) {
        const sep = document.createElement("div");
        sep.className = "sep";
        sep.textContent = "New recording";
        rows.push(sep);
      }
      lastRec = t.rec;
      const row = document.createElement("div");
      row.className = "turn";
      row.style.setProperty("--c", colorFor(t.speaker));
      const marker = document.createElement("span");
      marker.className = "marker";
      const body = document.createElement("div");
      body.className = "body";
      const who = document.createElement("div");
      who.className = "who";
      who.textContent = `Speaker ${t.speaker + 1}`;
      const when = document.createElement("span");
      when.className = "when";
      when.textContent = clock(t.start);
      who.appendChild(when);
      const p = document.createElement("p");
      p.textContent = t.text;
      body.append(who, p);
      row.append(marker, body);
      rows.push(row);
    }
    el.dialogue.replaceChildren(...rows);
  }

  function renderTranscript() {
    const multi = mode === "multi";
    el.single.hidden = multi;
    el.dialogue.hidden = !multi;
    el.placeholder.hidden = hasContent();
    el.placeholder.textContent = multi
      ? "Press Record and start talking. The dialogue appears here after you press Stop, one row per speaker."
      : "Press Record and start talking. Your words will appear here.";

    let words;
    if (multi) {
      renderDialogue();
      words = turns.reduce((n, t) => n + (t.text.trim() ? t.text.trim().split(/\s+/).length : 0), 0);
    } else {
      const trimmed = text.replace(/\s+$/, "");
      const cut = trimmed.lastIndexOf(" ") + 1;
      el.stable.textContent = text.slice(0, cut);
      el.live.textContent = text.slice(cut);
      el.transcript.classList.toggle("recording", !!session && !finishing);
      words = trimmed ? trimmed.split(/\s+/).length : 0;
    }
    el.count.textContent = words + (words === 1 ? " word" : " words");
    el.transcript.scrollTop = el.transcript.scrollHeight;
  }

  // ---------- status ----------
  function setStatus(state, message) {
    el.status.dataset.state = state;
    el.statusText.textContent = message;
  }
  const idleStatus = () => setStatus("idle", hasContent() ? "Stopped" : "Ready");

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
    timerId = setInterval(() => { el.timer.textContent = clock((Date.now() - t0) / 1000); }, 250);
  }
  function stopTimer() { clearInterval(timerId); timerId = null; }

  // ---------- recording ----------
  async function start() {
    if (mode === "multi" && info.diarization === "external" && !info.diarizer) {
      setStatus("error", "Choose a diarization model first.");
      return;
    }
    el.btn.disabled = true;
    updateControls();
    setStatus("idle", "Connecting…");
    let streams;
    try {
      streams = await openInputs();
    } catch (err) {
      fail(err.message);
      return;
    }

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/transcribe`);
    ws.binaryType = "arraybuffer";
    const s = { streams, ws, pending: [], ready: false, ctx: null, node: null, analyser: null, rec: ++recordings };
    session = s;
    // "Stop sharing" in the browser bar (or an unplugged microphone) ends the recording
    streams.flatMap((st) => st.getAudioTracks()).forEach((t) => t.addEventListener("ended", () => {
      if (session === s && !finishing && s.streams.every((st) => st.getAudioTracks().every((x) => x.readyState === "ended"))) stop();
    }));

    ws.onopen = () => ws.send(JSON.stringify({
      type: "start",
      mode,
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
        renderTranscript();
      } else if (msg.type === "turns") {
        for (const t of msg.turns) turns.push({ ...t, rec: s.rec });
        renderTranscript();
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
      // all ticked inputs are mixed into one signal
      const source = s.ctx.createGain();
      source.gain.value = s.streams.length > 1 ? 0.7 : 1;
      s.streams.forEach((st) => s.ctx.createMediaStreamSource(st).connect(source));
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
    if (mode === "single" && text && !/\s$/.test(text)) text += " ";
    setRecordingUI(true);
    updateControls();
    setStatus("live", streaming
      ? "Listening and transcribing"
      : (mode === "multi" ? "Recording. The dialogue appears when you press Stop"
                          : "Recording. Text appears when you press Stop"));
    startTimer();
    el.btn.disabled = false;
    renderTranscript();
  }

  function stop() {
    const s = session;
    if (!s || finishing) return;
    finishing = true;
    el.btn.disabled = true;
    updateControls();
    stopTimer();
    setRecordingUI(false);
    setStatus("idle", streaming ? "Finishing…"
      : (mode === "multi" ? "Transcribing and identifying speakers…" : "Transcribing…"));
    s.streams.forEach(stopStream);
    if (s.node) s.node.disconnect();
    if (s.ws.readyState === WebSocket.OPEN) s.ws.send(JSON.stringify({ type: "stop" }));
    // safety net; an offline model may need a while for a long recording
    setTimeout(() => { if (session === s) endSession(s); }, streaming ? 8000 : 300000);
  }

  function endSession(s) {
    if (session !== s) return;
    cleanup(s);
    session = null;
    finishing = false;
    if (s.limitHit) setStatus("idle", "Stopped at the recording length limit");
    else idleStatus();
    updateControls();
    renderTranscript();
  }

  function fail(message) {
    const s = session;
    if (s) { cleanup(s); session = null; }
    finishing = false;
    stopTimer();
    setRecordingUI(false);
    setStatus("error", message);
    updateControls();
    renderTranscript();
  }

  function cleanup(s) {
    s.streams.forEach(stopStream);
    try { if (s.ctx) s.ctx.close(); } catch (_) {}
    try { s.ws.onclose = null; s.ws.close(); } catch (_) {}
    stopTimer();
    setRecordingUI(false);
  }

  // ---------- buttons ----------
  el.btn.addEventListener("click", () => (session ? stop() : start()));
  el.clear.addEventListener("click", () => {
    text = "";
    turns = [];
    el.timer.textContent = "00:00";
    renderTranscript();
    if (!session) setStatus("idle", "Ready");
  });
  el.copy.addEventListener("click", async () => {
    const content = mode === "multi"
      ? turns.map((t) => `Speaker ${t.speaker + 1}: ${t.text}`).join("\n")
      : text.trim();
    try {
      await navigator.clipboard.writeText(content);
      const old = el.copy.textContent;
      el.copy.textContent = "Copied";
      setTimeout(() => (el.copy.textContent = old), 1200);
    } catch (_) {
      setStatus("error", "Copy is not allowed by the browser.");
    }
  });

  // ---------- menus and badges ----------
  function option(value, label, disabled) {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    o.disabled = !!disabled;
    return o;
  }

  function problemNote(item) {
    if (!item.downloaded) return " (not downloaded)";
    if (!item.available) return ` (needs: pip install ${item.needs})`;
    return "";
  }

  function renderMenus() {
    if (!info) return;
    const multi = mode === "multi";

    // model menu: only models that can do the chosen kind of recording
    const list = info.models.filter((m) => m.modes.includes(mode));
    const options = list.map((m) => option(m.id, m.label + problemNote(m), !m.downloaded || !m.available));
    const loadedHere = list.some((m) => m.id === info.model);
    if (!loadedHere) options.unshift(option("", "Choose a model…", true));
    el.modelSelect.replaceChildren(...options);
    el.modelSelect.value = loadedHere ? info.model : "";

    // language menu: from the loaded model
    const previous = el.langSelect.value;
    if (loadedHere && info.languages.length) {
      el.langSelect.replaceChildren(...info.languages.map((l) => option(l.code, l.label)));
      const codes = info.languages.map((l) => l.code);
      el.langSelect.value = codes.includes(previous) ? previous : info.default_language;
    } else {
      el.langSelect.replaceChildren(option("", loadedHere ? (info.language_note || "Not available") : "—"));
    }

    // diarization menu: only for multiple speakers
    el.diarField.hidden = !multi;
    if (multi) {
      if (loadedHere && info.diarization === "builtin") {
        el.diarSelect.replaceChildren(option("", "Built into the model"));
        el.diarSelect.value = "";
      } else {
        const d = info.diarizers.map((x) => option(x.id, x.label + problemNote(x), !x.downloaded || !x.available));
        if (!info.diarizer) d.unshift(option("", "Choose a diarization model…", true));
        el.diarSelect.replaceChildren(...d);
        el.diarSelect.value = info.diarizer || "";
      }
    }
    updateControls();
    renderBadges(loadedHere);
  }

  function renderBadges(loadedHere = true) {
    if (!info) return;
    const multi = mode === "multi";
    el.bDevice.textContent = info.cuda ? `${info.device} · CUDA` : "CPU (no CUDA)";
    el.bDevice.classList.toggle("warn", !info.cuda);
    el.bSpeakers.textContent = multi ? "Multiple speakers" : "Single speaker";

    el.bModel.textContent = loadedHere ? `Model: ${info.model_label}` : "Model: none chosen for this mode";
    el.bModel.classList.toggle("warn", !loadedHere);

    el.bDiar.hidden = !multi;
    if (multi) {
      if (loadedHere && info.diarization === "builtin") {
        el.bDiar.textContent = "Diarization: built into the model";
        el.bDiar.classList.remove("warn");
      } else if (info.diarizer) {
        const d = info.diarizers.find((x) => x.id === info.diarizer);
        el.bDiar.textContent = `Diarization: ${d ? d.label : info.diarizer}`;
        el.bDiar.classList.remove("warn");
      } else {
        el.bDiar.textContent = "Diarization: none chosen";
        el.bDiar.classList.add("warn");
      }
    }

    el.bMode.textContent = info.streaming
      ? `Live · ${info.latency_ms} ms latency`
      : `Text appears when you press Stop · up to ${Math.round(info.max_seconds / 60)} min`;

    if (loadedHere && info.languages.length) {
      const chosen = info.languages.find((l) => l.code === el.langSelect.value);
      el.bLang.textContent = `Language: ${chosen ? chosen.label : "—"}`;
    } else {
      el.bLang.textContent = `Language: ${loadedHere ? (info.language_note || "—") : "—"}`;
    }
  }

  function applyInfo(next) {
    info = next;
    streaming = info.streaming;
    renderMenus();
  }

  async function post(url, body, message) {
    busyLoading = true;
    updateControls();
    setStatus("idle", message);
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        let detail = r.statusText;
        try { const j = await r.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
        throw new Error(detail);
      }
      applyInfo(await r.json());
      return true;
    } catch (err) {
      setStatus("error", String(err.message || err));
      renderMenus();
      return false;
    } finally {
      busyLoading = false;
      updateControls();
    }
  }

  const loadModel = (id) => post("/api/model", { id }, "Loading model… (a large model can take a minute)");
  const loadDiarizer = (id) => post("/api/diarizer", { id }, "Loading diarization model…");

  // Make sure the loaded model and the diarization model fit the chosen kind of recording.
  async function ensureSetup() {
    if (!info.modes.includes(mode)) {
      const candidate = info.models.find((m) => m.modes.includes(mode) && m.downloaded && m.available);
      if (!candidate) {
        setStatus("error", "No downloaded model supports this. See the README for the download commands.");
        renderMenus();
        return;
      }
      if (!(await loadModel(candidate.id))) return;
    }
    if (mode === "multi" && info.diarization === "external" && !info.diarizer) {
      const d = info.diarizers.find((x) => x.downloaded && x.available);
      if (!d) {
        setStatus("error", "No diarization model is ready. Download one (see the README) or choose VibeVoice.");
        return;
      }
      if (!(await loadDiarizer(d.id))) return;
    }
    idleStatus();
  }

  function renderModeToggle() {
    const multi = mode === "multi";
    el.modeBtn.setAttribute("aria-checked", String(multi));
    el.modeText.textContent = multi ? "Multiple" : "Single";
    el.modeKnob.textContent = multi ? "2+" : "1";
  }

  el.modeBtn.addEventListener("click", async () => {
    mode = mode === "multi" ? "single" : "multi";
    try { localStorage.setItem("speakers-mode", mode); } catch (_) {}
    renderModeToggle();
    renderMenus();
    renderTranscript();
    await ensureSetup();
  });

  el.modelSelect.addEventListener("change", async () => {
    const wanted = el.modelSelect.value;
    if (!info || !wanted || wanted === info.model) return;
    if (await loadModel(wanted)) await ensureSetup();
    else renderMenus();
  });

  el.diarSelect.addEventListener("change", async () => {
    const wanted = el.diarSelect.value;
    if (!info || !wanted || wanted === info.diarizer) return;
    if (await loadDiarizer(wanted)) idleStatus();
    else renderMenus();
  });

  el.langSelect.addEventListener("change", () => renderBadges());

  updateControls();
  initInputs();
  // Come back in the Single / Multiple position used last time.
  async function restoreMode() {
    let saved = null;
    try { saved = localStorage.getItem("speakers-mode"); } catch (_) {}
    if (saved !== "multi" || mode === "multi") return;
    mode = "multi";
    renderModeToggle();
    renderMenus();
    renderTranscript();
    await ensureSetup();
  }

  fetch("/api/info").then((r) => r.json()).then(async (i) => { applyInfo(i); await restoreMode(); }).catch(() => {
    el.bDevice.textContent = "Server not reachable";
    el.bDevice.classList.add("warn");
  });

  renderTranscript();
})();
