/* ==========================================================================
   MoScribe UI (kmail brand)
   vanilla JS, no build step. Talks to the local FastAPI backend:
     GET  /api/health            engine + speaker-model status
     POST /api/transcribe        multipart upload -> job id
     WS   /ws/jobs/{id}          streaming tokens + chunks
     WS   /ws/live               live mic windows (24 kHz WAV)
     GET  /api/history           past transcripts
   ========================================================================== */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const WS_BASE = location.protocol === "https:" ? `wss://${location.host}` : `ws://${location.host}`;
  const SAMPLE_RATE = 24000;
  const SPK_COLORS = ["var(--spk0)", "var(--spk1)", "var(--spk2)",
                      "var(--spk3)", "var(--spk4)", "var(--spk5)"];

  /* ------------------------------------------------------------- state */

  const DEFAULTS = {
    threads: Math.max(2, Math.min(8, navigator.hardwareConcurrency || 8)),
    decoding: "greedy",
    temperature: 0.7,
    topP: 0.9,
    context: "",
    promptFormat: "text",
    chunkSec: 12,
    overlapSec: 2,
    speakerSep: "on",
    activePreset: null,
    layout: "auto",
    sensitivity: 0.78,
    maxSpeakers: 6,
  };

  // Context presets. This field is *biasing context* for the ASR, not an
  // instruction to a chat model: the useful content is domain vocabulary —
  // names, jargon, acronyms and the register the speaker is likely to use.
  const PRESETS = [
    {
      id: "meeting",
      layout: "bubbles",
      label: "Meeting",
      text: "Business meeting. Terms: agenda, minutes, action items, deadline, stakeholder, budget, quarterly results, roadmap, follow-up, sign-off, EBITDA, KPI. Names: Anna Kovács, Thomas Berger, Michael Huber. German and English spoken; keep speaker turns and numbers exact.",
    },
    {
      id: "standup",
      layout: "compact",
      label: "Engineering stand-up",
      text: "Engineering stand-up. Terms: deploy, merge request, pull request, commit, branch, rollback, CI pipeline, staging, production, latency, regression, refactor, migration, unit test, flaky test, incident, runbook, Kubernetes, Postgres, API, backend, frontend.",
    },
    {
      id: "interview",
      layout: "qa",
      label: "Interview",
      text: "Job interview. Terms: role, responsibilities, notice period, salary expectation, probation period, onboarding, team size, reporting line, remote work, benefits, references, CV, portfolio, career path.",
    },
    {
      id: "lecture",
      layout: "document",
      label: "Lecture",
      text: "Academic lecture. Terms: theorem, hypothesis, methodology, empirical, literature review, sample size, correlation, significance, citation, seminar, syllabus, exam, thesis, supervisor, faculty. Expect long monologues and Latin-derived technical vocabulary.",
    },
    {
      id: "public-sector",
      layout: "document",
      label: "Austrian public sector",
      text: "Austrian public administration, formal German (Amtsdeutsch). Terms: Bescheid, Akt, Erledigung, Genehmigung, Frist, Antrag, Sachbearbeiter, Abteilung, Magistrat, Landeshauptmann, Förderung, Richtlinie, Verordnung, Datenschutz-Grundverordnung, DSGVO, Aktenvermerk, Zuständigkeit, Amtssignatur. Keep names, dates, file numbers and amounts exactly as spoken.",
    },
    {
      id: "clinic",
      layout: "cards",
      label: "Clinical intake",
      text: "Clinical consultation. Terms: anamnesis, symptoms, onset, diagnosis, indication, contraindication, dosage, medication, blood pressure, referral, follow-up appointment, findings, laboratory values, allergies, previous conditions. Transcribe numbers and units verbatim; never paraphrase.",
    },
    {
      id: "legal",
      layout: "document",
      label: "Legal / contract",
      text: "Legal review. Terms: clause, paragraph, party, obligation, liability, warranty, indemnity, termination, governing law, jurisdiction, confidentiality, force majeure, amendment, annex, effective date, compliance.",
    },
    {
      id: "support",
      layout: "compact",
      label: "Support call",
      text: "Customer support call. Terms: ticket number, account, subscription, invoice, refund, replacement, warranty claim, escalation, workaround, software version, error message, serial number, callback. Transcribe identifiers, e-mail addresses and numbers character by character.",
    },
    {
      id: "podcast",
      layout: "bubbles",
      label: "Podcast / panel",
      text: "Recorded conversation or panel discussion with several speakers. Expect filler words, interruptions and overlapping speech. Terms: episode, guest, host, sponsor, audience question, chapter, transcript. Preserve colloquial phrasing; do not smooth it.",
    },
    {
      id: "workshop",
      layout: "cards",
      label: "Workshop",
      text: "Workshop or brainstorming session. Terms: whiteboard, sticky note, cluster, prioritise, vote, timebox, breakout group, facilitation, next steps, parking lot, retrospective, lessons learned.",
    },
  ];

  let settings = { ...DEFAULTS };
  try {
    const saved = JSON.parse(localStorage.getItem("vibescribe.settings") || "{}");
    settings = { ...settings, ...saved };
  } catch { /* ignore */ }

  const S = {
    mode: "live",
    turns: [],
    segments: [],
    committed: "",
    previewTail: "",
    duration: 0,
    rtf: null,
    busy: false,
    phase: "idle",
    file: null,
    fileDuration: null,
    totalChunks: 0,
    doneChunks: 0,
    currentChunk: -1,
    names: {},
    nameGuesses: new Map(),
    aiStatus: null,
    jobSocket: null,
    follow: true,
    health: null,
    mic: {
      active: false, ctx: null, stream: null, node: null, gain: null, ws: null,
      buf: [], bufLen: 0, inflight: false, firstWindow: true, t0: 0,
      timer: null, level: 0, finishedWaiter: null,
    },
  };

  /* ----------------------------------------------------------- helpers */

  const fmtTime = (sec) => {
    if (!isFinite(sec) || sec < 0) sec = 0;
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
             : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };
  const fmtStamp = (sec) => (sec == null ? "" : fmtTime(sec));
  const fmtBytes = (b) => {
    if (!b) return "0 B";
    const u = ["B", "KB", "MB", "GB"];
    const i = Math.min(u.length - 1, Math.floor(Math.log(b) / Math.log(1024)));
    return `${(b / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
  };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  let toastTimer = null;
  function toast(msg, isErr = false) {
    const el = $("toast");
    el.textContent = msg;
    el.className = "toast on" + (isErr ? " err" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = "toast"; }, isErr ? 5200 : 2800);
  }

  function speakerLabel(spk) {
    if (spk == null || spk < 0) return "Speaker";
    return S.names[spk] || `Speaker ${spk + 1}`;
  }
  const speakerColor = (spk) => SPK_COLORS[(spk == null || spk < 0 ? 5 : spk) % SPK_COLORS.length];

  function settingsPayload() {
    return {
      threads: settings.threads,
      decoding: settings.decoding,
      temperature: settings.temperature,
      top_p: settings.topP,
      context: settings.context,
      prompt_format: settings.promptFormat,
      speaker_separation: settings.speakerSep,
      speaker_sensitivity: settings.sensitivity,
      max_speakers: settings.maxSpeakers,
    };
  }

  function saveSettings() {
    try { localStorage.setItem("vibescribe.settings", JSON.stringify(settings)); } catch { /* quota */ }
  }

  /* ---------------------------------------------------------- rendering */

  /* -------------------------------------------------- speaker name guessing

     The ASR has no notion of identity, so a name can only come from what is
     actually said. Two sources, in order of trust:
       1. self-introductions inside a turn ("this is Anna", "ich bin Thomas") —
          that turn's speaker is almost certainly the named person;
       2. a name list in the applied prompt ("Names: Anna, Thomas") — assigned
          to speakers in order of first appearance, which is a guess and is
          labelled as one.
     Nothing is applied automatically: a wrong name in a transcript is worse
     than "Speaker 2", so the UI offers it as a suggestion.
  */

  const NAME_STOP = new Set([
    "the", "a", "an", "and", "but", "so", "not", "no", "yes", "okay", "ok", "well",
    "good", "great", "glad", "sorry", "afraid", "sure", "here", "there", "this",
    "that", "what", "who", "why", "how", "when", "just", "also", "still", "very",
    "thanks", "thank", "hello", "hi", "hey", "morning", "evening", "everyone",
    "everybody", "all", "going", "talking", "calling", "speaking", "now", "then",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "german", "english",
    "from", "with", "at", "in", "on", "of", "for", "about", "regarding", "to",
    "by", "near", "because", "when", "while", "if", "which", "who", "whom",
    "today", "tomorrow", "yesterday", "again", "too", "already", "actually",
  ]);

  const NAME_WORD = "[A-ZÄÖÜ][\\p{L}'’\\-]+";
  const NAME_RES = [
    new RegExp(`\\bmy name(?:'s| is)\\s+(${NAME_WORD}(?:\\s+${NAME_WORD})?)`, "iu"),
    new RegExp(`\\bthis is\\s+(${NAME_WORD}(?:\\s+${NAME_WORD})?)\\b`, "iu"),
    new RegExp(`\\b(?:i(?:'m| am))\\s+(${NAME_WORD}(?:\\s+${NAME_WORD})?)\\b`, "iu"),
    new RegExp(`\\bmein name ist\\s+(${NAME_WORD}(?:\\s+${NAME_WORD})?)`, "iu"),
    new RegExp(`\\bich bin\\s+(?:der |die |das )?(${NAME_WORD}(?:\\s+${NAME_WORD})?)\\b`, "iu"),
    new RegExp(`\\bhier (?:spricht|ist)\\s+(${NAME_WORD}(?:\\s+${NAME_WORD})?)`, "iu"),
    new RegExp(`\\b(${NAME_WORD})\\s+(?:here|speaking|am telefon|hier)\\b`, "iu"),
  ];

  function cleanName(raw) {
    if (!raw) return null;
    const words = raw.trim().split(/\s+/).filter(Boolean);
    const kept = [];
    for (const [i, w] of words.entries()) {
      const bare = w.replace(/[^\p{L}'’\-]/gu, "");
      if (!bare || NAME_STOP.has(bare.toLowerCase())) break;
      // The trigger phrase is matched case-insensitively, so a following word is
      // only part of the name when the speaker actually capitalised it —
      // otherwise "this is Sabine from the vendor" yields "Sabine From".
      if (i > 0 && !/^[A-ZÄÖÜ]/.test(bare)) break;
      kept.push(bare.length <= 2 ? bare.toUpperCase() : bare[0].toUpperCase() + bare.slice(1));
      if (kept.length === 2) break;
    }
    return kept.length ? kept.join(" ") : null;
  }

  function promptNames(context) {
    const m = /(?:names?|namen|sprecher(?:in)?|teilnehmer(?:in)?)\s*:\s*([^.\n|]+)/i.exec(context || "");
    if (!m) return [];
    return m[1]
      .split(/\s*,\s*|\s+\/\s+|\s+und\s+|\s+and\s+/i)
      .map((x) => cleanName(x))
      .filter(Boolean);
  }

  let nameCache = { sig: "", result: null };

  function suggestNames(turns, context) {
    const sig = turns.map((t) => `${t.speaker}:${t.text}`).join("|") + "#" + (context || "");
    if (nameCache.sig === sig) return nameCache.result;

    const firstSeen = [];
    const named = new Map();      // speaker -> {name, confidence, source}
    const taken = new Set();

    for (const turn of turns) {
      const spk = turn.speaker;
      if (spk == null || spk < 0) continue;
      if (!firstSeen.includes(spk)) firstSeen.push(spk);
      if (named.has(spk)) continue;
      for (const re of NAME_RES) {
        re.lastIndex = 0;
        const m = re.exec(turn.text || "");
        if (!m) continue;
        const name = cleanName(m[1]);
        if (!name) continue;
        const key = name.toLowerCase();
        if (taken.has(key)) break;
        taken.add(key);
        named.set(spk, { name, confidence: 0.9, source: "introduction" });
        break;
      }
    }

    // Fall back to names listed in the prompt, in speaking order.
    const listed = promptNames(context).filter((nm) => !taken.has(nm.toLowerCase()));
    for (const spk of firstSeen) {
      if (named.has(spk) || !listed.length) continue;
      const name = listed.shift();
      taken.add(name.toLowerCase());
      named.set(spk, { name, confidence: 0.4, source: "prompt" });
    }

    const result = named;
    nameCache = { sig, result };
    return result;
  }

  function transcriptText() {
    if (S.turns.length) {
      return S.turns.map((t) => `${speakerLabel(t.speaker)}: ${t.text}`).join("\n");
    }
    return (S.committed || "").trim();
  }

  function highlight(text, query) {
    const safe = esc(text);
    if (!query) return safe;
    try {
      const re = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
      return safe.replace(re, "<mark>$1</mark>");
    } catch { return safe; }
  }

  // Keyed DOM so arriving text never rebuilds what is already on screen. The
  // previous version re-created every turn on each streamed token, which is what
  // made the transcript flicker and re-run its entry animation.
  const dom = { turns: [], preview: null, query: "", layout: "" };

  const LAYOUTS = ["auto", "bubbles", "document", "cards", "compact", "qa"];

  // The template in use decides how the transcript reads: a meeting is a chat,
  // a lecture is a document, an interview is question/answer, a stand-up is a
  // dense list. Manual choice wins over the preset.
  function activeLayout() {
    if (settings.layout && settings.layout !== "auto") return settings.layout;
    const preset = presetById(settings.activePreset);
    if (preset && preset.layout) return preset.layout;
    return "bubbles";
  }

  let renderQueued = false;
  let rafId = 0;
  let paintTimer = 0;
  // Coalesce bursts of tokens into one paint, but never depend on rAF alone:
  // browsers starve it in background tabs, which would freeze the transcript.
  function render() {
    if (renderQueued) return;
    renderQueued = true;
    const flush = () => {
      if (!renderQueued) return;
      renderQueued = false;
      cancelAnimationFrame(rafId);
      clearTimeout(paintTimer);
      syncTranscript();
    };
    rafId = requestAnimationFrame(flush);
    paintTimer = setTimeout(flush, 80);
  }

  function syncTranscript() {
    const host = $("transcript");
    if (!host) return;
    const query = $("search").value;
    const wasNearBottom = host.scrollHeight - host.scrollTop - host.clientHeight < 90;
    const empty = $("emptyState");
    if (empty) empty.classList.toggle("hidden", !!(S.turns.length || S.committed || S.previewTail));

    const preview = ensurePreview(host);
    S.nameGuesses = suggestNames(S.turns, settings.context);

    const layout = activeLayout();
    if (dom.layout !== layout) { host.dataset.layout = layout; dom.layout = layout; }

    for (let i = 0; i < S.turns.length; i++) {
      const turn = S.turns[i];
      const sig = `${turn.speaker}|${turn.start}|${query}`;
      let rec = dom.turns[i];
      if (!rec) {
        const node = createTurn(turn);
        host.insertBefore(node, preview);
        rec = dom.turns[i] = { node, sig: "", text: null };
      }
      if (rec.sig !== sig) {
        paintTurn(rec.node, turn, query);
        rec.sig = sig;
        rec.text = turn.text;
      } else if (rec.text !== turn.text) {
        paintText(rec.node, turn.text, query);
        rec.text = turn.text;
      }
    }
    while (dom.turns.length > S.turns.length) {
      const gone = dom.turns.pop();
      gone.node.remove();
    }

    const tail = S.previewTail || "";
    if (tail) {
      preview.hidden = false;
      if (preview.textContent !== tail) preview.textContent = tail;
    } else {
      preview.hidden = true;
      if (preview.textContent) preview.textContent = "";
    }

    renderLegend();
    renderStats();
    updateDocMeta();

    if (S.follow && (wasNearBottom || S.mic.active)) host.scrollTop = host.scrollHeight;
  }

  function ensurePreview(host) {
    if (dom.preview && dom.preview.isConnected) return dom.preview;
    const p = document.createElement("div");
    p.className = "preview";
    p.hidden = true;
    host.appendChild(p);
    dom.preview = p;
    return p;
  }

  function createTurn(turn) {
    const el = document.createElement("article");
    el.className = "turn";
    el.style.setProperty("--c", speakerColor(turn.speaker));

    const head = document.createElement("div");
    head.className = "turnHead";

    const chip = document.createElement("button");
    chip.className = "spkChip";
    chip.title = "Click to rename this speaker";
    head.appendChild(chip);

    const hint = document.createElement("button");
    hint.className = "spkHint";
    hint.hidden = true;
    head.appendChild(hint);

    const time = document.createElement("span");
    time.className = "turnTime";
    head.appendChild(time);

    const body = document.createElement("p");
    body.className = "turnText";

    el.append(head, body);
    return el;
  }

  function paintTurn(el, turn, query) {
    const colour = speakerColor(turn.speaker);
    el.dataset.side = (turn.speaker != null && turn.speaker >= 0 && turn.speaker % 2) ? "right" : "left";
    el.style.setProperty("--c", colour);
    const chip = el.querySelector(".spkChip");
    chip.style.setProperty("--c", colour);
    chip.textContent = speakerLabel(turn.speaker);
    chip.onclick = () => beginRename(chip, turn.speaker);

    const hint = el.querySelector(".spkHint");
    const guess = S.nameGuesses.get(turn.speaker);
    const suggestable = guess && !S.names[turn.speaker];
    hint.hidden = !suggestable;
    if (suggestable) {
      hint.textContent = `${guess.name}?`;
      hint.title = guess.source === "introduction"
        ? `Heard in this speaker's own words — click to use "${guess.name}"`
        : `Name from the prompt, matching speaking order — click to use "${guess.name}"`;
      hint.style.setProperty("--c", colour);
      hint.onclick = (ev) => {
        ev.stopPropagation();
        S.names[turn.speaker] = guess.name;
        toast(`Named this speaker ${guess.name}`);
        render();
      };
    }
    el.querySelector(".turnTime").textContent = turn.start != null ? fmtStamp(turn.start) : "";
    el.querySelector(".turnText").style.setProperty("--c", colour);
    paintText(el, turn.text, query);
  }

  function paintText(el, text, query) {
    const body = el.querySelector(".turnText");
    if (!query) {                       // fast path: no markup, no parsing
      if (body.textContent !== text) body.textContent = text;
      return;
    }
    const html = highlight(text, query);
    if (body.innerHTML !== html) body.innerHTML = html;
  }

  function beginRename(chip, spk) {
    if (spk == null || spk < 0) return;
    const input = document.createElement("input");
    input.className = "spkInput";
    input.style.setProperty("--c", speakerColor(spk));
    input.value = S.names[spk] || `Speaker ${spk + 1}`;
    chip.replaceWith(input);
    input.focus();
    input.select();

    let settled = false;
    const finish = (save) => {
      if (settled) return;
      settled = true;
      document.removeEventListener("pointerdown", onOutside, true);
      if (save) {
        const v = input.value.trim();
        if (v && v !== `Speaker ${spk + 1}`) S.names[spk] = v;
        else delete S.names[spk];
      }
      render();
    };
    const onOutside = (e) => { if (e.target !== input) finish(true); };
    document.addEventListener("pointerdown", onOutside, true);

    input.onblur = () => finish(true);
    input.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    };
  }

  function renderLegend() {
    const legend = $("legend");
    if (!legend) return;
    const seen = [];
    S.turns.forEach((t) => {
      if (t.speaker != null && t.speaker >= 0 && !seen.includes(t.speaker)) seen.push(t.speaker);
    });
    if (seen.length < 2) { legend.classList.add("hidden"); legend.innerHTML = ""; return; }
    legend.classList.remove("hidden");
    legend.innerHTML = seen.map((spk) => {
      const n = S.turns.filter((t) => t.speaker === spk).length;
      return `<span class="legendChip" style="--c:${speakerColor(spk)}">${esc(speakerLabel(spk))} · ${n}</span>`;
    }).join("");
  }

  // The subtitle must describe what is on screen. Uploads set their own detail
  // line; anything else starts as the empty placeholder and would otherwise keep
  // saying "Nothing transcribed yet" next to a finished transcript.
  function updateDocMeta() {
    const meta = $("docMeta");
    if (!meta) return;
    const empty = !S.turns.length && !S.committed;
    if (empty) {
      meta.textContent = "Nothing transcribed yet";
      return;
    }
    if (meta.dataset.custom === "1") return;
    const speakers = new Set(S.turns.map((t) => t.speaker).filter((x) => x != null && x >= 0)).size;
    const last = S.turns.length ? S.turns[S.turns.length - 1] : null;
    const dur = S.duration || (last && last.end) || 0;
    const bits = [];
    if (dur) bits.push(fmtTime(dur));
    if (speakers) bits.push(`${speakers} speaker${speakers === 1 ? "" : "s"}`);
    bits.push(`${S.turns.length || 1} turn${S.turns.length === 1 ? "" : "s"}`);
    meta.textContent = bits.join(" · ");
  }

  function renderStats() {
    const text = S.turns.length
      ? S.turns.map((t) => t.text).join(" ")
      : S.committed;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    $("statWords").textContent = `${words} words`;
    $("statChars").textContent = `${text.length} chars`;
    const speakers = new Set(S.turns.map((t) => t.speaker).filter((s) => s != null && s >= 0)).size;
    $("statSpeakers").textContent = `${speakers} speaker${speakers === 1 ? "" : "s"}`;
    $("statTime").textContent = S.duration ? `${fmtTime(S.duration)} audio` : "—";

    const pill = $("pillRtf");
    if (S.rtf) {
      pill.dataset.state = S.rtf < 1 ? "ok" : "warn";
      pill.querySelector("em").textContent = `RTF ${Number(S.rtf).toFixed(2)}`;
    } else {
      pill.dataset.state = "idle";
      pill.querySelector("em").textContent = "—";
    }
  }

  // A stopped recorder is not a finished transcript: the tail window is still
  // being transcribed, and the UI has to say so until the server sends `final`.
  function setFinishing(on) {
    S.finishing = on;
    const chip = $("finishChip");
    if (chip) chip.hidden = !on;
    const orb = $("micBtn");
    if (orb) orb.classList.toggle("busy", on);
    if (on) setPhase("finishing", true);
  }

  function setPhase(text, busy = false) {
    const el = $("footHint");
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("busy", busy);
  }

  function resetSession() {
    S.turns = [];
    S.segments = [];
    S.committed = "";
    S.previewTail = "";
    S.duration = 0;
    S.rtf = null;
    S.names = {};
    S.currentChunk = -1;
    $("docTitle").textContent = "Untitled session";
    delete $("docMeta").dataset.custom;
    $("docMeta").textContent = "Nothing transcribed yet";
    setPhase("idle");
    render();
  }

  /* ------------------------------------------------------------- health */

  let wsWarned = false;

  function noteWebsocketSupport(d) {
    if (!d || !("websockets" in d) || wsWarned) return;
    if (d.websockets === null) {
      wsWarned = true;
      toast("This server has no WebSocket support — live transcription is disabled. "
          + "Install it with: pip install 'uvicorn[standard]'", true);
      const orb = $("micBtn");
      if (orb) {
        orb.disabled = true;
        orb.title = "server lacks WebSocket support (pip install 'uvicorn[standard]')";
      }
    }
  }

  async function pollHealth() {
    try {
      const r = await fetch("/api/health", { cache: "no-store" });
      const h = await r.json();
      S.health = h;
      noteWebsocketSupport(h);
      const e = h.engine || {};
      const pe = $("pillEngine");
      if (e.state === "ready") { pe.dataset.state = "ok"; pe.querySelector("em").textContent = "ready"; }
      else if (e.state === "loading") { pe.dataset.state = "warn"; pe.querySelector("em").textContent = "loading"; }
      else if (e.binary_ok === false) { pe.dataset.state = "err"; pe.querySelector("em").textContent = "no binary"; }
      else if (e.models_ok === false) { pe.dataset.state = "err"; pe.querySelector("em").textContent = "no models"; }
      else { pe.dataset.state = "warn"; pe.querySelector("em").textContent = "idle"; }

      const d = h.diarizer || {};
      const pd = $("pillSpeakers");
      pd.dataset.state = d.ok ? "ok" : "err";
      pd.querySelector("em").textContent = d.ok ? "resnet34" : "unavailable";

      const t = h.text_model || {};
      const pillAI = $("pillAI");
      if (pillAI) {
        pillAI.dataset.state = t.ok ? "ok" : (t.model_ok === false ? "err" : "warn");
        pillAI.querySelector("em").textContent = t.ok
          ? `bitnet ${Math.round((t.model_mb || 0) / 100) / 10} GB`
          : (t.model_ok === false ? "no model" : "cold");
        pillAI.title = t.ok ? "Local BitNet b1.58 2B ready" : (t.error || "starts on first use");
      }

      $("aboutEngine").textContent = e.binary
        ? `asr_stream_server · ${e.chunks_done || 0} windows transcribed · ${(e.sample_rate || 24000) / 1000} kHz · speaker model ${d.size_mb || "?"} MB`
        : "engine not built";
    } catch {
      const pe = $("pillEngine");
      pe.dataset.state = "err";
      pe.querySelector("em").textContent = "offline";
    }
  }

  /* --------------------------------------------------------------- tabs */

  function setMode(mode) {
    S.mode = mode;
    document.querySelectorAll(".seg[data-mode]").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", String(on));
    });
    $("livePanel").classList.toggle("hidden", mode !== "live");
    $("filePanel").classList.toggle("hidden", mode !== "file");
    if (mode !== "live" && S.mic.active) stopMic();
  }

  /* ---------------------------------------------------------- file mode */

  function setFile(f) {
    S.file = f || null;
    S.fileDuration = null;
    if (!f) {
      $("fileCard").classList.add("hidden");
      $("dropZone").classList.remove("hidden");
      return;
    }
    $("fileName").textContent = f.name;
    $("fileMeta").textContent = `${fmtBytes(f.size)} · ${f.type || "audio"}`;
    $("dropZone").classList.add("hidden");
    $("fileCard").classList.remove("hidden");
    $("progressFill").style.width = "0%";
    $("jobProgress").textContent = "ready";
    probeDuration(f);
  }

  async function probeDuration(f) {
    try {
      const url = URL.createObjectURL(f);
      const audio = new Audio();
      audio.preload = "metadata";
      audio.src = url;
      await new Promise((res, rej) => {
        audio.onloadedmetadata = res;
        audio.onerror = rej;
        setTimeout(rej, 4000);
      });
      if (isFinite(audio.duration) && audio.duration > 0) {
        S.fileDuration = audio.duration;
        $("fileMeta").textContent = `${fmtBytes(f.size)} · ${fmtTime(audio.duration)}`;
      }
      URL.revokeObjectURL(url);
    } catch { /* container may be undecodable in-browser; server decides */ }
  }

  async function startJob() {
    if (!S.file || S.busy) return;
    S.busy = true;
    $("transcribeBtn").disabled = true;
    $("transcribeBtn").textContent = "Uploading…";
    $("progressFill").style.width = "0%";
    $("jobProgress").textContent = "uploading audio";
    setPhase("uploading", true);

    const fd = new FormData();
    fd.append("file", S.file);
    Object.entries(settingsPayload()).forEach(([k, v]) => fd.append(k, v));
    fd.append("chunk_sec", settings.chunkSec);
    fd.append("overlap_sec", settings.overlapSec);

    let meta;
    try {
      const res = await fetch("/api/transcribe", { method: "POST", body: fd });
      meta = await res.json();
      if (!res.ok) throw new Error(meta.error || `HTTP ${res.status}`);
    } catch (err) {
      toast(String(err.message || err), true);
      finishJobUI();
      return;
    }

    resetSession();
    S.duration = meta.duration;
    S.totalChunks = meta.chunks;
    S.doneChunks = 0;
    $("docTitle").textContent = S.file.name;
    $("docMeta").dataset.custom = "1";
    $("docMeta").textContent = `${fmtTime(meta.duration)} · ${meta.chunks} windows · ${settings.speakerSep === "on" ? "speakers on" : "plain text"}`;
    $("transcribeBtn").textContent = "Transcribing…";
    render();

    const ws = new WebSocket(`${WS_BASE}/ws/jobs/${meta.job_id}`);
    S.jobSocket = ws;

    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "start") {
        $("jobProgress").textContent = `window 1 / ${m.chunks}`;
        setPhase("transcribing", true);
      } else if (m.type === "token") {
        if (m.chunk !== S.currentChunk) { S.currentChunk = m.chunk; S.previewTail = ""; }
        S.previewTail += m.text;
        render();
      } else if (m.type === "chunk_done") {
        S.committed = m.text;
        S.previewTail = "";
        S.segments.push(m.segment);
        S.doneChunks = S.segments.length;
        const pct = Math.round((S.doneChunks / S.totalChunks) * 100);
        $("progressFill").style.width = pct + "%";
        $("jobProgress").textContent = `window ${S.doneChunks} / ${S.totalChunks}`;
        if (m.segment.rtf) S.rtf = m.segment.rtf;
        if (settings.speakerSep === "on") S.turns = buildTurns(S.segments);
        render();
      } else if (m.type === "phase" && m.phase === "diarizing") {
        $("jobProgress").textContent = "identifying speakers…";
        setPhase("analysing voices", true);
      } else if (m.type === "done") {
        S.committed = m.text || S.committed;
        S.previewTail = "";
        S.segments = m.segments || S.segments;
        S.turns = m.turns && m.turns.length ? m.turns : buildTurns(S.segments);
        if (!settings.speakerSep || settings.speakerSep === "off") S.turns = [];
        S.rtf = m.rtf || S.rtf;
        S.duration = m.duration || S.duration;
        $("progressFill").style.width = "100%";
        $("jobProgress").textContent = m.cancelled
          ? "stopped"
          : `done in ${m.elapsed}s · ${m.speakers || 0} speaker(s)`;
        setPhase(m.cancelled ? "stopped" : "ready");
        render();
        S.follow = false;
        $("transcript").scrollTop = 0;      // finished job: read from the top
        loadHistory();
        toast(m.cancelled ? "Stopped — kept what was transcribed" : "Transcript ready");
        finishJobUI();
      } else if (m.type === "error") {
        toast(m.message, true);
        $("jobProgress").textContent = "failed";
        finishJobUI();
      }
    };
    ws.onerror = () => { toast("WebSocket error — is the backend running?", true); finishJobUI(); };
    ws.onclose = () => { S.jobSocket = null; finishJobUI(); };
  }

  function finishJobUI() {
    S.busy = false;
    S.currentChunk = -1;
    $("transcribeBtn").disabled = false;
    $("transcribeBtn").textContent = "Transcribe";
  }

  function buildTurns(segs) {
    const turns = [];
    segs.forEach((s) => {
      const text = (s.text || "").trim();
      if (!text) return;
      const spk = s.speaker == null ? -1 : s.speaker;
      const last = turns[turns.length - 1];
      if (last && last.speaker === spk) {
        last.text = `${last.text} ${text}`.trim();
        if (s.end != null) last.end = s.end;
      } else {
        turns.push({ speaker: spk, text, start: s.start, end: s.end });
      }
    });
    return turns;
  }

  /* ----------------------------------------------------------- mic mode */

  const micSupported = () => !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  function downsample(data, srcRate, dstRate) {
    if (srcRate === dstRate) return data;
    if (srcRate % dstRate === 0) {
      const f = srcRate / dstRate;
      const out = new Float32Array(Math.floor(data.length / f));
      for (let i = 0; i < out.length; i++) {
        let sum = 0;
        for (let j = 0; j < f; j++) sum += data[i * f + j] || 0;
        out[i] = sum / f;
      }
      return out;
    }
    const ratio = srcRate / dstRate;
    const out = new Float32Array(Math.floor(data.length / ratio));
    for (let i = 0; i < out.length; i++) {
      const pos = i * ratio, i0 = Math.floor(pos), i1 = Math.min(i0 + 1, data.length - 1);
      out[i] = data[i0] + (data[i1] - data[i0]) * (pos - i0);
    }
    return out;
  }

  function encodeWav(samples, sr) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++, off += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buf;
  }

  function micBufAppend(chunk) { S.mic.buf.push(chunk); S.mic.bufLen += chunk.length; }

  function micBufTake(n) {
    const out = new Float32Array(n);
    let filled = 0;
    for (const c of S.mic.buf) {
      const take = Math.min(c.length, n - filled);
      out.set(c.subarray(0, take), filled);
      filled += take;
      if (filled >= n) break;
    }
    return out;
  }

  function micBufDrop(n) {
    let left = n;
    while (left > 0 && S.mic.buf.length) {
      const c = S.mic.buf[0];
      if (c.length <= left) { left -= c.length; S.mic.buf.shift(); }
      else { S.mic.buf[0] = c.subarray(left); left = 0; }
    }
    S.mic.bufLen = Math.max(0, S.mic.bufLen - n);
  }

  function findQuietCut(samples, target, search, win) {
    // Snap a window boundary to the quietest 200 ms around `target` so we don't
    // chop a word in half — mid-word cuts are the main source of seam gibberish.
    let best = target, bestRms = Infinity;
    const start = Math.max(0, target - search);
    const end = Math.min(samples.length, target + search);
    const step = Math.max(1, win >> 1);
    for (let pos = start; pos + win <= end; pos += step) {
      let sum = 0, n = 0;
      for (let i = pos; i < pos + win; i += 4) { sum += samples[i] * samples[i]; n++; }
      const rms = Math.sqrt(sum / Math.max(1, n));
      if (rms < bestRms) { bestRms = rms; best = pos + (win >> 1); }
    }
    return bestRms < 0.01 ? best : target;
  }

  function maybeFlushWindow() {
    const m = S.mic;
    if (!m.active || !m.ws || m.ws.readyState !== WebSocket.OPEN) return;
    if (m.inflight) return;
    const sr = m.ctx.sampleRate;
    const windowSec = m.firstWindow ? 8 : 10;
    const want = Math.round(windowSec * sr);
    if (m.bufLen < want) return;

    const search = Math.round(1.0 * sr);
    const avail = Math.min(m.bufLen, want + search);
    const raw = micBufTake(avail);
    const cut = findQuietCut(raw, want, search, Math.round(0.2 * sr));
    const keep = Math.round(Math.min(settings.overlapSec, 2.5) * sr);
    micBufDrop(Math.max(1, cut - keep));

    const ds = downsample(raw.subarray(0, cut), sr, SAMPLE_RATE);
    m.ws.send(encodeWav(ds, SAMPLE_RATE));
    m.inflight = true;
    m.firstWindow = false;
    setPhase("transcribing window", true);
  }

  async function startMic() {
    if (S.mic.active) return;
    if (!micSupported()) { toast("This browser cannot record audio", true); return; }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (err) {
      const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
      toast(denied
        ? "Microphone blocked — allow it in the address-bar site settings, then reload"
        : `Microphone unavailable: ${err && err.name ? err.name : err}`, true);
      return;
    }

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    let running = false;
    for (let i = 0; i < 6; i++) {
      try {
        await Promise.race([
          ctx.resume(),
          new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), 600)),
        ]);
      } catch { /* keep retrying */ }
      if (ctx.state === "running") { running = true; break; }
      await new Promise((r) => setTimeout(r, 250));
    }
    if (!running) {
      toast("Audio is blocked — click the page once and try again", true);
      stream.getTracks().forEach((t) => t.stop());
      ctx.close().catch(() => {});
      return;
    }

    const m = S.mic;
    Object.assign(m, {
      active: true, ctx, stream, buf: [], bufLen: 0, inflight: false,
      firstWindow: true, t0: Date.now(), level: 0,
    });

    resetSession();
    setFinishing(false);
    setMode("live");
    $("micBtn").classList.add("rec");
    $("micBtn").setAttribute("aria-label", "Stop recording");
    $("micHint").textContent = "Listening — speak normally";
    $("micTime").textContent = "00:00";
    setPhase("listening", true);

    m.timer = setInterval(() => {
      const sec = (Date.now() - m.t0) / 1000;
      $("micTime").textContent = fmtTime(sec);
      if (m.ws && m.ws.readyState === WebSocket.OPEN) {
        m.ws.send(JSON.stringify({ type: "tick", elapsed: sec }));
      }
    }, 1000);

    const ws = new WebSocket(`${WS_BASE}/ws/live`);
    m.ws = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "config", ...settingsPayload() }));
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "ready") return;
      if (msg.type === "token") {
        S.previewTail += msg.text;
        render();
      } else if (msg.type === "window_done") {
        S.previewTail = "";
        if (msg.turns && msg.turns.length) {
          S.turns = msg.turns;
          S.committed = msg.text || S.committed;
          S.segments = S.turns.slice();
        } else {
          S.committed = msg.text || S.committed;
          if (msg.appended) {
            S.segments.push({ index: msg.seq, text: msg.appended, speaker: msg.speaker,
                              start: null, end: null });
            S.turns = buildTurns(S.segments);
          }
        }
        if (!settings.speakerSep || settings.speakerSep === "off") S.turns = [];
        if (msg.rtf) S.rtf = msg.rtf;
        S.duration = (Date.now() - m.t0) / 1000;
        render();
        m.inflight = false;
        if (m.active) maybeFlushWindow();
        if (m.finishedWaiter) { m.finishedWaiter(); m.finishedWaiter = null; }
      } else if (msg.type === "final") {
        if (msg.text) S.committed = msg.text;
        if (msg.turns && msg.turns.length) {
          S.turns = msg.turns;
          S.segments = msg.turns.slice();
        }
        if (!settings.speakerSep || settings.speakerSep === "off") S.turns = [];
        if (msg.duration) S.duration = msg.duration;
        setFinishing(false);
        $("micHint").textContent = "Tap the mic and start talking";
        setPhase("ready");
        render();
        S.follow = false;
        toast(`Transcript complete · ${msg.words || 0} words`);
        setTimeout(loadHistory, 400);
      } else if (msg.type === "error") {
        toast(msg.message, true);
        m.inflight = false;
        if (m.finishedWaiter) { m.finishedWaiter(); m.finishedWaiter = null; }
      }
    };
    ws.onerror = () => toast("live socket error", true);
    ws.onclose = () => {
      if (S.finishing) {
        setFinishing(false);
        $("micHint").textContent = "Tap the mic and start talking";
        setPhase("ready");
        loadHistory();
      }
    };

    const source = ctx.createMediaStreamSource(stream);
    const node = ctx.createScriptProcessor(4096, 1, 1);
    const gain = ctx.createGain();
    gain.gain.value = 0;               // keep the graph alive without echoing back
    m.node = node;
    m.gain = gain;

    node.onaudioprocess = (e) => {
      if (!m.active) return;
      const data = e.inputBuffer.getChannelData(0);
      micBufAppend(new Float32Array(data));
      let sum = 0;
      for (let i = 0; i < data.length; i += 8) sum += data[i] * data[i];
      m.level = Math.min(1, Math.sqrt(sum / (data.length / 8)) * 6);
      const lvl = $("levelFill");
      if (lvl) lvl.style.width = (m.level * 100).toFixed(0) + "%";
      const lv = $("levelVal");
      if (lv) lv.textContent = `${Math.round(m.level * 100)}%`;
      maybeFlushWindow();
    };
    source.connect(node);
    node.connect(gain);
    gain.connect(ctx.destination);
  }

  async function stopMic() {
    const m = S.mic;
    if (!m.active) return;
    m.active = false;
    clearInterval(m.timer);
    $("micBtn").classList.remove("rec");
    $("micBtn").setAttribute("aria-label", "Start recording");
    $("micHint").textContent = "Processing the last window…";
    $("levelFill").style.width = "0%";
    $("levelVal").textContent = "—";
    setPhase("finishing", true);

    const sr = m.ctx ? m.ctx.sampleRate : SAMPLE_RATE;
    if (m.bufLen > Math.round(1.2 * sr) && m.ws && m.ws.readyState === WebSocket.OPEN) {
      const raw = micBufTake(m.bufLen);
      micBufDrop(m.bufLen);
      const ds = downsample(raw, sr, SAMPLE_RATE);
      m.ws.send(encodeWav(ds, SAMPLE_RATE));
      m.inflight = true;
    }

    try { m.node.disconnect(); m.gain.disconnect(); } catch { /* already gone */ }
    try { m.stream.getTracks().forEach((t) => t.stop()); } catch { /* already gone */ }
    const ctx = m.ctx;
    setTimeout(() => { try { ctx.close(); } catch { /* noop */ } }, 800);

    S.duration = (Date.now() - m.t0) / 1000;

    const ws = m.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      setFinishing(true);
      const bail = setTimeout(() => { try { ws.close(); } catch { /* noop */ } }, 60000);
      const finish = () => {
        clearTimeout(bail);
        try { ws.send(JSON.stringify({ type: "finalize" })); } catch { /* closed */ }
        setTimeout(() => { try { ws.close(); } catch { /* noop */ } }, 250);
      };
      if (m.inflight) {
        m.finishedWaiter = finish;
      } else {
        finish();
      }
    }

    if (!S.finishing) {
      $("micHint").textContent = "Tap the mic and start talking";
      setPhase("ready");
    }
    render();
  }

  function toggleMic() {
    if (S.mic.active) stopMic();
    else startMic();
  }

  /* ------------------------------------------------------- text model setup */

  const LLM = { config: null, models: [], maxThreads: 16 };

  async function loadLLMConfig() {
    try {
      const r = await fetch("/api/ai/models", { cache: "no-store" });
      const data = await r.json();
      LLM.models = data.models || [];
      LLM.config = data.config || {};
      LLM.maxThreads = data.max_threads || 16;
      renderLLM();
    } catch {
      $("llmStatus").textContent = "text model service unreachable";
    }
  }

  function renderLLM() {
    const cfg = LLM.config || {};
    const sel = $("llmModel");

    sel.textContent = "";
    (LLM.models.length ? LLM.models : [{ name: "no GGUF found in models/llm", path: "", mb: 0 }])
      .forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.path || "";
        opt.textContent = m.mb ? `${m.name} · ${m.mb >= 1000 ? (m.mb / 1000).toFixed(1) + " GB" : m.mb + " MB"}` : m.name;
        opt.selected = !!m.active;
        sel.appendChild(opt);
      });
    if (cfg.model && !(LLM.models || []).some((m) => m.active)) {
      // a custom absolute path that is not inside models/llm
      const opt = document.createElement("option");
      opt.value = cfg.model;
      opt.textContent = cfg.model;
      opt.selected = true;
      sel.appendChild(opt);
    }

    const active = (LLM.models || []).find((m) => m.active);
    $("llmModelSize").textContent = active ? `${active.mb} MB` : (cfg.model ? "custom path" : "—");

    $("llmPath").value = cfg.model || "";
    $("llmThreads").max = String(LLM.maxThreads);
    $("llmThreads").value = cfg.threads || LLM.maxThreads;
    $("llmThreadsVal").textContent = cfg.threads || LLM.maxThreads;
    $("llmCtx").value = cfg.ctx || 4096;
    $("llmCtxVal").textContent = cfg.ctx || 4096;
    $("llmTemp").value = cfg.temperature ?? 0.3;
    $("llmTempVal").textContent = Number(cfg.temperature ?? 0.3).toFixed(2);
    $("llmTokens").value = cfg.max_tokens || 700;
    $("llmTokensVal").textContent = cfg.max_tokens || 700;
    $("llmRep").value = cfg.repeat_penalty || 1.08;
    $("llmRepVal").textContent = Number(cfg.repeat_penalty || 1.08).toFixed(2);
  }

  async function applyLLMConfig() {
    const btn = $("llmApply");
    btn.disabled = true;
    $("llmStatus").textContent = "loading model…";
    const custom = $("llmPath").value.trim();
    const payload = {
      model: custom || $("llmModel").value,
      threads: Number($("llmThreads").value),
      ctx: Number($("llmCtx").value),
      temperature: Number($("llmTemp").value),
      max_tokens: Number($("llmTokens").value),
      repeat_penalty: Number($("llmRep").value),
      reload: true,
    };
    try {
      const r = await fetch("/api/ai/config", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await r.json();
      LLM.config = data.config || LLM.config;
      const reload = data.reload || {};
      if (reload.ok) {
        $("llmStatus").textContent = `ready · loaded in ${reload.load_s}s`;
        toast("Text model reloaded");
      } else {
        $("llmStatus").textContent = `failed: ${reload.error || "unknown"}`;
        toast("Model reload failed", true);
      }
      await loadLLMConfig();
      pollHealth();
    } catch (err) {
      $("llmStatus").textContent = `failed: ${err.message || err}`;
      toast("Model config failed", true);
    } finally {
      btn.disabled = false;
    }
  }

  /* ------------------------------------------------------------- ai actions */

  const AI_TITLES = {
    summarize: "Summary", minutes: "Minutes", actions: "Action items",
    rewrite: "Cleaned transcript", translate_en: "English",
    translate_de: "German", custom: "Answer",
  };

  const AI = { controller: null, running: false, buffer: "", action: null, lastStatus: null };

  function aiSetBusy(busy) {
    AI.running = busy;
    document.querySelectorAll(".aiBtn[data-ai]").forEach((b) => { b.disabled = busy; });
    $("aiCustomBtn").disabled = busy;
    $("aiStop").classList.toggle("hidden", !busy);
    if (!busy) $("aiStatus").textContent = AI.lastStatus || "";
  }

  function openAIPanel() {
    const panel = $("aiPanel");
    panel.classList.add("on");
    panel.setAttribute("aria-hidden", "false");
    document.querySelector(".shell").classList.add("ai-open");
    return panel;
  }

  function closeAIPanel() {
    const panel = $("aiPanel");
    panel.classList.remove("on");
    panel.setAttribute("aria-hidden", "true");
    document.querySelector(".shell").classList.remove("ai-open");
  }

  function aiPaint() {
    const body = $("aiBody");
    if (!body) return;
    // The model writes light markdown; render just bold so headings don't show
    // literal asterisks. Everything else stays pre-wrap text.
    const html = esc(AI.buffer).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    if (body.innerHTML !== html) body.innerHTML = html;
  }

  async function runAI(action, extra = "") {
    if (AI.running) return;
    const transcript = transcriptText();
    if (!transcript) { toast("Nothing to work on yet", true); return; }

    const panel = openAIPanel();
    const body = $("aiBody");
    body.classList.remove("err");
    body.textContent = "";
    AI.buffer = "";
    AI.action = action;
    $("aiTitle").textContent = AI_TITLES[action] || "Answer";
    $("aiStatus").textContent = "thinking…";
    aiSetBusy(true);

    AI.controller = new AbortController();
    const started = performance.now();
    let ticks = 0;

    try {
      const res = await fetch("/api/ai/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: AI.controller.signal,
        body: JSON.stringify({
          action,
          transcript,
          extra,
          speakers: Object.fromEntries(Object.entries(S.names).map(([k, v]) => [k, v])),
        }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        pending += decoder.decode(value, { stream: true });
        const lines = pending.split("\n");
        pending = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          let msg;
          try { msg = JSON.parse(line.slice(5).trim()); } catch { continue; }
          if (msg.delta) {
            AI.buffer += msg.delta;
            if (++ticks % 3 === 0) aiPaint();     // keep up without thrashing
          } else if (msg.error) {
            throw new Error(msg.error);
          } else if (msg.done) {
            AI.lastStatus = `${msg.elapsed}s · ${AI.buffer.trim().split(/\s+/).length} words`;
          }
        }
      }
      aiPaint();
      if (!AI.buffer.trim()) { body.textContent = "The model returned nothing."; body.classList.add("err"); }
    } catch (err) {
      if (err.name === "AbortError") {
        aiPaint();
        AI.lastStatus = "stopped";
      } else {
        body.classList.add("err");
        body.textContent = String(err.message || err);
        AI.lastStatus = "failed";
      }
    } finally {
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      $("aiStatus").textContent = AI.lastStatus || `${secs}s`;
      const meta = $("aiMeta");
      if (meta) meta.textContent = AI.lastStatus || `${secs}s`;
      aiSetBusy(false);
      AI.controller = null;
    }
  }

  function aiResult() {
    const text = ($("aiBody").textContent || "").trim();
    if (!text) { toast("Nothing to share yet", true); return null; }
    const action = AI.action || "summary";
    const title = AI_TITLES[action] || "Result";
    const docTitle = $("docTitle").textContent || "transcript";
    const stamp = new Date();
    return { text, action, title, docTitle, stamp };
  }

  function aiMarkdown(r) {
    return [
      `# ${r.title} — ${r.docTitle}`,
      "",
      `_MoScribe · kmail · ${r.stamp.toLocaleString()} · ${AI.lastStatus || ""}_`,
      "",
      r.text,
      "",
    ].join("\n");
  }

  function downloadBlob(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1500);
  }

  async function shareAI(kind) {
    const r = aiResult();
    if (!r) return;
    const slug = `${r.action}-${r.stamp.toISOString().slice(0, 10)}`;

    if (kind === "copy" || kind === "copy-md") {
      const body = kind === "copy" ? r.text : aiMarkdown(r);
      try {
        await navigator.clipboard.writeText(body);
        toast(kind === "copy" ? "Copied" : "Copied as Markdown");
      } catch {
        const ta = document.createElement("textarea");
        ta.value = body;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
        toast("Copied");
      }
      return;
    }

    if (kind === "txt") {
      downloadBlob(new Blob([r.text + "\n"], { type: "text/plain;charset=utf-8" }), `moscribe-${slug}.txt`);
      toast("Saved .txt");
      return;
    }

    if (kind === "md") {
      downloadBlob(new Blob([aiMarkdown(r)], { type: "text/markdown;charset=utf-8" }), `moscribe-${slug}.md`);
      toast("Saved .md");
      return;
    }

    if (kind === "json") {
      const payload = {
        app: "MoScribe", brand: "kmail", action: r.action, title: r.title,
        document: r.docTitle, created: r.stamp.toISOString(),
        result: r.text,
        transcript: transcriptText(),
        speakers: S.names,
      };
      downloadBlob(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
                   `moscribe-${slug}.json`);
      toast("Saved .json");
      return;
    }

    if (kind === "mail") {
      const subject = encodeURIComponent(`${r.title} — ${r.docTitle} (MoScribe)`);
      const body = encodeURIComponent(r.text.slice(0, 1800));
      location.href = `mailto:?subject=${subject}&body=${body}`;
      return;
    }

    if (kind === "native") {
      try {
        await navigator.share({ title: `${r.title} — ${r.docTitle}`, text: r.text });
      } catch { /* user cancelled */ }
    }
  }

  function stopAI() {
    if (AI.controller) AI.controller.abort();
  }

  function wireAI() {
    document.querySelectorAll(".aiBtn[data-ai]").forEach((b) => {
      b.onclick = () => runAI(b.dataset.ai);
    });
    $("aiCustomBtn").onclick = () => {
      const input = $("aiCustomInput");
      if (input.classList.contains("hidden")) {
        input.classList.remove("hidden");
        input.focus();
        return;
      }
      const q = input.value.trim();
      if (!q) { input.classList.add("hidden"); return; }
      runAI("custom", q);
    };
    $("aiCustomInput").onkeydown = (e) => {
      if (e.key === "Enter") { e.preventDefault(); $("aiCustomBtn").click(); }
      if (e.key === "Escape") { e.target.classList.add("hidden"); }
    };
    $("aiStop").onclick = stopAI;
    $("aiClose").onclick = () => { stopAI(); closeAIPanel(); };
    $("aiShare").onclick = (e) => {
      e.stopPropagation();
      const menu = $("aiShareMenu");
      if (menu.hidden) {
        const r = e.currentTarget.getBoundingClientRect();
        menu.hidden = false;
        menu.style.left = "auto";
        menu.style.right = `${Math.max(10, innerWidth - r.right)}px`;
        menu.style.top = `${Math.min(innerHeight - 260, r.bottom + 8)}px`;
      } else {
        menu.hidden = true;
      }
    };
    if (!navigator.share) $("aiShareNative").hidden = true;
    document.querySelectorAll("#aiShareMenu button").forEach((b) => {
      b.onclick = () => { $("aiShareMenu").hidden = true; shareAI(b.dataset.share); };
    });
  }

  /* ------------------------------------------------------------ history */

  async function loadHistory() {
    try {
      const r = await fetch("/api/history", { cache: "no-store" });
      const { items } = await r.json();
      const list = $("historyList");
      list.innerHTML = "";
      if (!items || !items.length) {
        list.innerHTML = '<p class="fine">Transcriptions you make appear here.</p>';
        return;
      }
      items.slice(0, 40).forEach((it) => {
        const el = document.createElement("div");
        el.className = "hitem";
        const mins = it.duration ? Math.floor(it.duration / 60) : 0;
        const secs = it.duration ? Math.round(it.duration % 60) : 0;
        el.innerHTML = `<button class="hDel" title="Delete">✕</button>
          <div class="hTitle"></div>
          <div class="hMeta">${it.mode === "live" ? "live" : "file"} · ${mins}:${String(secs).padStart(2, "0")} · ${it.words || 0} w${it.speakers ? ` · ${it.speakers} spk` : ""}${it.rtf ? ` · RTF ${it.rtf}` : ""}</div>`;
        el.querySelector(".hTitle").textContent = it.title || "transcript";
        el.onclick = () => {
          resetSession();
          S.committed = it.text || "";
          S.segments = it.segments || [];
          S.turns = it.turns && it.turns.length ? it.turns : buildTurns(S.segments);
          if (!settings.speakerSep || settings.speakerSep === "off") {
            S.turns = [];
            S.committed = it.text || "";
          }
          S.duration = it.duration || 0;
          S.rtf = it.rtf || null;
          $("docTitle").textContent = it.title || "transcript";
          $("docMeta").textContent = `${it.mode === "live" ? "live dictation" : "uploaded file"} · ${new Date((it.created || 0) * 1000).toLocaleString()}`;
          render();
          S.follow = false;
          $("transcript").scrollTop = 0;
          closeDrawers();
          toast("Loaded from history");
        };
        el.querySelector(".hDel").onclick = async (ev) => {
          ev.stopPropagation();
          await fetch(`/api/history/${it.id}`, { method: "DELETE" });
          loadHistory();
        };
        list.appendChild(el);
      });
    } catch { /* history is non-critical */ }
  }


  /* ------------------------------------------------------------ vitals */

  const VITALS = { timer: null, prev: { cpu: 0, ram: 0 }, shown: { cpu: 0, ram: 0 } };
  let vAnim = null;

  function sparkPath(hist, w = 100, h = 26) {
    if (!hist || hist.length < 2) return { line: "", fill: "" };
    const max = 100;
    const step = w / (hist.length - 1);
    const pts = hist.map((v, i) => [i * step, h - (Math.max(0, Math.min(max, v)) / max) * (h - 3) - 1.5]);
    let d = `M ${pts[0][0].toFixed(2)} ${pts[0][1].toFixed(2)}`;
    for (let i = 1; i < pts.length; i++) {
      const [x, y] = pts[i];
      const [px, py] = pts[i - 1];
      const cx = (px + x) / 2;
      d += ` C ${cx.toFixed(2)} ${py.toFixed(2)} ${cx.toFixed(2)} ${y.toFixed(2)} ${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    return { line: d, fill: `${d} L ${w} ${h} L 0 ${h} Z` };
  }

  function drawSpark(id, hist) {
    const svg = document.getElementById(id);
    if (!svg) return;
    const { line, fill } = sparkPath(hist);
    svg.querySelector(".sparkLine").setAttribute("d", line);
    svg.querySelector(".sparkFill").setAttribute("d", fill);
  }

  // numbers roll instead of snapping, so the panel feels alive
  function rollTo(kind, target) {
    const el = document.getElementById(kind === "cpu" ? "vCpu" : "vRam");
    if (!el) return;
    const from = VITALS.shown[kind] || 0;
    VITALS.shown[kind] = target;
    el.textContent = `${target.toFixed(kind === "cpu" ? 0 : 1)}%`;
    // background tabs starve rAF, so the value is already correct before animating
    if (document.visibilityState !== "visible") return;
    const t0 = performance.now();
    const dur = 420;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - k, 3);
      const v = from + (target - from) * eased;
      el.textContent = `${v.toFixed(kind === "cpu" ? 0 : 1)}%`;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function paintVitals(d) {
    if (!d || typeof d.cpu !== "number") return;
    drawSpark("sparkCpu", d.cpu_hist || []);
    drawSpark("sparkRam", d.ram_hist || []);
    rollTo("cpu", d.cpu);
    rollTo("ram", d.ram);
    [["cpu", d.cpu], ["ram", d.ram]].forEach(([kind, v]) => {
      const row = document.querySelector(`.vRow[data-kind="${kind}"]`);
      if (!row) return;
      row.classList.toggle("warn", v >= 80 && v < 92);
      row.classList.toggle("crit", v >= 92);
    });
    const used = d.used / 1073741824;
    const total = d.total / 1073741824;
    const sub = $("vSub");
    if (sub) {
      sub.textContent = d.total
        ? `${used.toFixed(1)} / ${total.toFixed(0)} GB · load ${d.load.toFixed(2)} · ${d.cores} cores`
        : "—";
    }
  }

  async function pollVitals() {
    try {
      const r = await fetch("/api/system", { cache: "no-store" });
      if (r.ok) paintVitals(await r.json());
    } catch { /* cosmetic only */ }
  }

  function startVitals() {
    if (VITALS.timer) return;
    pollVitals();
    VITALS.timer = setInterval(pollVitals, 2000);
  }

  /* ------------------------------------------------------------ export */

  function download(fmt) {
    const text = transcriptText();
    if (!text) { toast("Nothing to export yet", true); return; }
    const base = `${(S.file && S.file.name ? S.file.name.replace(/\.[^.]+$/, "") : "vibescribe")}-${Date.now()}`;
    let blob, name;

    if (fmt === "txt") {
      blob = new Blob([text + "\n"], { type: "text/plain;charset=utf-8" });
      name = `${base}.txt`;
    } else if (fmt === "md") {
      const body = S.turns.length
        ? `# ${$("docTitle").textContent}\n\n` +
          S.turns.map((t) => `**${speakerLabel(t.speaker)}** · ${fmtStamp(t.start)}\n\n${t.text}\n`).join("\n")
        : `# ${$("docTitle").textContent}\n\n${text}\n`;
      blob = new Blob([body], { type: "text/markdown;charset=utf-8" });
      name = `${base}.md`;
    } else if (fmt === "srt") {
      const segs = (S.turns.length ? S.turns : S.segments)
        .filter((s) => s.text && s.start != null);
      const body = segs.length
        ? segs.map((s, i) =>
            `${i + 1}\n${srtTime(s.start)} --> ${srtTime(s.end != null ? s.end : s.start + 4)}\n` +
            `${s.speaker != null && s.speaker >= 0 ? speakerLabel(s.speaker) + ": " : ""}${s.text}\n`).join("\n")
        : `1\n00:00:00,000 --> ${srtTime(S.duration || 0)}\n${text}\n`;
      blob = new Blob([body], { type: "application/x-subrip;charset=utf-8" });
      name = `${base}.srt`;
    } else {
      blob = new Blob([JSON.stringify({
        text, duration: S.duration, rtf: S.rtf, mode: S.mode,
        speakers: S.turns.length ? [...new Set(S.turns.map((t) => t.speaker))].filter((s) => s >= 0) : [],
        names: S.names, turns: S.turns, segments: S.segments,
        created: new Date().toISOString(),
      }, null, 2)], { type: "application/json" });
      name = `${base}.json`;
    }

    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1500);
    toast(`Saved ${name}`);
  }

  function srtTime(sec) {
    sec = Math.max(0, Number(sec) || 0);
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.round((sec - Math.floor(sec)) * 1000);
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
  }

  async function copyTranscript() {
    const text = transcriptText();
    if (!text) { toast("Nothing to copy yet", true); return; }
    try {
      await navigator.clipboard.writeText(text);
      toast("Copied to clipboard");
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      toast("Copied to clipboard");
    }
  }

  /* ------------------------------------------------------------ drawers */

  function openHelp() {
    const el = $("helpScreen");
    el.classList.add("on");
    el.setAttribute("aria-hidden", "false");
    el.scrollTop = 0;
  }

  function closeHelp() {
    const el = $("helpScreen");
    el.classList.remove("on");
    el.setAttribute("aria-hidden", "true");
  }

  function setSettingsTab(name) {
    const tabs = document.querySelectorAll(".drawerTab");
    if (!tabs.length) return;
    const pane = document.querySelector(`.tabPane[data-pane="${name}"]`);
    if (!pane) name = "prompt";
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", String(on));
    });
    document.querySelectorAll(".tabPane").forEach((p) => {
      p.classList.toggle("hidden", p.dataset.pane !== name);
    });
    try { localStorage.setItem("moscribe.settingsTab", name); } catch { /* ignore */ }
  }

  function openDrawer(id) {
    closeDrawers();
    $(id).classList.add("on");
    $(id).setAttribute("aria-hidden", "false");
    $("scrim").classList.add("on");
  }
  function closeDrawers() {
    ["settingsDrawer", "historyDrawer"].forEach((id) => {
      $(id).classList.remove("on");
      $(id).setAttribute("aria-hidden", "true");
    });
    $("scrim").classList.remove("on");
    $("exportMenu").hidden = true;
    $("aiShareMenu").hidden = true;
  }

  /* ------------------------------------------------------------- wiring */

  const presetById = (id) => PRESETS.find((p) => p.id === id) || null;

  function detectPreset(text) {
    const t = (text || "").trim();
    if (!t) return null;
    const hit = PRESETS.find((p) => p.text.trim() === t);
    return hit ? hit.id : "custom";
  }

  const LAYOUT_LABELS = {
    auto: "Auto — match the template",
    bubbles: "Bubbles — conversation",
    document: "Document — prose",
    cards: "Cards — one record per speaker",
    compact: "Compact — dense log",
    qa: "Q&A — question and answer",
  };

  function renderLayoutSelect() {
    const sel = $("layoutSel");
    if (!sel) return;
    if (sel.options.length !== LAYOUTS.length) {
      sel.textContent = "";
      LAYOUTS.forEach((id) => {
        const o = document.createElement("option");
        o.value = id;
        o.textContent = LAYOUT_LABELS[id] || id;
        sel.appendChild(o);
      });
    }
    sel.value = settings.layout || "auto";
  }

  function renderPresets() {
    const row = $("presetRow");
    if (!row) return;
    if (row.childElementCount !== PRESETS.length + 1) {
      row.textContent = "";
      PRESETS.forEach((preset) => {
        const b = document.createElement("button");
        b.className = "preset";
        b.dataset.preset = preset.id;
        b.textContent = preset.label;
        b.title = preset.text.slice(0, 160);
        b.onclick = () => applyPreset(preset);
        row.appendChild(b);
      });
      const clear = document.createElement("button");
      clear.className = "preset";
      clear.dataset.preset = "";
      clear.textContent = "Clear";
      clear.onclick = () => applyPreset(null);
      row.appendChild(clear);
    }
    row.querySelectorAll(".preset").forEach((b) => {
      b.classList.toggle("active", !!b.dataset.preset && b.dataset.preset === settings.activePreset);
    });
  }

  function renderPromptTag() {
    const tag = $("promptTag");
    const pill = $("pillPrompt");
    const preset = presetById(settings.activePreset);
    const hasText = !!(settings.context || "").trim();

    let label = "none";
    let state = "none";
    if (preset) { label = preset.label; state = "set"; }
    else if (hasText) { label = "Custom"; state = "custom"; }

    if (tag) { tag.textContent = label; tag.dataset.state = state; }
    if (pill) {
      pill.dataset.state = state === "none" ? "idle" : "ok";
      pill.querySelector("em").textContent = label;
      pill.title = hasText ? settings.context.slice(0, 180) : "No prompt applied";
    }
  }

  function applyPreset(preset) {
    if (!preset) {
      settings.context = "";
      settings.activePreset = null;
    } else {
      settings.context = preset.text;
      settings.activePreset = preset.id;
    }
    $("context").value = settings.context;
    saveSettings();
    renderPresets();
    renderPromptTag();
    pushLiveConfig();
    toast(preset ? `Prompt applied: ${preset.label}` : "Prompt cleared");
  }

  // An already-open dictation socket keeps its own copy of the config; push the
  // new prompt into it so the rest of the session benefits immediately.
  function pushLiveConfig() {
    const m = S.mic;
    if (m.active && m.ws && m.ws.readyState === WebSocket.OPEN) {
      try { m.ws.send(JSON.stringify({ type: "config", ...settingsPayload() })); } catch { /* socket closing */ }
    }
  }

  // Explicit commit for hand-written context — same path the presets take.
  function commitPrompt() {
    settings.context = $("context").value.trim();
    settings.activePreset = detectPreset(settings.context);
    saveSettings();
    renderPresets();
    renderPromptTag();
    pushLiveConfig();
    toast(settings.context
      ? "Prompt applied — used for the next transcription"
      : "Prompt cleared");
  }

  function syncSettingsUI() {
    $("threads").max = String(Math.max(2, Math.min(16, navigator.hardwareConcurrency || 8)));
    $("threads").value = settings.threads;
    $("threadsVal").textContent = settings.threads;
    $("chunk").value = settings.chunkSec;
    $("chunkVal").textContent = `${settings.chunkSec} s`;
    $("overlap").value = settings.overlapSec;
    $("overlapVal").textContent = `${Number(settings.overlapSec)} s`;
    $("temperature").value = settings.temperature;
    $("tempVal").textContent = Number(settings.temperature).toFixed(2);
    $("topP").value = settings.topP;
    $("topPVal").textContent = Number(settings.topP).toFixed(2);
    $("context").value = settings.context || "";
    renderPresets();
    renderLayoutSelect();
    renderPromptTag();
    $("sens").value = settings.sensitivity;
    $("maxSpk").value = settings.maxSpeakers;
    $("maxSpkVal").textContent = settings.maxSpeakers;
    $("sensVal").textContent = sensLabel(settings.sensitivity);
    $("spkToggle").setAttribute("aria-checked", String(settings.speakerSep === "on"));
    $("spkFields").classList.toggle("hidden", settings.speakerSep !== "on");

    document.querySelectorAll(".seg2[data-decoding]").forEach((b) => {
      b.classList.toggle("active", b.dataset.decoding === settings.decoding);
    });
    document.querySelectorAll(".seg2[data-pformat]").forEach((b) => {
      b.classList.toggle("active", b.dataset.pformat === settings.promptFormat);
    });
  }

  function sensLabel(v) {
    if (v <= 0.70) return "merges similar voices";
    if (v <= 0.76) return "lenient";
    if (v <= 0.82) return "balanced";
    if (v <= 0.87) return "strict";
    return "splits similar voices";
  }

  function wire() {
    document.querySelectorAll(".seg[data-mode]").forEach((b) => {
      b.onclick = () => setMode(b.dataset.mode);
    });

    $("micBtn").onclick = toggleMic;

    // file
    $("fileInput").onchange = (e) => setFile(e.target.files[0]);
    $("clearFile").onclick = (e) => { e.preventDefault(); setFile(null); $("fileInput").value = ""; };
    $("transcribeBtn").onclick = startJob;

    const dz = $("dropZone");
    ["dragenter", "dragover"].forEach((evt) => dz.addEventListener(evt, (e) => {
      e.preventDefault(); dz.classList.add("over");
    }));
    ["dragleave", "drop"].forEach((evt) => dz.addEventListener(evt, (e) => {
      e.preventDefault(); dz.classList.remove("over");
    }));
    dz.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) { setFile(f); setMode("file"); }
    });

    // speakers
    $("spkToggle").onclick = () => {
      settings.speakerSep = settings.speakerSep === "on" ? "off" : "on";
      $("spkToggle").setAttribute("aria-checked", String(settings.speakerSep === "on"));
      $("spkFields").classList.toggle("hidden", settings.speakerSep !== "on");
      saveSettings();
      if (settings.speakerSep === "off") {
        S.turns = [];
        render();
      } else if (S.segments.length) {
        S.turns = buildTurns(S.segments);
        render();
      }
    };
    $("sens").oninput = (e) => {
      settings.sensitivity = Number(e.target.value);
      $("sensVal").textContent = sensLabel(settings.sensitivity);
      saveSettings();
    };
    $("maxSpk").oninput = (e) => {
      settings.maxSpeakers = Number(e.target.value);
      $("maxSpkVal").textContent = settings.maxSpeakers;
      saveSettings();
    };

    // settings drawer
    document.querySelectorAll(".drawerTab").forEach((t) => {
      t.onclick = () => setSettingsTab(t.dataset.tab);
    });
    $("openHelp").onclick = () => openHelp();
    $("closeHelp").onclick = () => closeHelp();
    $("openSettings").onclick = () => {
      openDrawer("settingsDrawer");
      let last = "prompt";
      try { last = localStorage.getItem("moscribe.settingsTab") || "prompt"; } catch { /* ignore */ }
      setSettingsTab(last);
      loadLLMConfig();
    };
    $("llmApply").onclick = applyLLMConfig;
    ["llmThreads", "llmCtx", "llmTemp", "llmTokens", "llmRep"].forEach((id) => {
      $(id).oninput = (e) => {
        const out = { llmThreads: ["llmThreadsVal", (v) => v],
                      llmCtx: ["llmCtxVal", (v) => v],
                      llmTemp: ["llmTempVal", (v) => Number(v).toFixed(2)],
                      llmTokens: ["llmTokensVal", (v) => v],
                      llmRep: ["llmRepVal", (v) => Number(v).toFixed(2)] }[id];
        $(out[0]).textContent = out[1](e.target.value);
      };
    });
    $("llmModel").onchange = () => { $("llmPath").value = ""; };
    $("openHistory").onclick = () => { openDrawer("historyDrawer"); loadHistory(); };
    $("refreshHistory").onclick = loadHistory;
    $("scrim").onclick = closeDrawers;
    document.querySelectorAll("[data-close]").forEach((b) => { b.onclick = closeDrawers; });

    $("threads").oninput = (e) => {
      settings.threads = Number(e.target.value);
      $("threadsVal").textContent = settings.threads;
      saveSettings();
    };
    $("chunk").oninput = (e) => {
      settings.chunkSec = Number(e.target.value);
      $("chunkVal").textContent = `${settings.chunkSec} s`;
      saveSettings();
    };
    $("overlap").oninput = (e) => {
      settings.overlapSec = Number(e.target.value);
      $("overlapVal").textContent = `${settings.overlapSec} s`;
      saveSettings();
    };
    $("temperature").oninput = (e) => {
      settings.temperature = Number(e.target.value);
      $("tempVal").textContent = settings.temperature.toFixed(2);
      saveSettings();
    };
    $("topP").oninput = (e) => {
      settings.topP = Number(e.target.value);
      $("topPVal").textContent = settings.topP.toFixed(2);
      saveSettings();
    };
    $("context").oninput = (e) => {
      settings.context = e.target.value;
      settings.activePreset = detectPreset(settings.context);
      saveSettings();
      renderPresets();
      renderPromptTag();
    };
    $("layoutSel").onchange = (e) => {
      settings.layout = e.target.value;
      saveSettings();
      syncTranscript();          // immediate: background tabs throttle coalesced paints
      toast(e.target.value === "auto"
        ? `Layout follows the template (${activeLayout()})`
        : `Layout: ${LAYOUT_LABELS[e.target.value]}`);
    };
    $("applyPrompt").onclick = commitPrompt;
    document.querySelectorAll(".seg2[data-decoding]").forEach((b) => {
      b.onclick = () => {
        settings.decoding = b.dataset.decoding;
        syncSettingsUI();
        saveSettings();
      };
    });
    document.querySelectorAll(".seg2[data-pformat]").forEach((b) => {
      b.onclick = () => {
        settings.promptFormat = b.dataset.pformat;
        syncSettingsUI();
        saveSettings();
      };
    });

    // actions
    $("copyBtn").onclick = copyTranscript;
    $("exportBtn").onclick = () => download("txt");
    $("exportMenuBtn").onclick = (e) => {
      e.stopPropagation();
      const menu = $("exportMenu");
      const r = e.currentTarget.getBoundingClientRect();
      menu.hidden = !menu.hidden;
      menu.style.right = "auto";
      menu.style.left = `${Math.max(10, r.left - 120)}px`;
      menu.style.top = `${Math.min(window.innerHeight - 190, r.bottom + 6)}px`;
    };
    document.querySelectorAll("#exportMenu button").forEach((b) => {
      b.onclick = () => { $("exportMenu").hidden = true; download(b.dataset.export); };
    });
    document.addEventListener("click", (e) => {
      if (!$("exportMenu").hidden && !$("exportMenu").contains(e.target)) $("exportMenu").hidden = true;
      if (!$("aiShareMenu").hidden && !$("aiShareMenu").contains(e.target)
          && !$("aiShare").contains(e.target)) $("aiShareMenu").hidden = true;
    });

    $("clearBtn").onclick = () => {
      if (S.mic.active) stopMic();
      resetSession();
      toast("New session");
    };

    $("search").oninput = () => render();
    $("transcript").addEventListener("scroll", (e) => {
      const el = e.currentTarget;
      S.follow = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    });

    document.addEventListener("keydown", (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      const typing = tag === "input" || tag === "textarea" || e.target.isContentEditable;
      if (typing) return;

      if (e.code === "Space") { e.preventDefault(); toggleMic(); }
      else if (e.key === "Escape") {
        if ($("helpScreen").classList.contains("on")) closeHelp();
        else if (S.mic.active) stopMic();
        else closeDrawers();
      }
      else if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "c") {
        e.preventDefault(); copyTranscript();
      }
    });

    window.addEventListener("beforeunload", (e) => {
      if (S.busy || S.mic.active) { e.preventDefault(); e.returnValue = ""; }
    });
  }

  /* --------------------------------------------------------------- boot */

  let booted = false;
  function boot() {
    if (booted) return;
    booted = true;
    wire();
    wireAI();
    syncSettingsUI();
    setMode("live");
    render();
    pollHealth();
    setInterval(pollHealth, 4000);
    loadLLMConfig();
    loadHistory();
    startVitals();

    // Preload models so the first window isn't a cold start.
    fetch("/api/engine/warmup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...settingsPayload(), chunk_sec: settings.chunkSec }),
    }).catch(() => {});
  }

  window.__vibescribe = { S, settings, PRESETS, LAYOUTS, activeLayout, applyPreset, commitPrompt, shareAI, aiMarkdown,
                        settingsPayload, pushLiveConfig, render, setMode, startMic, stopMic, setFinishing };
  document.addEventListener("DOMContentLoaded", boot);
  if (document.readyState === "interactive" || document.readyState === "complete") boot();
})();
