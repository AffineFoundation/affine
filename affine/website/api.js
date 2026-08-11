/** Data client: prefer /api/v1 (affine-dash), fall back to Hippius data/*.json. */

const API = "/api/v1";
const POLL_MS = 15000;

// Asset version from our own module URL (api.js?v=N) — appended to duel
// fetches so a deploy busts long-lived browser caches of old payloads.
const SITE_V = new URL(import.meta.url).searchParams.get("v") || "0";

/** "api" | "static" | null (undetected) */
let mode = null;
let detectPromise = null;

async function getJSON(url, { signal } = {}) {
  try {
    const r = await fetch(url, {
      signal,
      headers: { Accept: "application/json" },
    });
    if (r.status === 304) return { notModified: true };
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export async function detectMode({ signal } = {}) {
  if (mode) return mode;
  if (detectPromise) return detectPromise;
  detectPromise = (async () => {
    const health = await getJSON(`${API}/health`, { signal });
    mode = health && health.ok ? "api" : "static";
    return mode;
  })();
  try {
    return await detectPromise;
  } finally {
    detectPromise = null;
  }
}

export function currentMode() {
  return mode;
}

export async function fetchSnapshot(signal) {
  const m = await detectMode({ signal });
  if (m === "api") return getJSON(`${API}/snapshot`, { signal });
  return getJSON("data/dashboard.json", { signal });
}

export async function fetchHistory({ limit = 100, q = "", event = "" } = {}, signal) {
  const m = await detectMode({ signal });
  if (m === "api") {
    const params = new URLSearchParams({ limit: String(limit) });
    if (q) params.set("q", q);
    if (event) params.set("event", event);
    const data = await getJSON(`${API}/history?${params}`, { signal });
    if (!data) return [];
    return data.items || [];
  }
  let items = await getJSON("data/history.json", { signal });
  if (!Array.isArray(items)) return [];
  if (event) items = items.filter((r) => r.event === event);
  if (q) {
    const needle = q.toLowerCase();
    items = items.filter((r) =>
      [r.event, r.repo, r.hotkey, r.error_code, r.rejection_reason, r.challenge_id]
        .join(" ").toLowerCase().includes(needle));
  }
  return items.slice(0, limit);
}

export async function fetchBenchmarks(signal) {
  const m = await detectMode({ signal });
  if (m === "api") return getJSON(`${API}/benchmarks`, { signal });
  return getJSON("data/benchmarks.json", { signal });
}

export async function fetchDuel(challengeId, signal) {
  const m = await detectMode({ signal });
  if (m === "api") {
    return getJSON(
      `${API}/duels/${encodeURIComponent(challengeId)}?v=${SITE_V}`, { signal });
  }
  // Static archive: best-effort from slim history.json (no full artifact).
  const items = await fetchHistory({ limit: 200 }, signal);
  const row = items.find((r) => r.challenge_id === challengeId);
  return row ? { ...row, has_artifact: false, has_series: false } : null;
}

export async function fetchDuelSeries(challengeId, signal) {
  const m = await detectMode({ signal });
  if (m !== "api") return null;
  return getJSON(`${API}/duels/${encodeURIComponent(challengeId)}/series`, { signal });
}

/** Validator log lines mentioning this duel. API mode only. */
export async function fetchDuelLog(challengeId, signal) {
  const m = await detectMode({ signal });
  if (m !== "api") return null;
  return getJSON(`${API}/duels/${encodeURIComponent(challengeId)}/log`, { signal });
}

export function duelTurnUrl(challengeId, turnId) {
  return `${API}/duels/${encodeURIComponent(challengeId)}/turn`
    + `?turn_id=${encodeURIComponent(turnId)}`;
}

/** Full rollout detail for one turn of a duel. API mode only. */
export async function fetchDuelTurn(challengeId, turnId, signal) {
  const m = await detectMode({ signal });
  if (m !== "api") return null;
  return getJSON(duelTurnUrl(challengeId, turnId), { signal });
}

/* ---------- dataset browser (corpus D) ---------- */

/** Corpus stats: epoch, counts, mix, length histogram. API mode only. */
export async function fetchDataset(signal) {
  const m = await detectMode({ signal });
  if (m !== "api") return null;
  return getJSON(`${API}/dataset?v=${SITE_V}`, { signal });
}

/** One page of turn index rows with filters. API mode only. */
export async function fetchDatasetTurns(
  { limit = 50, cursor = 0, source = "", language = "", phase = "",
    repo = "", q = "" } = {}, signal) {
  const m = await detectMode({ signal });
  if (m !== "api") return null;
  const params = new URLSearchParams({
    limit: String(limit), cursor: String(cursor),
  });
  if (source) params.set("source", source);
  if (language) params.set("language", language);
  if (phase) params.set("phase", phase);
  if (repo) params.set("repo", repo);
  if (q) params.set("q", q);
  return getJSON(`${API}/dataset/turns?${params}`, { signal });
}

export function datasetTurnUrl(turnId) {
  return `${API}/dataset/turn?turn_id=${encodeURIComponent(turnId)}`;
}

/** Full turn content (prompt prefix + reference action). API mode only. */
export async function fetchDatasetTurn(turnId, signal) {
  const m = await detectMode({ signal });
  if (m !== "api") return null;
  return getJSON(datasetTurnUrl(turnId), { signal });
}

/** Historical SN registration burn (τ) from TMC, downsampled by affine-dash. */
export async function fetchRegHistory(signal) {
  const m = await detectMode({ signal });
  if (m === "api") return getJSON(`${API}/market/reg-history`, { signal });
  return getJSON("data/reg_history.json", { signal });
}

/**
 * Live snapshot updates. SSE on dash-api; poll data/dashboard.json on Hippius.
 */
export function watchSnapshot(onSnapshot, { onStatus } = {}) {
  let closed = false;
  let es = null;
  let pollTimer = null;
  let backoff = 1000;

  const setStatus = (s) => onStatus && onStatus(s);

  const stopPoll = () => {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  };

  const startPoll = (label = "poll") => {
    if (pollTimer || closed) return;
    setStatus(label);
    const tick = async () => {
      if (closed) return;
      try {
        const snap = await fetchSnapshot();
        if (snap && !snap.notModified) onSnapshot(snap);
        backoff = 1000;
      } catch {
        backoff = Math.min(backoff * 2, 15000);
      }
      if (!closed) pollTimer = setTimeout(tick, POLL_MS);
    };
    pollTimer = setTimeout(tick, 0);
  };

  const startSSE = () => {
    if (closed || typeof EventSource === "undefined") {
      startPoll();
      return;
    }
    try {
      es = new EventSource(`${API}/stream`);
    } catch {
      startPoll();
      return;
    }
    setStatus("sse");
    es.addEventListener("snapshot", (ev) => {
      try {
        onSnapshot(JSON.parse(ev.data));
        backoff = 1000;
      } catch { /* ignore */ }
    });
    es.onerror = () => {
      if (closed) return;
      try { es.close(); } catch { /* ignore */ }
      es = null;
      setStatus("sse-retry");
      stopPoll();
      setTimeout(() => {
        if (closed) return;
        startPoll();
        setTimeout(() => {
          if (closed) return;
          stopPoll();
          startSSE();
        }, Math.min(backoff, 10000));
        backoff = Math.min(backoff * 2, 30000);
      }, backoff);
    };
  };

  detectMode().then((m) => {
    if (closed) return;
    if (m === "api") startSSE();
    else startPoll("static");
  });

  return () => {
    closed = true;
    stopPoll();
    if (es) {
      try { es.close(); } catch { /* ignore */ }
      es = null;
    }
  };
}

export function fingerprint(obj) {
  try {
    return JSON.stringify(obj);
  } catch {
    return String(Math.random());
  }
}
