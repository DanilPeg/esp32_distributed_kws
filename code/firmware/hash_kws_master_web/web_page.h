// Single-page dashboard served by hash_kws_master_web.ino.
// Inlined here as a PROGMEM-friendly raw literal so the master ESP32
// is fully self-contained — no SPIFFS, no SD, no host required.
//
// JavaScript is plain vanilla (no frameworks); WebSocket auto-reconnects.

#pragma once

#include <Arduino.h>

static const char kHashKwsDashboardHtml[] PROGMEM = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hash KWS Distributed Master</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0d0f12;
    --panel: #161a20;
    --panel-2: #1e242c;
    --line: #283038;
    --text: #e6eef7;
    --muted: #8a96a4;
  }
  html,body { background: var(--bg); color: var(--text); }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0; padding: 18px; max-width: 980px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: 0.3px; }
  .sub { color: var(--muted); font-size: 12px; margin-bottom: 16px; }
  .stats { display: flex; flex-wrap: wrap; gap: 14px; padding: 10px 14px; background: var(--panel); border-radius: 10px; margin-bottom: 16px; font-size: 13px; }
  .stats span { white-space: nowrap; }
  .stats b { color: var(--text); }
  .conn-on  { color: #41d287; }
  .conn-off { color: #f06363; }
  .nodes { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }
  .node { background: var(--panel); border-radius: 10px; padding: 14px; border-left: 4px solid var(--line); transition: border-color .25s; }
  .node.online { border-left-color: #41d287; }
  .node.stale  { border-left-color: #ffae3a; }
  .node-title { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; }
  .node-label { font-size: 30px; font-weight: 600; margin-top: 6px; min-height: 36px; }
  .node-meta  { font-size: 11px; color: var(--muted); margin-top: 4px; }
  .panel-title { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .8px; margin: 0 0 8px; }
  .fusion-list { background: var(--panel); border-radius: 10px; padding: 8px 0; max-height: 60vh; overflow-y: auto; }
  .fusion { display: flex; gap: 12px; padding: 8px 14px; border-bottom: 1px solid var(--line); align-items: baseline; }
  .fusion:last-child { border-bottom: none; }
  .fusion-label { font-size: 18px; font-weight: 600; min-width: 80px; }
  .fusion-meta { font-size: 11px; color: var(--muted); }
  .label-yes    { color: #41d287; }
  .label-no     { color: #f06363; }
  .label-up     { color: #6ec3ff; }
  .label-down   { color: #ffae3a; }
  .label-left   { color: #c5b3ff; }
  .label-right  { color: #ff85c8; }
  .label-on     { color: #b6ff63; }
  .label-off    { color: #888; }
  .label-stop   { color: #ff7373; }
  .label-go     { color: #4ad6ff; }
  .label-unknown, .label-silence { color: #5a6470; }
  @media (max-width: 700px) { .nodes { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Hash KWS Distributed Master</h1>
<div class="sub">3 hashed students &mdash; ensemble aggregation on this ESP32</div>
<div class="stats">
  <span id="conn" class="conn-off">connecting&hellip;</span>
  <span>fusion total: <b id="ft">0</b></span>
  <span>packets: <b id="pk">0</b></span>
  <span>rejected: <b id="rj">0</b></span>
  <span>aggregator: <b id="agg">mean_logits</b></span>
  <span>uptime: <b id="up">0</b>s</span>
</div>
<div class="nodes" id="nodes">
  <div class="node" id="n1"><div class="node-title">node 1 &mdash; ens_a</div><div class="node-label">&mdash;</div><div class="node-meta">no packets yet</div></div>
  <div class="node" id="n2"><div class="node-title">node 2 &mdash; ens_b</div><div class="node-label">&mdash;</div><div class="node-meta">no packets yet</div></div>
  <div class="node" id="n3"><div class="node-title">node 3 &mdash; ens_c</div><div class="node-label">&mdash;</div><div class="node-meta">no packets yet</div></div>
</div>
<p class="panel-title">Fusion decisions (newest first)</p>
<div class="fusion-list" id="fl"></div>
<script>
const $ = id => document.getElementById(id);
const LABELS = ["yes","no","up","down","left","right","on","off","stop","go","unknown","silence"];
const AGG_MODES = ["mean_logits","temperature_scaled","learned_weights"];
const STALE_MS = 4000;
const lastSeen = {1:0, 2:0, 3:0};
function labelText(idx) { return (idx >= 0 && idx < LABELS.length) ? LABELS[idx] : "?"; }
function nodeFresh(id) {
  const el = $("n"+id); if (!el) return;
  const dt = Date.now() - (lastSeen[id]||0);
  el.classList.toggle("online", dt < STALE_MS);
  el.classList.toggle("stale", lastSeen[id] && dt >= STALE_MS);
}
setInterval(() => { for (const k of [1,2,3]) nodeFresh(k); }, 1000);
function setNode(n) {
  const el = $("n" + n.node); if (!el) return;
  lastSeen[n.node] = Date.now();
  const lbl = labelText(n.label);
  const lblEl = el.querySelector(".node-label");
  lblEl.textContent = lbl;
  lblEl.className = "node-label label-" + lbl;
  el.querySelector(".node-meta").textContent =
    "score=" + n.score + " margin=" + n.margin + " packets=" + n.packets;
  nodeFresh(n.node);
}
function pushFusion(f) {
  const lbl = labelText(f.label);
  const t = new Date().toLocaleTimeString();
  const row = document.createElement("div");
  row.className = "fusion";
  row.innerHTML =
    '<span class="fusion-label label-'+lbl+'">'+lbl+'</span>' +
    '<span class="fusion-meta">score='+f.score+' margin='+f.margin+' voters='+f.voters+' &middot; '+t+'</span>';
  const fl = $("fl");
  fl.insertBefore(row, fl.firstChild);
  while (fl.childElementCount > 50) fl.removeChild(fl.lastChild);
}
function applyCounters(c) {
  if (!c) return;
  if ("fusion"   in c) $("ft").textContent  = c.fusion;
  if ("packets"  in c) $("pk").textContent  = c.packets;
  if ("rejected" in c) $("rj").textContent  = c.rejected;
  if ("agg_mode" in c) $("agg").textContent = AGG_MODES[c.agg_mode] || ("mode "+c.agg_mode);
  if ("uptime_s" in c) $("up").textContent  = c.uptime_s;
}
let ws;
function connect() {
  ws = new WebSocket("ws://" + location.host + "/ws");
  ws.onopen = () => { $("conn").textContent = "live"; $("conn").className = "conn-on"; };
  ws.onclose = () => { $("conn").textContent = "disconnected"; $("conn").className = "conn-off"; setTimeout(connect, 1000); };
  ws.onmessage = e => {
    let m; try { m = JSON.parse(e.data); } catch (_) { return; }
    if (m.type === "snapshot") {
      (m.nodes||[]).forEach(setNode);
      (m.fusion||[]).forEach(pushFusion);
      applyCounters(m.counters);
    } else if (m.type === "node") {
      setNode(m); applyCounters(m.counters);
    } else if (m.type === "fusion") {
      pushFusion(m); applyCounters(m.counters);
    } else if (m.type === "stats") {
      applyCounters(m);
    }
  };
}
connect();
</script>
</body>
</html>)HTML";
