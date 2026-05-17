// web_page.h — embedded dashboard HTML stored in PROGMEM flash.
// Served via AsyncWebServer::send_P() — never loaded into RAM.
// Single-file SPA: inline CSS + inline JS, no external dependencies.

#pragma once

static const char kDashboardHtml[] PROGMEM = R"rawhtml(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hash KWS Master</title>
<style>
:root{
  --bg:#111;--card:#1c1c1e;--border:#2a2a2e;
  --txt:#e0e0e0;--sub:#808080;
  --green:#41d287;--amber:#ffae3a;--red:#f06363;--blue:#6ec3ff;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:14px/1.5 system-ui,sans-serif;padding:14px;max-width:900px;margin:0 auto}
a{color:var(--blue)}

/* ── Header ── */
.hdr{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;gap:8px}
.hdr h1{font-size:17px;font-weight:700;line-height:1.2}
.hdr .sub{font-size:11px;color:var(--sub);margin-top:2px}
.badge{font-size:11px;font-weight:700;padding:3px 9px;border-radius:12px;white-space:nowrap;flex-shrink:0;margin-top:2px}
.badge.live{background:#1a3d2a;color:var(--green)}
.badge.disc{background:#3a1a1a;color:var(--red)}

/* ── Counters strip ── */
.ctr{background:var(--card);border:1px solid var(--border);border-radius:8px;
     padding:9px 14px;margin-bottom:12px;display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--sub)}
.ctr span{color:var(--txt);font-weight:600}

/* ── Ensemble hero ── */
.hero{background:linear-gradient(135deg,#152233 0%,#0f151c 100%);
      border:1px solid var(--border);border-radius:10px;
      padding:16px 18px;margin-bottom:12px;display:flex;align-items:center;gap:18px;
      transition:opacity .3s}
.hero.faded{opacity:.45}
.hero-left{flex-shrink:0}
.hero-cap{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:.7px;margin-bottom:4px}
.hero-lbl{font-size:44px;font-weight:800;line-height:1;letter-spacing:-0.5px}
.hero-right{flex:1;display:flex;flex-direction:column;gap:6px;font-size:12px;color:var(--sub);min-width:0}
.hero-right .row{display:flex;flex-wrap:wrap;gap:10px}
.hero-right b{color:var(--txt);font-weight:600}
.mode-badge{display:inline-block;font-size:10px;font-weight:700;text-transform:uppercase;
            letter-spacing:.5px;padding:2px 7px;border-radius:10px;
            background:#22344a;color:var(--blue);border:1px solid #2c4760}

/* ── Agg-diag tiny strip ── */
.diag{background:var(--card);border:1px dashed var(--border);border-radius:8px;
      padding:7px 12px;margin-bottom:12px;display:flex;flex-wrap:wrap;gap:12px;
      font-size:11px;color:var(--sub);font-family:ui-monospace,Menlo,monospace}
.diag span{color:var(--txt);font-weight:600}
.diag .warn{color:var(--amber)}

/* ── Node tiles ── */
.nodes{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}
@media(max-width:520px){.nodes{grid-template-columns:1fr}}
.tile{background:var(--card);border:1px solid var(--border);border-left:4px solid #333;
      border-radius:8px;padding:12px;transition:border-left-color .3s,opacity .3s}
.tile.online{border-left-color:var(--green)}
.tile.stale {border-left-color:var(--amber)}
.tile.offline{border-left-color:var(--red);opacity:.62}
.tile.never  {border-left-color:#3a3a3e;opacity:.55}
.tile-head{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px}
.tile-lbl{font-size:30px;font-weight:800;margin-bottom:3px;line-height:1}
.tile-meta{font-size:11px;color:var(--sub)}

/* ── Section title ── */
.sec{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}

/* ── Latency block ── */
.lat-wrap{background:var(--card);border:1px solid var(--border);border-radius:8px;
          padding:12px;margin-bottom:12px}
.lat-row{display:flex;align-items:center;gap:10px;margin-bottom:7px}
.lat-row:last-child{margin-bottom:0}
.lat-name{font-size:11px;color:var(--sub);width:56px;flex-shrink:0}
.lat-bar-bg{flex:1;height:6px;background:#222;border-radius:3px;overflow:hidden}
.lat-bar{height:100%;border-radius:3px;transition:width .4s,background .4s}
.lat-nums{font-size:11px;color:var(--sub);text-align:right;width:180px;flex-shrink:0;white-space:nowrap}

/* ── Video section (separate MCU, image classifier) ── */
.video-wrap{background:var(--card);border:1px solid var(--border);border-radius:10px;
            padding:14px 16px;margin-top:12px;border-left:4px solid #5a6470;
            transition:border-left-color .3s}
.video-wrap.online{border-left-color:#c5b3ff}
.video-wrap.stale {border-left-color:var(--amber)}
.video-head{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
.video-title{font-size:12px;color:var(--sub);text-transform:uppercase;letter-spacing:.7px}
.video-grid{display:grid;grid-template-columns:1.4fr 1fr;gap:14px}
@media(max-width:520px){.video-grid{grid-template-columns:1fr}}
.video-pri{padding-right:12px;border-right:1px solid var(--border)}
@media(max-width:520px){.video-pri{border-right:none;padding-right:0;border-bottom:1px solid var(--border);padding-bottom:10px}}
.v-cap{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.v-lbl{font-size:38px;font-weight:800;line-height:1;letter-spacing:-0.5px;margin-bottom:4px}
.v-sub{font-size:11px;color:var(--sub);font-family:ui-monospace,Menlo,monospace}
.v-raw-lbl{font-size:22px;font-weight:700;line-height:1;margin-bottom:3px}
.v-meta{font-size:11px;color:var(--sub);margin-top:8px;display:flex;flex-wrap:wrap;gap:10px}
.v-meta b{color:var(--txt);font-weight:600}

/* ── Fusion list ── */
.fusion-wrap{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px}
.f-item{display:flex;align-items:center;gap:10px;padding:5px 0;
        border-bottom:1px solid var(--border);font-size:13px}
.f-item:last-child{border-bottom:none}
.f-lbl{font-weight:700;width:68px;flex-shrink:0}
.f-meta{flex:1;font-size:11px;color:var(--sub)}
.f-mode{font-size:10px;color:var(--blue);background:#22344a;border:1px solid #2c4760;
        padding:1px 6px;border-radius:8px;font-weight:600;letter-spacing:.3px;flex-shrink:0}
.f-time{font-size:11px;color:var(--sub);white-space:nowrap}
.empty{font-size:12px;color:var(--sub);padding:4px 0}
</style>
</head>
<body>

<!-- ── Header ── -->
<div class="hdr">
  <div>
    <h1>Hash KWS Master</h1>
    <div class="sub">3 hashed students &mdash; ensemble aggregation on this ESP32</div>
  </div>
  <div id="badge" class="badge disc">&#9675; disconnected</div>
</div>

<!-- ── Counters strip ── -->
<div class="ctr">
  <div>nodes&nbsp;<span id="c-nodes">&#x2014;</span></div>
  <div>fusion&nbsp;<span id="c-fusion">&#x2014;</span></div>
  <div>packets&nbsp;<span id="c-packets">&#x2014;</span></div>
  <div>rejected&nbsp;<span id="c-rejected">&#x2014;</span></div>
  <div>aggregator&nbsp;<span id="c-agg">&#x2014;</span></div>
  <div>min&nbsp;voters&nbsp;<span id="c-minv">&#x2014;</span></div>
  <div>window&nbsp;<span id="c-win">&#x2014;</span></div>
  <div>uptime&nbsp;<span id="c-up">&#x2014;</span></div>
</div>

<!-- ── Ensemble hero (latest averaged decision) ── -->
<div class="hero" id="hero-wrap">
  <div class="hero-left">
    <div class="hero-cap">Ensemble (averaged across nodes)</div>
    <div class="hero-lbl" id="hero-lbl" style="color:#5a6470">&#x2014;</div>
  </div>
  <div class="hero-right">
    <div class="row">
      <div>mode&nbsp;<span class="mode-badge" id="hero-mode">&#x2014;</span></div>
      <div>voters&nbsp;<b id="hero-voters">&#x2014;</b>/<span id="hero-need">&#x2014;</span></div>
      <div>score&nbsp;<b id="hero-score">&#x2014;</b></div>
      <div>margin&nbsp;<b id="hero-margin">&#x2014;</b></div>
    </div>
    <div class="row" style="font-size:11px">
      <div>last&nbsp;<b id="hero-time">never</b></div>
      <div>total&nbsp;decisions&nbsp;<b id="hero-total">0</b></div>
    </div>
  </div>
</div>

<!-- ── Aggregator diagnostics (live) ── -->
<div class="diag">
  <div>resolves&nbsp;<span id="d-res">0</span></div>
  <div>decisions&nbsp;<span id="d-dec">0</span></div>
  <div>no&nbsp;voters&nbsp;<span id="d-nov" class="warn">0</span></div>
  <div>low&nbsp;voters&nbsp;<span id="d-low" class="warn">0</span></div>
  <div>dedup&nbsp;skip&nbsp;<span id="d-ded">0</span></div>
  <div>last&nbsp;voters&nbsp;<span id="d-lv">0</span></div>
  <div>last&nbsp;mode&nbsp;<span id="d-lm">&#x2014;</span></div>
</div>

<!-- ── Node tiles ── -->
<div class="nodes">
  <div class="tile" id="tile-0">
    <div class="tile-head">Node 1 &mdash; ens_a</div>
    <div class="tile-lbl" id="lbl-0" style="color:#5a6470">&#x2014;</div>
    <div class="tile-meta" id="meta-0">no packets yet</div>
  </div>
  <div class="tile" id="tile-1">
    <div class="tile-head">Node 2 &mdash; ens_b</div>
    <div class="tile-lbl" id="lbl-1" style="color:#5a6470">&#x2014;</div>
    <div class="tile-meta" id="meta-1">no packets yet</div>
  </div>
  <div class="tile" id="tile-2">
    <div class="tile-head">Node 3 &mdash; ens_c</div>
    <div class="tile-lbl" id="lbl-2" style="color:#5a6470">&#x2014;</div>
    <div class="tile-meta" id="meta-2">no packets yet</div>
  </div>
</div>

<!-- ── Latency block ── -->
<div class="lat-wrap">
  <div class="sec">Inference latency (invoke_ms per node)</div>
  <div id="lat-rows">
    <div class="lat-row"><div class="lat-name">Node 1</div><div class="lat-bar-bg"><div class="lat-bar" style="width:0%"></div></div><div class="lat-nums">no data</div></div>
    <div class="lat-row"><div class="lat-name">Node 2</div><div class="lat-bar-bg"><div class="lat-bar" style="width:0%"></div></div><div class="lat-nums">no data</div></div>
    <div class="lat-row"><div class="lat-name">Node 3</div><div class="lat-bar-bg"><div class="lat-bar" style="width:0%"></div></div><div class="lat-nums">no data</div></div>
  </div>
</div>

<!-- ── Fusion list ── -->
<div class="fusion-wrap">
  <div class="sec">Fusion decisions (newest first)</div>
  <div id="fusion-list"><div class="empty">No decisions yet</div></div>
</div>

<!-- ── Video MCU (independent image classifier, optional) ── -->
<div class="video-wrap" id="video-wrap">
  <div class="video-head">
    <div>
      <div class="video-title">Video MCU &mdash; image classifier (node 4)</div>
      <div class="v-meta" id="video-meta-top">
        <span>status&nbsp;<b id="v-status">offline</b></span>
        <span>window&nbsp;<b id="v-win">&#x2014;</b></span>
        <span>ring&nbsp;<b id="v-ring">&#x2014;</b></span>
      </div>
    </div>
    <div id="v-badge" class="badge disc">&#9675; offline</div>
  </div>
  <div class="video-grid">
    <div class="video-pri">
      <div class="v-cap">Smoothed (mean over 1.2&thinsp;s window)</div>
      <div class="v-lbl" id="v-agg-lbl" style="color:#5a6470">&#x2014;</div>
      <div class="v-sub" id="v-agg-sub">no frames yet</div>
      <div class="v-meta">
        <span>voters&nbsp;<b id="v-agg-voters">0</b></span>
        <span>score&nbsp;<b id="v-agg-score">&#x2014;</b></span>
        <span>margin&nbsp;<b id="v-agg-margin">&#x2014;</b></span>
        <span>decisions&nbsp;<b id="v-agg-dec">0</b></span>
        <span>dedup&nbsp;skip&nbsp;<b id="v-agg-ded">0</b></span>
      </div>
    </div>
    <div>
      <div class="v-cap">Last raw frame</div>
      <div class="v-raw-lbl" id="v-raw-lbl" style="color:#5a6470">&#x2014;</div>
      <div class="v-sub" id="v-raw-sub">no packets yet</div>
      <div class="v-meta">
        <span>packets&nbsp;<b id="v-raw-pkts">0</b></span>
        <span>seq&nbsp;<b id="v-raw-seq">&#x2014;</b></span>
        <span>last&nbsp;<b id="v-raw-age">never</b></span>
        <span>lat&nbsp;<b id="v-raw-lat">&#x2014;</b></span>
      </div>
    </div>
  </div>
</div>

<script>
(function(){
// ── Constants ────────────────────────────────────────────────────────────────
const COLORS={
  yes:'#41d287',no:'#f06363',up:'#6ec3ff',down:'#ffae3a',
  left:'#c5b3ff',right:'#ff85c8',on:'#b6ff63',off:'#888888',
  stop:'#ff7373',go:'#4ad6ff',unknown:'#5a6470',silence:'#5a6470',
  // Video classes — purple family so they don't collide with audio palette.
  person:'#c5b3ff',face:'#e0c0ff',car:'#ffae3a',bicycle:'#41d287',
  motorbike:'#ff85c8',dog:'#ffd166',cat:'#ffa07a',bird:'#6ec3ff',
  hand:'#b6ff63',stop_sign:'#ff7373',no_obj:'#5a6470'
};
const AGG_NAMES=['mean_logits','temperature_scaled','learned_weights'];
const LAT_MAX_MS=800;   // bar is 100% at this value
const STALE_MS=4000;    // tile turns amber after this
const VIDEO_STALE_MS=5000; // video card turns amber after this
// Noise labels are suppressed in plates (hero + tiles). They never reach
// the fusion list because the aggregator filters them too, but we keep
// a JS-side guard so any stale entry from before this change still hides.
const NOISE_LABELS=new Set(['unknown','silence','_unknown_','_silence_']);
function isNoise(label){ return !!label && NOISE_LABELS.has(label); }

// ── State ────────────────────────────────────────────────────────────────────
// lastMs = browser Date.now() when we last received a packet for that node
const nodes=[
  {label:null,score:0,margin:0,packets:0,lastMs:null,lat:{min:0,med:0,p95:0,max:0}},
  {label:null,score:0,margin:0,packets:0,lastMs:null,lat:{min:0,med:0,p95:0,max:0}},
  {label:null,score:0,margin:0,packets:0,lastMs:null,lat:{min:0,med:0,p95:0,max:0}},
];
let fusionList=[];   // [{label,score,margin,voters,mode,mode_name,at}] newest first, max 50
let lastHero=null;   // most recent fusion record for the hero block
let videoState=null; // last `video` subobject from WS (null = never received)
let videoLastMs=null;// browser ts of last video update, for staleness check
let ws=null,reconnTimer=null;

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect(){
  const url='ws://'+location.host+'/ws';
  try{ ws=new WebSocket(url); } catch(e){ schedReconn(); return; }
  ws.onopen=()=>setLive(true);
  ws.onclose=()=>{ setLive(false); schedReconn(); };
  ws.onerror=()=>{};
  ws.onmessage=e=>{
    let m;
    try{ m=JSON.parse(e.data); } catch(ex){ return; }
    if(m.type==='snapshot'){
      (m.nodes||[]).forEach(n=>applyNode(n));
      fusionList=(m.fusion||[]).map(f=>({...f,at:Date.now()}));
      if(fusionList.length){ lastHero=fusionList[0]; }
      applyCounters(m.counters);
      applyDiag(m.agg_diag);
      applyVideo(m.video);
      renderAll();
    } else if(m.type==='video'){
      applyVideo(m.video);
      applyCounters(m.counters);
      renderVideo();
    } else if(m.type==='node'){
      applyNode(m);
      applyCounters(m.counters);
      applyDiag(m.agg_diag);
      renderTile(m.node-1);
      renderLat();
    } else if(m.type==='fusion'){
      const rec={...m,at:Date.now()};
      fusionList.unshift(rec);
      if(fusionList.length>50) fusionList.pop();
      lastHero=rec;
      applyCounters(m.counters);
      applyDiag(m.agg_diag);
      renderHero();
      renderFusion();
    } else if(m.type==='agg_diag'){
      applyCounters(m.counters);
      applyDiag(m.agg_diag);
    }
  };
}
function schedReconn(){
  clearTimeout(reconnTimer);
  reconnTimer=setTimeout(connect,1000);
}

// ── State helpers ─────────────────────────────────────────────────────────────
function applyNode(n){
  const i=n.node-1; if(i<0||i>2) return;
  const s=nodes[i];
  s.label=n.label;
  s.score=n.score;
  s.margin=n.margin;
  s.packets=n.packets;
  s.lastMs=Date.now();
  s.everSeen=n.ever_seen!==false; // when present in payload, trust it
  if(n.lat) s.lat={...n.lat};
}

function applyCounters(c){
  if(!c) return;
  setText('c-fusion',c.fusion);
  setText('c-packets',c.packets);
  setText('c-rejected',c.rejected);
  setText('c-agg',c.agg_mode_name||AGG_NAMES[c.agg_mode]||('mode '+c.agg_mode));
  setText('c-minv',c.min_voters!=null?c.min_voters:'—');
  setText('c-win',c.window_ms!=null?(c.window_ms+'ms'):'—');
  setText('c-up',c.uptime_s+'s');
  setText('hero-need',c.min_voters!=null?c.min_voters:'?');
  setText('hero-total',c.fusion!=null?c.fusion:0);
  // Connectivity pill: e.g. "audio 2/3, video 1/1" or "audio 0/3, video 0/1".
  if(c.audio_total!=null){
    const ao=c.audio_online!=null?c.audio_online:0;
    const at=c.audio_total;
    const vo=c.video_online?1:0;
    setText('c-nodes','audio '+ao+'/'+at+', video '+vo+'/1');
  }
}

function applyDiag(d){
  if(!d) return;
  setText('d-res',d.resolves);
  setText('d-dec',d.decisions);
  setText('d-nov',d.no_voters);
  setText('d-low',d.low_voters);
  setText('d-ded',d.dedup_skip);
  setText('d-lv',d.last_voters);
  setText('d-lm',d.last_mode_name||AGG_NAMES[d.last_mode]||'—');
}

function applyVideo(v){
  if(!v) return;
  videoState=v;
  if(v.ever_seen) videoLastMs=Date.now();
}

// ── DOM helpers ───────────────────────────────────────────────────────────────
function setText(id,v){ const el=document.getElementById(id); if(el) el.textContent=v; }

function setLive(live){
  const el=document.getElementById('badge');
  if(!el) return;
  el.className='badge '+(live?'live':'disc');
  el.textContent=live?'● live':'○ disconnected';
}

// ── Tile render ───────────────────────────────────────────────────────────────
const OFFLINE_MS=10000;  // tile turns red after this (was just stale-amber before)

function tileClass(i){
  const s=nodes[i];
  if(!s.lastMs) return 'never';
  const age=Date.now()-s.lastMs;
  if(age>OFFLINE_MS) return 'offline';
  if(age>STALE_MS)   return 'stale';
  return 'online';
}

function tileMeta(i){
  const s=nodes[i];
  const cls=tileClass(i);
  if(cls==='never')   return 'never seen';
  const ageS=Math.round((Date.now()-s.lastMs)/1000);
  if(cls==='offline') return 'offline · last '+ageS+'s ago';
  if(cls==='stale')   return 'stale · last '+ageS+'s ago';
  return 'score='+s.score+' margin='+s.margin+' pkts='+s.packets;
}

function renderTile(i){
  const s=nodes[i];
  const tile=document.getElementById('tile-'+i);
  const lblEl=document.getElementById('lbl-'+i);
  const metaEl=document.getElementById('meta-'+i);
  if(!tile) return;
  const cls=tileClass(i);
  tile.className='tile '+cls;
  if(!s.label||cls==='never'||cls==='offline'){
    lblEl.textContent='—';
    lblEl.style.color='#5a6470';
    metaEl.textContent=tileMeta(i);
  } else if(isNoise(s.label)){
    // Don't promote noise labels (unknown/silence) to the plate.
    // The tile stays "online" coloured but the label cell shows a dash;
    // meta tells you which noise class the node actually emitted so you
    // can still see it's not just dead air.
    lblEl.textContent='—';
    lblEl.style.color='#5a6470';
    metaEl.textContent='('+s.label+') · pkts='+s.packets;
  } else {
    lblEl.textContent=s.label;
    lblEl.style.color=COLORS[s.label]||'#e0e0e0';
    metaEl.textContent=tileMeta(i);
  }
}

// ── Latency bar render ────────────────────────────────────────────────────────
function latColor(med){
  if(med<200) return '#41d287';
  if(med<500) return '#ffae3a';
  return '#f06363';
}

function renderLat(){
  const names=['Node 1','Node 2','Node 3'];
  let html='';
  for(let i=0;i<3;i++){
    const lt=nodes[i].lat||{min:0,med:0,p95:0,max:0};
    const pct=Math.min(100,Math.round((lt.p95||0)/LAT_MAX_MS*100));
    const col=latColor(lt.med||0);
    const nums=lt.med?
      'med '+lt.med+'ms p95 '+lt.p95+'ms max '+lt.max+'ms':
      'no data';
    html+='<div class="lat-row">'
      +'<div class="lat-name">'+names[i]+'</div>'
      +'<div class="lat-bar-bg"><div class="lat-bar" style="width:'+pct+'%;background:'+col+'"></div></div>'
      +'<div class="lat-nums">'+nums+'</div>'
      +'</div>';
  }
  const el=document.getElementById('lat-rows');
  if(el) el.innerHTML=html;
}

// ── Fusion list render ────────────────────────────────────────────────────────
function fmtTime(at){
  const d=new Date(at);
  const hh=d.getHours().toString().padStart(2,'0');
  const mm=d.getMinutes().toString().padStart(2,'0');
  const ss=d.getSeconds().toString().padStart(2,'0');
  return hh+':'+mm+':'+ss;
}

function renderFusion(){
  const el=document.getElementById('fusion-list');
  if(!el) return;
  // Drop any legacy noise entries that snuck in before the aggregator
  // started masking them. Keeps the list aligned with the plate policy.
  const visible=fusionList.filter(f=>!isNoise(f.label));
  if(!visible.length){
    el.innerHTML='<div class="empty">No decisions yet</div>';
    return;
  }
  el.innerHTML=visible.map(f=>{
    const col=COLORS[f.label]||'#e0e0e0';
    return '<div class="f-item">'
      +'<div class="f-lbl" style="color:'+col+'">'+f.label+'</div>'
      +'<div class="f-meta">score='+f.score+' margin='+f.margin+' voters='+f.voters+'</div>'
      +((f.mode_name||AGG_NAMES[f.mode])?('<div class="f-mode">'+(f.mode_name||AGG_NAMES[f.mode])+'</div>'):'')
      +'<div class="f-time">'+fmtTime(f.at)+'</div>'
      +'</div>';
  }).join('');
}

// ── Ensemble hero render ──
const HERO_FADE_MS=8000;  // hero result fades after this if no new fusion

function renderHero(){
  const lbl=document.getElementById('hero-lbl');
  const wrap=document.getElementById('hero-wrap');
  if(!lbl) return;
  if(!lastHero){
    if(wrap) wrap.className='hero faded';
    lbl.textContent='—';
    lbl.style.color='#5a6470';
    setText('hero-mode','—');
    setText('hero-voters','—');
    setText('hero-score','—');
    setText('hero-margin','—');
    setText('hero-time','never');
    return;
  }
  const f=lastHero;
  // Defensive: the aggregator already masks noise classes, but if any
  // legacy fusion arrives with one we still refuse to put it on the hero.
  if(isNoise(f.label)){
    if(wrap) wrap.className='hero faded';
    lbl.textContent='—';
    lbl.style.color='#5a6470';
    setText('hero-mode',f.mode_name||AGG_NAMES[f.mode]||'—');
    setText('hero-voters',f.voters!=null?f.voters:'—');
    setText('hero-score','—');
    setText('hero-margin','—');
    setText('hero-time',fmtTime(f.at));
    return;
  }
  const stale=(Date.now()-f.at)>HERO_FADE_MS;
  if(wrap) wrap.className='hero'+(stale?' faded':'');
  lbl.textContent=f.label;
  lbl.style.color=COLORS[f.label]||'#e0e0e0';
  setText('hero-mode',f.mode_name||AGG_NAMES[f.mode]||'—');
  setText('hero-voters',f.voters!=null?f.voters:'—');
  setText('hero-score',f.score!=null?f.score:'—');
  setText('hero-margin',f.margin!=null?f.margin:'—');
  setText('hero-time',fmtTime(f.at));
}

// ── Video card render ────────────────────────────────────────────────────────
function fmtAge(ms){
  if(ms==null) return 'never';
  if(ms<1000)  return Math.round(ms)+'ms ago';
  if(ms<60000) return Math.round(ms/100)/10+'s ago';
  return Math.round(ms/6000)/10+'m ago';
}

function renderVideo(){
  const wrap=document.getElementById('video-wrap');
  const badge=document.getElementById('v-badge');
  if(!wrap||!badge) return;
  const v=videoState;
  // No data ever — show offline placeholder.
  if(!v||!v.ever_seen){
    wrap.className='video-wrap';
    badge.className='badge disc';
    badge.textContent='○ offline';
    setText('v-status','offline');
    setText('v-win',v&&v.agg?(v.agg.window_ms+'ms'):'—');
    setText('v-ring',v&&v.agg?v.agg.ring_size:'—');
    setText('v-agg-lbl','—');
    document.getElementById('v-agg-lbl').style.color='#5a6470';
    setText('v-agg-sub','no frames yet');
    setText('v-agg-voters',0);
    setText('v-agg-score','—');
    setText('v-agg-margin','—');
    setText('v-agg-dec',v&&v.agg?v.agg.decisions:0);
    setText('v-agg-ded',v&&v.agg?v.agg.dedup_skip:0);
    setText('v-raw-lbl','—');
    document.getElementById('v-raw-lbl').style.color='#5a6470';
    setText('v-raw-sub','no packets yet');
    setText('v-raw-pkts',0);
    setText('v-raw-seq','—');
    setText('v-raw-age','never');
    setText('v-raw-lat','—');
    return;
  }
  const browserAge=videoLastMs?(Date.now()-videoLastMs):null;
  const stale=browserAge!=null && browserAge>VIDEO_STALE_MS;
  const online=!stale && v.online!==false;
  wrap.className='video-wrap '+(online?'online':'stale');
  badge.className='badge '+(online?'live':'disc');
  badge.textContent=online?'● online':(stale?'○ stale':'○ offline');
  setText('v-status',online?'online':(stale?'stale':'offline'));
  // Smoothed (aggregator) side.
  const agg=v.agg||{};
  setText('v-win',(agg.window_ms!=null?agg.window_ms+'ms':'—'));
  setText('v-ring',agg.ring_size!=null?agg.ring_size:'—');
  setText('v-agg-voters',agg.voters!=null?agg.voters:0);
  setText('v-agg-score',agg.score!=null?agg.score:'—');
  setText('v-agg-margin',agg.margin!=null?agg.margin:'—');
  setText('v-agg-dec',agg.decisions!=null?agg.decisions:0);
  setText('v-agg-ded',agg.dedup_skip!=null?agg.dedup_skip:0);
  const aggLbl=document.getElementById('v-agg-lbl');
  if(agg.has_decision&&agg.label){
    aggLbl.textContent=agg.label;
    aggLbl.style.color=COLORS[agg.label]||'#c5b3ff';
    setText('v-agg-sub','averaged over '+agg.voters+' frame'+(agg.voters===1?'':'s'));
  } else {
    aggLbl.textContent='—';
    aggLbl.style.color='#5a6470';
    setText('v-agg-sub','no frames in window');
  }
  // Raw last-frame side.
  const rawLbl=document.getElementById('v-raw-lbl');
  if(v.label){
    rawLbl.textContent=v.label;
    rawLbl.style.color=COLORS[v.label]||'#c5b3ff';
    setText('v-raw-sub','score='+v.score+' margin='+v.margin);
  } else {
    rawLbl.textContent='—';
    rawLbl.style.color='#5a6470';
    setText('v-raw-sub','—');
  }
  setText('v-raw-pkts',v.packets!=null?v.packets:0);
  setText('v-raw-seq',v.last_seq!=null?v.last_seq:'—');
  setText('v-raw-age',fmtAge(browserAge));
  const lat=v.lat||{};
  setText('v-raw-lat',lat.med?('med '+lat.med+'ms p95 '+lat.p95+'ms'):'no data');
}

// ── Full render pass ──────────────────────────────────────────────────────────
function renderAll(){
  for(let i=0;i<3;i++) renderTile(i);
  renderLat();
  renderHero();
  renderFusion();
  renderVideo();
}

// ── Stale tile / hero check (every 500 ms) ──────────────────────────────────
setInterval(()=>{
  for(let i=0;i<3;i++) renderTile(i);
  renderHero();    // re-evaluate hero staleness from browser clock
  renderVideo();   // re-evaluate video staleness from browser clock
},500);

// ── Init ──────────────────────────────────────────────────────────────────────
renderLat();    // show empty bars before first data
renderHero();   // show empty hero before first data
renderVideo();  // show offline placeholder before first data
connect();
})();
</script>
</body>
</html>)rawhtml";
