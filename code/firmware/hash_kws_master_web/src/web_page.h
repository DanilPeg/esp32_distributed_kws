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

/* ── Node tiles ── */
.nodes{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}
@media(max-width:520px){.nodes{grid-template-columns:1fr}}
.tile{background:var(--card);border:1px solid var(--border);border-left:4px solid #333;
      border-radius:8px;padding:12px;transition:border-left-color .3s}
.tile.online{border-left-color:var(--green)}
.tile.stale {border-left-color:var(--amber)}
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

/* ── Fusion list ── */
.fusion-wrap{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px}
.f-item{display:flex;align-items:center;gap:10px;padding:5px 0;
        border-bottom:1px solid var(--border);font-size:13px}
.f-item:last-child{border-bottom:none}
.f-lbl{font-weight:700;width:68px;flex-shrink:0}
.f-meta{flex:1;font-size:11px;color:var(--sub)}
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
  <div>fusion&nbsp;<span id="c-fusion">&#x2014;</span></div>
  <div>packets&nbsp;<span id="c-packets">&#x2014;</span></div>
  <div>rejected&nbsp;<span id="c-rejected">&#x2014;</span></div>
  <div>aggregator&nbsp;<span id="c-agg">&#x2014;</span></div>
  <div>uptime&nbsp;<span id="c-up">&#x2014;</span></div>
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

<script>
(function(){
// ── Constants ────────────────────────────────────────────────────────────────
const COLORS={
  yes:'#41d287',no:'#f06363',up:'#6ec3ff',down:'#ffae3a',
  left:'#c5b3ff',right:'#ff85c8',on:'#b6ff63',off:'#888888',
  stop:'#ff7373',go:'#4ad6ff',unknown:'#5a6470',silence:'#5a6470'
};
const AGG_NAMES=['mean_logits','temperature_scaled','learned_weights'];
const LAT_MAX_MS=800;   // bar is 100% at this value
const STALE_MS=4000;    // tile turns amber after this

// ── State ────────────────────────────────────────────────────────────────────
// lastMs = browser Date.now() when we last received a packet for that node
const nodes=[
  {label:null,score:0,margin:0,packets:0,lastMs:null,lat:{min:0,med:0,p95:0,max:0}},
  {label:null,score:0,margin:0,packets:0,lastMs:null,lat:{min:0,med:0,p95:0,max:0}},
  {label:null,score:0,margin:0,packets:0,lastMs:null,lat:{min:0,med:0,p95:0,max:0}},
];
let fusionList=[];   // [{label,score,margin,voters,at}] newest first, max 50
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
      applyCounters(m.counters);
      renderAll();
    } else if(m.type==='node'){
      applyNode(m);
      applyCounters(m.counters);
      renderTile(m.node-1);
      renderLat();
    } else if(m.type==='fusion'){
      fusionList.unshift({...m,at:Date.now()});
      if(fusionList.length>50) fusionList.pop();
      applyCounters(m.counters);
      renderFusion();
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
  if(n.lat) s.lat={...n.lat};
}

function applyCounters(c){
  if(!c) return;
  setText('c-fusion',c.fusion);
  setText('c-packets',c.packets);
  setText('c-rejected',c.rejected);
  setText('c-agg',AGG_NAMES[c.agg_mode]||('mode '+c.agg_mode));
  setText('c-up',c.uptime_s+'s');
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
function tileClass(i){
  const s=nodes[i];
  if(!s.lastMs) return '';
  return (Date.now()-s.lastMs>STALE_MS)?'stale':'online';
}

function renderTile(i){
  const s=nodes[i];
  const tile=document.getElementById('tile-'+i);
  const lblEl=document.getElementById('lbl-'+i);
  const metaEl=document.getElementById('meta-'+i);
  if(!tile) return;
  tile.className='tile '+tileClass(i);
  if(!s.label){
    lblEl.textContent='—';
    lblEl.style.color='#5a6470';
    metaEl.textContent='no packets yet';
  } else {
    lblEl.textContent=s.label;
    lblEl.style.color=COLORS[s.label]||'#e0e0e0';
    metaEl.textContent='score='+s.score+' margin='+s.margin+' pkts='+s.packets;
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
  if(!fusionList.length){
    el.innerHTML='<div class="empty">No decisions yet</div>';
    return;
  }
  el.innerHTML=fusionList.map(f=>{
    const col=COLORS[f.label]||'#e0e0e0';
    return '<div class="f-item">'
      +'<div class="f-lbl" style="color:'+col+'">'+f.label+'</div>'
      +'<div class="f-meta">score='+f.score+' margin='+f.margin+' voters='+f.voters+'</div>'
      +'<div class="f-time">'+fmtTime(f.at)+'</div>'
      +'</div>';
  }).join('');
}

// ── Full render pass ──────────────────────────────────────────────────────────
function renderAll(){
  for(let i=0;i<3;i++) renderTile(i);
  renderLat();
  renderFusion();
}

// ── Stale tile check (every 500 ms) ──────────────────────────────────────────
setInterval(()=>{ for(let i=0;i<3;i++) renderTile(i); },500);

// ── Init ──────────────────────────────────────────────────────────────────────
renderLat();   // show empty bars before first data
connect();
})();
</script>
</body>
</html>)rawhtml";
