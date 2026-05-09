#pragma once

#include <pgmspace.h>

static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distributed NN Network</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;
  --text:#e0e0e0;--dim:#888;--accent:#5b9bd5;
  --green:#4caf50;--red:#ef5350;--orange:#ffa726;
  --font:'Segoe UI',system-ui,sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:14px;line-height:1.5}
a{color:var(--accent)}

/* Header */
.header{background:#14161e;border-bottom:1px solid var(--border);padding:12px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
.header h1{font-size:18px;font-weight:600;letter-spacing:.5px}
.header .meta{font-size:12px;color:var(--dim);display:flex;gap:16px;flex-wrap:wrap}
.header .meta span{white-space:nowrap}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:middle}
.dot-on{background:var(--green)}.dot-off{background:var(--red)}

/* Layout */
.container{padding:16px 24px;display:flex;flex-direction:column;gap:16px}
.row{display:flex;gap:16px;flex-wrap:wrap}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;flex:1;min-width:220px}
.card h2{font-size:14px;font-weight:600;color:var(--accent);margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}

/* Node cards */
.nodes-row{display:flex;gap:12px;flex-wrap:wrap}
.node-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;min-width:200px;flex:1;max-width:300px}
.node-card .name{font-size:15px;font-weight:600;margin-bottom:4px}
.node-card .type{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px}
.node-card .stat{margin-top:8px;font-size:12px;color:var(--dim)}
.node-card .stat b{color:var(--text)}
.node-card .result{margin-top:6px;font-size:16px;font-weight:700;color:var(--accent)}

/* Table */
.feed-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:8px 10px;border-bottom:2px solid var(--border);color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
tr:hover td{background:rgba(91,155,213,.06)}
.score-bar{display:inline-block;height:14px;border-radius:3px;background:var(--accent);vertical-align:middle;min-width:2px}
.score-val{display:inline-block;width:36px;text-align:right;margin-right:4px;font-weight:600}

/* Stats row */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.stat-box{text-align:center;padding:12px;background:rgba(91,155,213,.07);border-radius:6px}
.stat-box .val{font-size:24px;font-weight:700;color:var(--accent)}
.stat-box .lbl{font-size:11px;color:var(--dim);margin-top:2px;text-transform:uppercase;letter-spacing:.5px}

/* Log */
.log-area{max-height:220px;overflow-y:auto;font-family:'Consolas','Courier New',monospace;font-size:12px;background:#12141b;border-radius:6px;padding:10px}
.log-line{padding:1px 0;color:var(--dim)}
.log-line .ts{color:var(--accent);margin-right:8px}

/* WS indicator */
.ws-status{font-size:11px}
.ws-ok{color:var(--green)}.ws-err{color:var(--red)}

/* Responsive */
@media(max-width:700px){
  .container{padding:10px}
  .row{flex-direction:column}
  .node-card{max-width:100%}
}
</style>
</head>
<body>

<div class="header">
  <h1>Distributed NN Network</h1>
  <div class="meta">
    <span>IP: <b id="hIP">-</b></span>
    <span>Uptime: <b id="hUptime">-</b></span>
    <span>Nodes: <b id="hNodes">0/0</b></span>
    <span class="ws-status" id="wsStatus">WS: connecting...</span>
  </div>
</div>

<div class="container">
  <!-- Node cards -->
  <div>
    <h2 style="font-size:13px;color:var(--dim);margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px">Nodes</h2>
    <div class="nodes-row" id="nodesRow">
      <div class="node-card" style="color:var(--dim);display:flex;align-items:center;justify-content:center;min-height:90px">
        Waiting for nodes...
      </div>
    </div>
  </div>

  <!-- Stats -->
  <div class="card">
    <h2>Statistics</h2>
    <div class="stats-grid">
      <div class="stat-box"><div class="val" id="sTotal">0</div><div class="lbl">Processed</div></div>
      <div class="stat-box"><div class="val" id="sQueue">0</div><div class="lbl">In Queue</div></div>
      <div class="stat-box"><div class="val" id="sAvgLat">-</div><div class="lbl">Avg Queue Lat.</div></div>
      <div class="stat-box"><div class="val" id="sRate">-</div><div class="lbl">Packets / sec</div></div>
      <div class="stat-box"><div class="val" id="sOnline">0</div><div class="lbl">Online Nodes</div></div>
    </div>
  </div>

  <!-- Live feed -->
  <div class="card">
    <h2>Live Feed</h2>
    <div class="feed-wrap">
      <table>
        <thead>
          <tr>
            <th>Time</th><th>Node</th>
            <th>Top-1</th><th>Score</th>
            <th>Top-2</th><th>Score</th>
            <th>Top-3</th><th>Score</th>
            <th>Infer ms</th><th>Q-Lat ms</th>
          </tr>
        </thead>
        <tbody id="feedBody"></tbody>
      </table>
    </div>
  </div>

  <!-- Log -->
  <div class="card">
    <h2>System Log</h2>
    <div class="log-area" id="logArea"></div>
  </div>
</div>

<script>
(function(){
  const MAX_FEED = 60;
  const MAX_LOG  = 40;

  const $  = id => document.getElementById(id);
  const feedBody  = $('feedBody');
  const logArea   = $('logArea');
  const nodesRow  = $('nodesRow');

  let nodes = {};
  let totalProcessed = 0;
  let latencySum = 0;
  let latencyCount = 0;
  let rateTs = [];

  function msToTime(ms){
    let s = Math.floor(ms/1000);
    let m = Math.floor(s/60); s %= 60;
    let h = Math.floor(m/60); m %= 60;
    let milli = ms % 1000;
    return String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+
           String(s).padStart(2,'0')+'.'+String(milli).padStart(3,'0');
  }

  function scoreBar(score){
    return '<span class="score-val">'+score+'%</span>'+
           '<span class="score-bar" style="width:'+Math.max(score,2)+'px"></span>';
  }

  function nodeTypeStr(t){
    if(t===1) return 'CAMERA';
    if(t===2) return 'MICRO';
    return 'GENERIC';
  }

  function updateNodes(){
    let html='';
    let onlineC=0, totalC=0;
    for(let id in nodes){
      let n = nodes[id];
      totalC++;
      let online = n.online;
      if(online) onlineC++;
      html += '<div class="node-card">'+
        '<div class="name"><span class="status-dot '+(online?'dot-on':'dot-off')+'"></span>'+n.id+'</div>'+
        '<div class="type">'+nodeTypeStr(n.type)+'</div>'+
        '<div class="stat">Packets: <b>'+n.packets+'</b></div>'+
        '<div class="stat">Last: <b>'+msToTime(n.lastSeen)+'</b></div>'+
        '<div class="result">'+n.topLabel+' '+n.topScore+'%</div>'+
      '</div>';
    }
    if(!totalC) html='<div class="node-card" style="color:var(--dim);display:flex;align-items:center;justify-content:center;min-height:90px">Waiting for nodes...</div>';
    nodesRow.innerHTML = html;
    $('hNodes').textContent = onlineC+'/'+totalC;
    $('sOnline').textContent = onlineC;
  }

  function addFeedRow(d){
    let tr = document.createElement('tr');
    let qlat = (d.processed_at && d.received_at) ? (d.processed_at - d.received_at) : '-';
    tr.innerHTML =
      '<td>'+msToTime(d.received_at)+'</td>'+
      '<td>'+d.node_id+'</td>'+
      '<td>'+d.top[0].label+'</td><td>'+scoreBar(d.top[0].score)+'</td>'+
      '<td>'+d.top[1].label+'</td><td>'+scoreBar(d.top[1].score)+'</td>'+
      '<td>'+d.top[2].label+'</td><td>'+scoreBar(d.top[2].score)+'</td>'+
      '<td>'+d.inference_ms+'</td>'+
      '<td>'+qlat+'</td>';
    feedBody.prepend(tr);
    while(feedBody.children.length > MAX_FEED) feedBody.removeChild(feedBody.lastChild);
  }

  function addLog(ts, text){
    let div = document.createElement('div');
    div.className = 'log-line';
    div.innerHTML = '<span class="ts">['+msToTime(ts)+']</span> '+text;
    logArea.prepend(div);
    while(logArea.children.length > MAX_LOG) logArea.removeChild(logArea.lastChild);
  }

  function updateStats(d){
    totalProcessed++;
    $('sTotal').textContent = totalProcessed;

    if(d.processed_at && d.received_at){
      let lat = d.processed_at - d.received_at;
      latencySum += lat;
      latencyCount++;
      $('sAvgLat').textContent = Math.round(latencySum/latencyCount)+' ms';
    }

    let now = Date.now();
    rateTs.push(now);
    while(rateTs.length && rateTs[0] < now - 5000) rateTs.shift();
    let rate = (rateTs.length / 5).toFixed(1);
    $('sRate').textContent = rate;
  }

  function handleResult(d){
    addFeedRow(d);
    updateStats(d);

    let n = nodes[d.node_id];
    if(!n){
      n = {id:d.node_id, type:d.node_type, packets:0, lastSeen:0, topLabel:'-', topScore:0, online:true};
      nodes[d.node_id] = n;
    }
    n.packets++;
    n.lastSeen = d.received_at;
    n.topLabel = d.top[0].label;
    n.topScore = d.top[0].score;
    n.online = true;
    n.type = d.node_type;
    updateNodes();
  }

  function handleState(s){
    if(s.uptime)  $('hUptime').textContent = msToTime(s.uptime);
    if(s.ip)      $('hIP').textContent = s.ip;
    if(s.queue_depth !== undefined) $('sQueue').textContent = s.queue_depth;

    if(s.nodes){
      for(let nd of s.nodes){
        nodes[nd.id] = {
          id: nd.id, type: nd.type, packets: nd.packets,
          lastSeen: nd.last_seen, topLabel: nd.top_label||'-',
          topScore: nd.top_score||0, online: nd.online
        };
      }
      updateNodes();
    }
    if(s.history){
      for(let i=s.history.length-1; i>=0; i--) addFeedRow(s.history[i]);
    }
    if(s.logs){
      for(let i=s.logs.length-1; i>=0; i--) addLog(s.logs[i].t, s.logs[i].m);
    }
    totalProcessed = s.total_processed || 0;
    $('sTotal').textContent = totalProcessed;
  }

  // ---- WebSocket ----------------------------------------------------------
  let ws;
  let reconnectDelay = 1000;

  function wsConnect(){
    let loc = window.location;
    let uri = 'ws://'+loc.hostname+':'+loc.port+'/ws';
    ws = new WebSocket(uri);

    ws.onopen = function(){
      $('wsStatus').innerHTML = '<span class="ws-ok">WS: connected</span>';
      reconnectDelay = 1000;
    };
    ws.onclose = function(){
      $('wsStatus').innerHTML = '<span class="ws-err">WS: disconnected</span>';
      setTimeout(wsConnect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, 10000);
    };
    ws.onerror = function(){ ws.close(); };

    ws.onmessage = function(evt){
      try{
        let msg = JSON.parse(evt.data);
        if(msg.type === 'result')      handleResult(msg.data);
        else if(msg.type === 'state')  handleState(msg.data);
        else if(msg.type === 'log')    addLog(msg.data.t, msg.data.m);
        else if(msg.type === 'nodes')  {
          if(msg.data && msg.data.nodes){
            for(let nd of msg.data.nodes){
              if(nodes[nd.id]) nodes[nd.id].online = nd.online;
            }
            updateNodes();
          }
        }
        else if(msg.type === 'queue')  $('sQueue').textContent = msg.data.depth;
      }catch(e){ console.error('WS parse error',e); }
    };
  }

  wsConnect();

  setInterval(function(){
    $('sQueue').textContent;
  }, 2000);
})();
</script>

</body>
</html>
)rawliteral";
