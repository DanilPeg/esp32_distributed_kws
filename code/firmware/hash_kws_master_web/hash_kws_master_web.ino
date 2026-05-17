// hash_kws_master_web.ino — 4th ESP32 with on-board dashboard.
//
// Replaces the host-side FastAPI dashboard for the distributed audio KWS
// demo. The master:
//   1. Joins WiFi (STA — your existing router; or AP fallback).
//   2. Joins ESP-NOW on the WiFi's channel.
//   3. Receives HashKwsEspNowPacket from three inference nodes
//      (ens_a / ens_b / ens_c).
//   4. Runs hash_ensemble_aggregator (mean_logits by default).
//   5. AsyncWebServer + WebSocket dashboard at the master's IP — any
//      device on the same WiFi sees it in a browser.
//
// >>> CONFIGURE THESE BEFORE FLASHING <<<
//   #define HASH_KWS_STA_SSID "your-router"
//   #define HASH_KWS_STA_PASS "your-password"
//
// CHANNEL CONSTRAINT (important):
//   When master is in STA mode, ESP-NOW is forced onto the WiFi channel
//   of the router. The three inference boards (micro_speech.ino) have
//   HASH_KWS_ESPNOW_CHANNEL hardcoded at build time. Both must match.
//   Easiest: on Windows host run
//      netsh wlan show interfaces
//   look at the "Channel" line, then rebuild all 4 sketches with
//   -DHASH_KWS_ESPNOW_CHANNEL=<that channel>.
//   Most home routers default to 1, 6, or 11.
//
// Required Arduino libraries (Library Manager):
//   - ESP Async WebServer (mathieucarbou fork) — supports esp32 core 3.x
//   - Async TCP            (mathieucarbou fork) — supports esp32 core 3.x
//   - ArduinoJson          (>= 7.0)
//
// Build flags (or #define at top of this file):
//   -DHASH_KWS_STA_SSID=\"your-router\"     (required)
//   -DHASH_KWS_STA_PASS=\"your-pass\"       (required)
//   -DHASH_KWS_ESPNOW_CHANNEL=1             (must match router's channel)
//   -DHASH_KWS_AGG_MODE=0                   (0=mean_logits, 1=temperature_scaled, 2=learned_weights)
//   -DHASH_KWS_WIFI_MODE=1                  (1=STA default, 0=AP fallback only)
//   -DHASH_KWS_AP_FALLBACK=1                (if STA fails, run as AP)
//   -DHASH_KWS_AP_SSID=\"esp32-hash-master\"
//   -DHASH_KWS_AP_PASS=\"12345678\"

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <ESPmDNS.h>
#include <algorithm>
#include <string.h>

// ESP-NOW receive callback signature changed in ESP-IDF 5.1. PlatformIO's
// Arduino-mode preprocessor auto-generates function prototypes for the .ino
// regardless of #if guards, which makes a version-conditional callback fail
// to compile on platforms where the new type doesn't exist. The shim lives
// in espnow_recv_glue.cpp (a separate .cpp, NOT scanned for prototypes),
// which forwards into HashKwsHandleRecvPacket below. extern "C" so the
// .cpp side can link to it cleanly.
extern "C" void HashKwsEspNowRegisterRecv(void);
extern "C" void HashKwsHandleRecvPacket(const uint8_t* src_mac,
                                        const uint8_t* data, int len);

#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>

#include "hash_ensemble_aggregator.h"
#include "aggregator_params.h"
#include "web_page.h"
#include "camera_link_protocol.h"

#ifndef HASH_KWS_ESPNOW_CHANNEL
#define HASH_KWS_ESPNOW_CHANNEL 1
#endif
#ifndef HASH_KWS_AGG_NUM_NODES
#define HASH_KWS_AGG_NUM_NODES 3
#endif
#ifndef HASH_KWS_AGG_NUM_CLASSES
#define HASH_KWS_AGG_NUM_CLASSES 12
#endif
#ifndef HASH_KWS_AGG_WINDOW_MS
#define HASH_KWS_AGG_WINDOW_MS 1200
#endif
#ifndef HASH_KWS_AGG_MODE
#define HASH_KWS_AGG_MODE 0
#endif
#ifndef HASH_KWS_MASTER_FORWARD_NODE_EVENTS
// Forward per-node infer/episode/emit Serial lines from received packets,
// in the same format inference nodes would print themselves. Lets a host
// demux per-node telemetry from a single USB cable on the master.
#define HASH_KWS_MASTER_FORWARD_NODE_EVENTS 1
#endif
#ifndef HASH_KWS_AGG_NOISE_BOOST
// Logit-domain bias added to noise classes (unknown / silence) before
// picking top1. Suppresses false positives on real keywords when the
// ensemble is in a low-confidence state. Set to 0 to disable. Typical
// useful range: 8..32 (logit units, per-channel int8 mean).
#define HASH_KWS_AGG_NOISE_BOOST 24.0f
#endif
#ifndef HASH_KWS_POST_COMMAND_QUIET_MS
// After publishing a real-command fusion (label < 10), suppress further
// command publications for this many ms. Must be >= HASH_KWS_AGG_WINDOW_MS
// so residual slots from the published command age out of the window
// before any new command is allowed. Kills tail-of-keyword ghosts (e.g.
// `right` -> `up` 1.2s later) and sticky republishes of the same label.
// Silence/unknown are unaffected.
#define HASH_KWS_POST_COMMAND_QUIET_MS 1300
#endif

// ---- Camera trigger config -----------------------------------------------
// Which KWS label fires the camera node. Index into kCategoryLabels:
//   0=yes  1=no  2=up  3=down  4=left  5=right  6=on  7=off  8=stop  9=go
// Change this single number to swap target word.
#ifndef HASH_KWS_CAMERA_TRIGGER_LABEL_IDX
#define HASH_KWS_CAMERA_TRIGGER_LABEL_IDX 0   // default: "yes"
#endif
// Minimum gap between two trigger emissions, regardless of fusion cadence.
// Camera inference is ~280 ms; firing more often than 1/s starts queueing.
#ifndef HASH_KWS_CAMERA_TRIGGER_QUIET_MS
#define HASH_KWS_CAMERA_TRIGGER_QUIET_MS 1500
#endif
// Reply staleness: if a camera reply lands later than this after its trigger,
// flag it on the dashboard but still record it.
#ifndef HASH_KWS_CAMERA_REPLY_STALE_MS
#define HASH_KWS_CAMERA_REPLY_STALE_MS 2500
#endif
#ifndef HASH_KWS_WIFI_MODE
#define HASH_KWS_WIFI_MODE 1   // 1 = STA (default), 0 = AP
#endif
#ifndef HASH_KWS_AP_FALLBACK
#define HASH_KWS_AP_FALLBACK 1 // when STA fails, fall back to AP so demo still works
#endif
#ifndef HASH_KWS_AP_SSID
#define HASH_KWS_AP_SSID "esp32-hash-master"
#endif
#ifndef HASH_KWS_AP_PASS
#define HASH_KWS_AP_PASS "12345678"
#endif
#ifndef HASH_KWS_STA_SSID
#define HASH_KWS_STA_SSID ""    // <-- set your WiFi SSID
#endif
#ifndef HASH_KWS_STA_PASS
#define HASH_KWS_STA_PASS ""    // <-- set your WiFi password
#endif
#ifndef HASH_KWS_STA_CONNECT_TIMEOUT_MS
#define HASH_KWS_STA_CONNECT_TIMEOUT_MS 15000
#endif
#ifndef LED_RGB_PIN
#define LED_RGB_PIN 48
#endif
// mDNS hostname → http://<MDNS_HOSTNAME>.local/
// Browser-friendly entry point that doesn't require knowing the IP.
#ifndef MDNS_HOSTNAME
#define MDNS_HOSTNAME "micro_network"
#endif
// Per-node latency ring depth (in packets). Used to compute min/p50/p95/max
// of invoke_ms shown in the dashboard. 30 covers ~6 s at 5 fps audio rate.
#ifndef LAT_RING_SIZE
#define LAT_RING_SIZE 30
#endif

// ---- Mirror of HashKwsEspNowPacket (must match micro_speech.ino) ----------

constexpr uint32_t kHashKwsEspNowMagic   = 0x4B485731UL;  // "KHW1"
constexpr uint8_t  kHashKwsEspNowVersion = 1;

struct __attribute__((packed)) HashKwsEspNowPacket {
  uint32_t magic;
  uint8_t  version;
  uint8_t  node;
  uint16_t seq;
  uint32_t t_ms;
  uint16_t invoke_ms;
  uint8_t  kind;
  uint8_t  label;
  uint8_t  score;
  uint8_t  margin;
  uint8_t  recent_max;
  uint8_t  flags;
  int8_t   logits[HASH_KWS_AGG_NUM_CLASSES];
  uint16_t crc16;
};

static const char* kCategoryLabels[HASH_KWS_AGG_NUM_CLASSES] = {
  "yes", "no", "up", "down", "left", "right",
  "on", "off", "stop", "go", "unknown", "silence",
};

// ---- Globals --------------------------------------------------------------

static AsyncWebServer server(80);
static AsyncWebSocket ws("/ws");

static hash_kws_ensemble::Aggregator g_aggregator;

// All struct definitions live BEFORE the first function so that PlatformIO's
// Arduino-mode auto-generated function prototypes (inserted at the top of the
// .ino) see complete types when they reference them. Reordering matters here:
// putting FusionEntry/CameraLinkState below the first function would break the
// PIO build with cryptic "X does not name a type" errors at unrelated lines.

struct LatStats { uint16_t min, med, p95, max; };

struct PerNodeState {
  uint8_t  node_id;
  bool     ever_seen;
  uint32_t last_seen_ms;
  uint8_t  last_label;
  uint8_t  last_score;
  uint8_t  last_margin;
  uint32_t packets;
  uint16_t last_invoke_ms;
  uint8_t  last_kind;       // 0=infer, 1=episode, 2=emit
  int8_t   last_recent_max;
  uint16_t last_seq;
  // Rolling invoke_ms ring for min/p50/p95/max stats in the UI.
  uint16_t lat_ring[LAT_RING_SIZE];
  uint8_t  lat_head;
  uint8_t  lat_count;
};

struct FusionEntry {
  uint8_t  label;
  int16_t  score;
  int16_t  margin;
  uint8_t  voters;
  uint32_t time_ms;
};

struct CameraLinkState {
  bool     ever_triggered;
  bool     ever_replied;
  uint16_t last_trigger_id;
  uint8_t  last_trigger_label;     // KWS label index
  uint32_t last_trigger_ms;        // master uptime when fired
  // Last reply (may belong to an older trigger if camera missed one)
  bool     last_reply_ok;
  bool     last_reply_stale;       // reply landed past HASH_KWS_CAMERA_REPLY_STALE_MS
  uint16_t last_reply_trigger_id;
  uint8_t  last_reply_label;       // 0 = no_person, 1 = person
  uint8_t  last_reply_score_q8;
  uint16_t last_reply_latency_ms;
  uint16_t last_reply_fb_ms;
  uint16_t last_reply_invoke_ms;
  uint32_t last_reply_ms;          // master uptime when reply landed
  uint32_t triggers_sent;
  uint32_t triggers_failed_tx;
  uint32_t replies_received;
  uint32_t replies_stale;
};

// Camera-side heartbeat snapshot. Master stores the most recent
// CameraStatusPacket so appendCamera can expose camera health to the
// dashboard ("trig_recv / inf / rep_ok/rep_fail / heap / psram").
// Without this the WS payload would only carry master-side counters and
// the on-board dashboard's camera-status row would stay empty even though
// the camera is broadcasting kind=cam_status every 2 s.
struct CameraRemoteStatus {
  bool     ever_received;
  uint32_t last_ms;                // master uptime when packet landed
  uint32_t status_packets_seen;
  uint8_t  channel;
  uint32_t uptime_ms;
  uint32_t packets_seen;
  uint16_t triggers_received;
  uint16_t triggers_rejected;
  uint16_t inferences_done;
  uint16_t replies_sent;
  uint16_t replies_failed;
  uint16_t free_heap_kb;
  uint16_t psram_free_kb;
};

// ---- Globals (after all structs are declared) ----------------------------
static PerNodeState g_nodes[HASH_KWS_AGG_NUM_NODES];

constexpr int kFusionRingSize = 30;
static FusionEntry g_fusion_ring[kFusionRingSize];
static int g_fusion_ring_count = 0;
static int g_fusion_ring_head = 0;
static uint32_t g_fusion_total = 0;

static volatile uint32_t g_packets_received = 0;
static volatile uint32_t g_packets_rejected = 0;
static uint32_t g_last_decision_print_ms = 0;
static uint32_t g_last_decision_label = 255;
static const uint32_t kDecisionDedupMs = 800;
static uint32_t g_last_command_emit_ms = 0;
// First 10 labels are real commands; 10 = unknown, 11 = silence.
static const uint8_t kNoiseLabelStart = 10;

static const uint8_t kBroadcastMac[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
static bool          g_broadcast_peer_added = false;

static CameraLinkState     g_camera         = {};
static CameraRemoteStatus  g_camera_remote  = {};
static uint16_t            g_trigger_seq    = 0;
static uint32_t            g_last_camera_trigger_ms = 0;

// ---- Latency helpers (require PerNodeState) ------------------------------

static void LatPush(PerNodeState& n, uint16_t invoke_ms) {
  n.lat_ring[n.lat_head] = invoke_ms;
  n.lat_head = (n.lat_head + 1) % LAT_RING_SIZE;
  if (n.lat_count < LAT_RING_SIZE) n.lat_count++;
}

static LatStats LatCompute(const PerNodeState& n) {
  if (n.lat_count == 0) return {0, 0, 0, 0};
  uint16_t buf[LAT_RING_SIZE];
  for (uint8_t i = 0; i < n.lat_count; ++i) buf[i] = n.lat_ring[i];
  std::sort(buf, buf + n.lat_count);
  return {
    buf[0],
    buf[n.lat_count / 2],
    buf[(uint32_t(n.lat_count) * 95u) / 100u],
    buf[n.lat_count - 1]
  };
}

// ---- CRC + packet parse ---------------------------------------------------

static uint16_t HashCrc16(const uint8_t* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint16_t>(data[i]);
    for (int bit = 0; bit < 8; ++bit) {
      if ((crc & 1) != 0) {
        crc = static_cast<uint16_t>((crc >> 1) ^ 0xA001);
      } else {
        crc = static_cast<uint16_t>(crc >> 1);
      }
    }
  }
  return crc;
}

static bool ValidatePacket(const HashKwsEspNowPacket& p) {
  if (p.magic != kHashKwsEspNowMagic) return false;
  if (p.version != kHashKwsEspNowVersion) return false;
  if (p.node == 0 || p.node > HASH_KWS_AGG_NUM_NODES) return false;
  if (p.label >= HASH_KWS_AGG_NUM_CLASSES) return false;
  const uint16_t want = HashCrc16(reinterpret_cast<const uint8_t*>(&p),
                                  sizeof(p) - sizeof(p.crc16));
  return p.crc16 == want;
}

static hash_kws_ensemble::SourceKind PacketKindToSource(uint8_t kind) {
  using hash_kws_ensemble::SourceKind;
  switch (kind) {
    case 1: return SourceKind::kEpisode;
    case 2: return SourceKind::kEmit;
    default: return SourceKind::kInfer;
  }
}

// ---- WebSocket helpers ----------------------------------------------------

static const char* AggModeName(uint8_t m) {
  static const char* names[] = {"mean_logits","temperature_scaled","learned_weights"};
  return (m < 3) ? names[m] : "unknown";
}

static void appendCounters(JsonObject obj) {
  obj["fusion"]         = g_fusion_total;
  obj["packets"]        = static_cast<uint32_t>(g_packets_received);
  obj["rejected"]       = static_cast<uint32_t>(g_packets_rejected);
  obj["agg_mode"]       = HASH_KWS_AGG_MODE;
  obj["agg_mode_name"]  = AggModeName((uint8_t)HASH_KWS_AGG_MODE);
  obj["uptime_s"]       = millis() / 1000UL;
  // Per-source counts so the header pill can read "audio 2/3, camera 1/1".
  const uint32_t now = millis();
  uint8_t audio_online = 0;
  for (uint8_t i = 0; i < HASH_KWS_AGG_NUM_NODES; ++i) {
    if (g_nodes[i].ever_seen && (now - g_nodes[i].last_seen_ms) < 4000UL) {
      audio_online++;
    }
  }
  obj["audio_total"]  = HASH_KWS_AGG_NUM_NODES;
  obj["audio_online"] = audio_online;
  obj["video_online"] = (g_camera.ever_replied &&
                        (now - g_camera.last_reply_ms) < 8000UL) ? 1 : 0;
}

static void appendNode(JsonObject obj, const PerNodeState& n) {
  obj["node"]       = n.node_id;
  obj["label"]      = n.last_label;
  obj["score"]      = n.last_score;
  obj["margin"]     = n.last_margin;
  obj["packets"]    = n.packets;
  obj["invoke_ms"]  = n.last_invoke_ms;
  obj["kind"]       = n.last_kind;
  obj["recent_max"] = n.last_recent_max;
  obj["seq"]        = n.last_seq;
  // Rolling latency stats for the dashboard's per-node bars.
  LatStats ls = LatCompute(n);
  JsonObject lat = obj.createNestedObject("lat");
  lat["min"] = ls.min;
  lat["med"] = ls.med;
  lat["p95"] = ls.p95;
  lat["max"] = ls.max;
  lat["count"] = n.lat_count;
}

static void appendFusion(JsonObject obj, const FusionEntry& f) {
  obj["label"]     = f.label;
  obj["score"]     = f.score;
  obj["margin"]    = f.margin;
  obj["voters"]    = f.voters;
  obj["time_ms"]   = f.time_ms;
  obj["mode"]      = HASH_KWS_AGG_MODE;
  obj["mode_name"] = AggModeName((uint8_t)HASH_KWS_AGG_MODE);
}

static void sendSnapshot(AsyncWebSocketClient* client) {
  // Snapshot is the largest message: 3 nodes + up to 30 fusion entries +
  // camera summary. 6 KB covers it comfortably on both WROOM-32 and C3.
  DynamicJsonDocument doc(6144);
  doc["type"] = "snapshot";
  JsonArray nodes_arr = doc.createNestedArray("nodes");
  for (uint8_t i = 0; i < HASH_KWS_AGG_NUM_NODES; ++i) {
    if (!g_nodes[i].ever_seen) continue;
    appendNode(nodes_arr.createNestedObject(), g_nodes[i]);
  }
  JsonArray fusion_arr = doc.createNestedArray("fusion");
  for (int i = 0; i < g_fusion_ring_count; ++i) {
    int idx = (g_fusion_ring_head - 1 - i + kFusionRingSize) % kFusionRingSize;
    appendFusion(fusion_arr.createNestedObject(), g_fusion_ring[idx]);
  }
  JsonObject counters = doc.createNestedObject("counters");
  appendCounters(counters);
  JsonObject camera = doc.createNestedObject("camera");
  appendCamera(camera);
  String out; serializeJson(doc, out);
  client->text(out);
}

static void broadcastNode(const PerNodeState& n) {
  DynamicJsonDocument doc(1024);
  doc["type"]       = "node";
  doc["node"]       = n.node_id;
  doc["label"]      = n.last_label;
  doc["score"]      = n.last_score;
  doc["margin"]     = n.last_margin;
  doc["packets"]    = n.packets;
  doc["invoke_ms"]  = n.last_invoke_ms;
  doc["kind"]       = n.last_kind;
  doc["recent_max"] = n.last_recent_max;
  doc["seq"]        = n.last_seq;
  LatStats ls = LatCompute(n);
  JsonObject lat = doc.createNestedObject("lat");
  lat["min"] = ls.min;
  lat["med"] = ls.med;
  lat["p95"] = ls.p95;
  lat["max"] = ls.max;
  lat["count"] = n.lat_count;
  JsonObject counters = doc.createNestedObject("counters");
  appendCounters(counters);
  String out; serializeJson(doc, out);
  ws.textAll(out);
}

static void broadcastFusion(const FusionEntry& f) {
  DynamicJsonDocument doc(768);
  doc["type"]      = "fusion";
  doc["label"]     = f.label;
  doc["score"]     = f.score;
  doc["margin"]    = f.margin;
  doc["voters"]    = f.voters;
  doc["time_ms"]   = f.time_ms;
  doc["mode"]      = HASH_KWS_AGG_MODE;
  doc["mode_name"] = AggModeName((uint8_t)HASH_KWS_AGG_MODE);
  JsonObject counters = doc.createNestedObject("counters");
  appendCounters(counters);
  String out; serializeJson(doc, out);
  ws.textAll(out);
}

// ---- Camera link helpers --------------------------------------------------

// Wire shape mirrors loaders._build_camera_summary on the FastAPI bridge
// so the embedded web_page.h JS (a near-verbatim port of the FastAPI
// dashboard template) can read the same field names from both sources.
// Earlier the master was emitting flat `last_reply_*` keys while the JS
// was reading nested `last_reply.*` -> card stayed blank.
static void appendCamera(JsonObject obj) {
  const uint32_t now = millis();
  obj["trigger_word_idx"] = HASH_KWS_CAMERA_TRIGGER_LABEL_IDX;
  obj["trigger_word"]     = kCategoryLabels[HASH_KWS_CAMERA_TRIGGER_LABEL_IDX];

  // Tail-style counters used by the dashboard's compact bar.
  obj["triggers_total_tail"] = g_camera.triggers_sent;
  obj["replies_total_tail"]  = g_camera.replies_received;
  obj["stale_total_tail"]    = g_camera.replies_stale;
  obj["status_total_tail"]   = g_camera_remote.status_packets_seen;
  obj["triggers_failed"]     = g_camera.triggers_failed_tx;

  // last_trigger object (only when we have actually fired one).
  if (g_camera.ever_triggered) {
    JsonObject trg = obj.createNestedObject("last_trigger");
    trg["trigger_id"]   = g_camera.last_trigger_id;
    trg["trigger_word"] = (g_camera.last_trigger_label < HASH_KWS_AGG_NUM_CLASSES)
                          ? kCategoryLabels[g_camera.last_trigger_label]
                          : "?";
    trg["t_ms"]         = g_camera.last_trigger_ms;
    trg["age_sec"]      = (now - g_camera.last_trigger_ms) / 1000.0f;
  }

  // last_reply object.
  if (g_camera.ever_replied) {
    JsonObject rep = obj.createNestedObject("last_reply");
    rep["trigger_id"] = g_camera.last_reply_trigger_id;
    rep["label"]      = (g_camera.last_reply_label == 1) ? "person" : "no_person";
    rep["score"]      = g_camera.last_reply_score_q8 / 255.0f;
    rep["status"]     = g_camera.last_reply_ok ? "ok" : "fail";
    rep["latency_ms"] = g_camera.last_reply_latency_ms;
    rep["fb_ms"]      = g_camera.last_reply_fb_ms;
    rep["invoke_ms"]  = g_camera.last_reply_invoke_ms;
    rep["stale"]      = g_camera.last_reply_stale ? 1 : 0;
    rep["age_sec"]    = (now - g_camera.last_reply_ms) / 1000.0f;
  }

  // last_status object — camera-side heartbeat snapshot.
  if (g_camera_remote.ever_received) {
    JsonObject st = obj.createNestedObject("last_status");
    st["channel"]           = g_camera_remote.channel;
    st["uptime_ms"]         = g_camera_remote.uptime_ms;
    st["packets_seen"]      = g_camera_remote.packets_seen;
    st["triggers_received"] = g_camera_remote.triggers_received;
    st["triggers_rejected"] = g_camera_remote.triggers_rejected;
    st["inferences_done"]   = g_camera_remote.inferences_done;
    st["replies_sent"]      = g_camera_remote.replies_sent;
    st["replies_failed"]    = g_camera_remote.replies_failed;
    st["free_heap_kb"]      = g_camera_remote.free_heap_kb;
    st["psram_free_kb"]     = g_camera_remote.psram_free_kb;
    st["age_sec"]           = (now - g_camera_remote.last_ms) / 1000.0f;
  }
}

static void broadcastCamera() {
  DynamicJsonDocument doc(1536);
  doc["type"] = "camera";
  JsonObject camera = doc.createNestedObject("camera");
  appendCamera(camera);
  JsonObject counters = doc.createNestedObject("counters");
  appendCounters(counters);
  String out; serializeJson(doc, out);
  ws.textAll(out);
}

// Ensure broadcast peer is registered. Idempotent.
// Uses HASH_KWS_ESPNOW_CHANNEL (compile-time) rather than the runtime
// g_active_channel because (a) g_active_channel is declared further down
// in the file (Arduino preprocessor moves functions but not variables), and
// (b) peer.channel=0 makes ESP-NOW use the WiFi interface's current channel
// anyway, so the explicit value here is mostly documentation.
static bool ensureBroadcastPeer() {
  if (g_broadcast_peer_added) return true;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, kBroadcastMac, 6);
  peer.channel = HASH_KWS_ESPNOW_CHANNEL;
  peer.ifidx   = WIFI_IF_STA;
  peer.encrypt = false;
  esp_err_t e = esp_now_add_peer(&peer);
  if (e != ESP_OK && e != ESP_ERR_ESPNOW_EXIST) {
    Serial.printf("[espnow] add_peer(broadcast) fail err=0x%x\n", (unsigned)e);
    return false;
  }
  g_broadcast_peer_added = true;
  return true;
}

static void sendCameraTrigger(uint8_t kws_label_idx) {
  if (!ensureBroadcastPeer()) {
    g_camera.triggers_failed_tx++;
    return;
  }
  CameraCmdPacket cmd = {};
  cmd.kind          = kCamCmdKindInferRequest;
  cmd.trigger_id    = ++g_trigger_seq;
  cmd.t_ms_master   = millis();
  cmd.trigger_label = kws_label_idx;
  CameraLinkSignCmd(&cmd);
  esp_err_t e = esp_now_send(kBroadcastMac,
                             reinterpret_cast<const uint8_t*>(&cmd),
                             sizeof(cmd));
  if (e != ESP_OK) {
    g_camera.triggers_failed_tx++;
    Serial.printf("hash_evt kind=cam_trigger phase=tx status=fail err=0x%x trigger_id=%u\n",
                  (unsigned)e, (unsigned)cmd.trigger_id);
    return;
  }
  g_camera.ever_triggered    = true;
  g_camera.last_trigger_id   = cmd.trigger_id;
  g_camera.last_trigger_label = kws_label_idx;
  g_camera.last_trigger_ms   = cmd.t_ms_master;
  g_camera.triggers_sent++;
  g_last_camera_trigger_ms   = cmd.t_ms_master;
  Serial.printf("hash_evt kind=cam_trigger node=master trigger_id=%u trigger_label=%s\n",
                (unsigned)cmd.trigger_id, kCategoryLabels[kws_label_idx]);
  broadcastCamera();
}

static void handleCameraStatus(const uint8_t* data, int len) {
  if (len != static_cast<int>(sizeof(CameraStatusPacket))) return;
  CameraStatusPacket s;
  memcpy(&s, data, sizeof(s));
  if (!CameraLinkValidateStatus(s)) {
    g_packets_rejected++;
    return;
  }
  // Latch the snapshot so appendCamera can surface camera health on the WS
  // dashboard. Without storing it here the on-board renderCamera would only
  // ever see the all-zero default and the status row would stay empty.
  g_camera_remote.ever_received      = true;
  g_camera_remote.last_ms            = millis();
  g_camera_remote.status_packets_seen++;
  g_camera_remote.channel            = s.channel;
  g_camera_remote.uptime_ms          = s.uptime_ms;
  g_camera_remote.packets_seen       = s.packets_seen;
  g_camera_remote.triggers_received  = s.triggers_received;
  g_camera_remote.triggers_rejected  = s.triggers_rejected;
  g_camera_remote.inferences_done    = s.inferences_done;
  g_camera_remote.replies_sent       = s.replies_sent;
  g_camera_remote.replies_failed     = s.replies_failed;
  g_camera_remote.free_heap_kb       = s.free_heap_kb;
  g_camera_remote.psram_free_kb      = s.psram_free_kb;
  Serial.printf("hash_evt kind=cam_status node=camera channel=%u uptime_ms=%lu "
                "packets_seen=%lu triggers_received=%u triggers_rejected=%u "
                "inferences_done=%u replies_sent=%u replies_failed=%u "
                "free_heap_kb=%u psram_free_kb=%u\n",
                (unsigned)s.channel, (unsigned long)s.uptime_ms,
                (unsigned long)s.packets_seen,
                (unsigned)s.triggers_received, (unsigned)s.triggers_rejected,
                (unsigned)s.inferences_done,
                (unsigned)s.replies_sent, (unsigned)s.replies_failed,
                (unsigned)s.free_heap_kb, (unsigned)s.psram_free_kb);
  broadcastCamera();
}

static void handleCameraReply(const uint8_t* /*src_mac*/,
                              const uint8_t* data, int len) {
  if (len != static_cast<int>(sizeof(CameraReplyPacket))) return;
  CameraReplyPacket r;
  memcpy(&r, data, sizeof(r));
  if (!CameraLinkValidateReply(r)) {
    g_packets_rejected++;
    return;
  }
  const uint32_t now = millis();
  const bool stale = (g_camera.ever_triggered) &&
      ((now - g_camera.last_trigger_ms) > HASH_KWS_CAMERA_REPLY_STALE_MS);

  g_camera.ever_replied          = true;
  g_camera.last_reply_ok         = (r.kind == kCamRepKindInferDone);
  g_camera.last_reply_stale      = stale;
  g_camera.last_reply_trigger_id = r.trigger_id;
  g_camera.last_reply_label      = r.label;
  g_camera.last_reply_score_q8   = r.score_q8;
  g_camera.last_reply_latency_ms = r.latency_ms;
  g_camera.last_reply_fb_ms      = r.fb_ms;
  g_camera.last_reply_invoke_ms  = r.invoke_ms;
  g_camera.last_reply_ms         = now;
  g_camera.replies_received++;
  if (stale) g_camera.replies_stale++;

  const char* lbl_name = (r.label == 1) ? "person" : "no_person";
  Serial.printf("hash_evt kind=cam_reply node=master trigger_id=%u status=%s "
                "label=%s score=%.3f fb_ms=%u invoke_ms=%u latency_ms=%u "
                "stale=%d\n",
                (unsigned)r.trigger_id,
                g_camera.last_reply_ok ? "ok" : "fail",
                lbl_name, r.score_q8 / 255.0f,
                (unsigned)r.fb_ms, (unsigned)r.invoke_ms,
                (unsigned)r.latency_ms, (int)stale);
  broadcastCamera();
}

// ---- ESP-NOW callback -----------------------------------------------------
// HandleRecvPacket multiplexes by length: HashKwsEspNowPacket → aggregator
// path; CameraReplyPacket → handleCameraReply; CameraStatusPacket →
// handleCameraStatus. ESP-NOW callback wrappers below pick the right
// signature per ESP-IDF version and forward into here.

// Real packet handler. Always takes src_mac as plain pointer so the body
// stays version-agnostic. Called from espnow_recv_glue.cpp wrapper which
// adapts the ESP-IDF version-specific callback signature. extern "C" so
// the .cpp side links to it.
extern "C" void HashKwsHandleRecvPacket(const uint8_t* src_mac,
                                        const uint8_t* data, int len) {
  // Camera link multiplex by length (same channel as audio nodes).
  if (len == static_cast<int>(sizeof(CameraReplyPacket))) {
    handleCameraReply(src_mac, data, len);
    return;
  }
  if (len == static_cast<int>(sizeof(CameraStatusPacket))) {
    handleCameraStatus(data, len);
    return;
  }
  if (len != static_cast<int>(sizeof(HashKwsEspNowPacket))) {
    g_packets_rejected++;
    return;
  }
  HashKwsEspNowPacket p;
  memcpy(&p, data, sizeof(p));
  if (!ValidatePacket(p)) {
    g_packets_rejected++;
    return;
  }
  const bool ok = g_aggregator.submit(
      /*node_id=*/p.node,
      /*source=*/PacketKindToSource(p.kind),
      /*device_time_ms=*/p.t_ms,
      /*host_arrival_ms=*/millis(),
      /*logits=*/p.logits,
      /*num_classes=*/HASH_KWS_AGG_NUM_CLASSES);
  if (!ok) {
    g_packets_rejected++;
    return;
  }
  g_packets_received++;
  PerNodeState& slot = g_nodes[p.node - 1];
  slot.node_id         = p.node;
  slot.ever_seen       = true;
  slot.last_seen_ms    = millis();
  slot.last_label      = p.label;
  slot.last_score      = p.score;
  slot.last_margin     = p.margin;
  slot.last_invoke_ms  = p.invoke_ms;
  slot.last_kind       = p.kind;
  slot.last_recent_max = static_cast<int8_t>(p.recent_max);
  slot.last_seq        = p.seq;
  slot.packets++;
  if (p.invoke_ms > 0) LatPush(slot, p.invoke_ms);

#if HASH_KWS_MASTER_FORWARD_NODE_EVENTS
  // 1=infer, 2=emit. Episode state is signalled as a flag bit in p.flags,
  // not as a kind value, so kind=episode never appears on the wire.
  const char* kind_str = (p.kind == 2) ? "emit" : "infer";
  const char* label_str = (p.label < HASH_KWS_AGG_NUM_CLASSES)
                          ? kCategoryLabels[p.label]
                          : "?";
  Serial.printf(
      "hash_evt kind=%s node=%u t=%lu invoke_ms=%u top1=%s top1_score=%d margin=%d recent_max=%d seq=%u\n",
      kind_str,
      static_cast<unsigned>(p.node),
      static_cast<unsigned long>(p.t_ms),
      static_cast<unsigned>(p.invoke_ms),
      label_str,
      static_cast<int>(p.score),
      static_cast<int>(p.margin),
      static_cast<int>(static_cast<int8_t>(p.recent_max)),
      static_cast<unsigned>(p.seq));
#endif

  // Note: WebSocket broadcast from receive context is fine on this stack
  // (esp_now_recv_cb runs in the WiFi task; AsyncTCP queues the writes).
  broadcastNode(slot);
}

// (ESP-NOW recv-callback wrapper lives in espnow_recv_glue.cpp — see
// HashKwsEspNowRegisterRecv extern decl at the top of this file.)

// ---- Aggregator polling + fusion broadcast --------------------------------

static void rgbForLabel(uint8_t label) {
  uint8_t r = 0, g = 0, b = 0;
  switch (label) {
    case 0:  g = 64;  break;
    case 1:  r = 64;  break;
    case 2:  b = 64;  break;
    case 3:  r = 64; g = 32; break;
    case 4:  g = 64; b = 64; break;
    case 5:  r = 64; b = 64; break;
    case 6:  r = 32; g = 64; break;
    case 7:  r = 16; g = 16; b = 16; break;
    case 8:  r = 64; b = 32; break;
    case 9:  g = 32; b = 64; break;
    case 10: r = 8;  g = 8;  break;
    default: break;
  }
  neopixelWrite(LED_RGB_PIN, r, g, b);
}

static void pollAggregator() {
  hash_kws_ensemble::Resolved out;
  g_aggregator.resolve(millis(), &out);
  if (!out.has_decision) return;
  const uint32_t now = millis();
  const bool is_command = out.label < kNoiseLabelStart;
  const bool in_post_command_quiet =
      is_command &&
      ((now - g_last_command_emit_ms) < HASH_KWS_POST_COMMAND_QUIET_MS);
  if (in_post_command_quiet) return;
  const bool dedup = (out.label == g_last_decision_label) &&
                     ((now - g_last_decision_print_ms) < kDecisionDedupMs);
  if (dedup) return;
  g_last_decision_print_ms = now;
  g_last_decision_label = out.label;
  if (is_command) g_last_command_emit_ms = now;

  // Push to ring + counters
  FusionEntry e = { out.label, out.score, out.margin, out.num_voters, now };
  g_fusion_ring[g_fusion_ring_head] = e;
  g_fusion_ring_head = (g_fusion_ring_head + 1) % kFusionRingSize;
  if (g_fusion_ring_count < kFusionRingSize) g_fusion_ring_count++;
  g_fusion_total++;

  rgbForLabel(out.label);
  Serial.printf(
      "hash_evt kind=fusion node=master label=%s score=%d margin=%d voters=%d window_total=%u mode=%d packets=%lu rejected=%lu\n",
      kCategoryLabels[out.label], out.score, out.margin, out.num_voters,
      static_cast<unsigned>(out.total_in_window),
      static_cast<int>(out.mode_used),
      static_cast<unsigned long>(g_packets_received),
      static_cast<unsigned long>(g_packets_rejected));
  broadcastFusion(e);

  // Camera trigger: if fusion matches the configured keyword, broadcast a
  // CameraCmdPacket. Throttled by HASH_KWS_CAMERA_TRIGGER_QUIET_MS so the
  // camera doesn't have to queue.
  if (out.label == HASH_KWS_CAMERA_TRIGGER_LABEL_IDX) {
    if ((now - g_last_camera_trigger_ms) >= HASH_KWS_CAMERA_TRIGGER_QUIET_MS ||
        g_last_camera_trigger_ms == 0) {
      sendCameraTrigger(out.label);
    }
  }
}

// ---- WiFi setup -----------------------------------------------------------

static bool g_running_in_ap_mode = false;
static uint8_t g_active_channel = HASH_KWS_ESPNOW_CHANNEL;

static IPAddress startApMode() {
  // WIFI_AP_STA (not WIFI_AP): the SoftAP runs as the dashboard / camera link
  // anchor, and STA stays radio-up (unassociated). We need STA up because
  // ensureBroadcastPeer registers the ESP-NOW broadcast peer with
  // peer.ifidx = WIFI_IF_STA — under WIFI_AP that interface is disabled and
  // esp_now_send returns ESP_ERR_ESPNOW_IF (0x306C). Symptom: master prints
  //   hash_evt kind=cam_trigger phase=tx status=fail err=0x306c
  // on every "yes" fusion. Audio receive is unaffected (recv cb fires on any
  // enabled iface), so only the camera trigger path was broken.
  WiFi.mode(WIFI_AP_STA);
  bool ok = WiFi.softAP(HASH_KWS_AP_SSID, HASH_KWS_AP_PASS, HASH_KWS_ESPNOW_CHANNEL);
  esp_wifi_set_channel(HASH_KWS_ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  IPAddress ip = WiFi.softAPIP();
  g_running_in_ap_mode = true;
  g_active_channel = HASH_KWS_ESPNOW_CHANNEL;
  // Bridge-friendly: hash_evt kind=wifi phase=ap_fallback ... (parsed by
  // hash_kws_master_demux_bridge.py + dashboard FastAPI state.json).
  Serial.printf("hash_evt kind=wifi phase=ap_fallback ssid=%s pass=%s "
                "ip=%s channel=%d ok=%d\n",
                HASH_KWS_AP_SSID, HASH_KWS_AP_PASS,
                ip.toString().c_str(), HASH_KWS_ESPNOW_CHANNEL, (int)ok);
  return ip;
}

// STA → AP fallback (Sergey's pattern, with my STA SSID/pass defines and
// the channel-mismatch WARN preserved). Emits hash_evt lines that the
// host-side bridge picks up — so even when WiFi can't be reached, plugging
// the master into USB and starting hash_kws_master_demux_bridge.py + the
// FastAPI dashboard gives the operator the same state via Serial.
static IPAddress bringUpWifi() {
#if HASH_KWS_WIFI_MODE == 1
  if (sizeof(HASH_KWS_STA_SSID) <= 1) {
    Serial.println("hash_evt kind=wifi phase=sta_skip reason=empty_ssid");
    return startApMode();
  }
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(HASH_KWS_STA_SSID, HASH_KWS_STA_PASS);
  Serial.printf("hash_evt kind=wifi phase=sta_connect ssid=%s\n", HASH_KWS_STA_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED &&
         (millis() - t0) < HASH_KWS_STA_CONNECT_TIMEOUT_MS) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    g_running_in_ap_mode = false;
    g_active_channel = WiFi.channel();
    Serial.printf("hash_evt kind=wifi phase=sta_ok ip=%s rssi=%d channel=%u\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI(),
                  g_active_channel);
    if (g_active_channel != HASH_KWS_ESPNOW_CHANNEL) {
      Serial.printf(
          "hash_evt kind=wifi phase=channel_mismatch router_channel=%u "
          "compiled_channel=%d note=inference_nodes_must_match\n",
          g_active_channel, HASH_KWS_ESPNOW_CHANNEL);
    }
    return WiFi.localIP();
  }
  Serial.println("hash_evt kind=wifi phase=sta_timeout");
#if HASH_KWS_AP_FALLBACK
  return startApMode();
#else
  Serial.println("hash_evt kind=wifi phase=offline note=ap_fallback_disabled");
  esp_wifi_set_channel(HASH_KWS_ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  return IPAddress(0, 0, 0, 0);
#endif
#else
  return startApMode();
#endif
}

// ---- WebSocket events -----------------------------------------------------

static void onWsEvent(AsyncWebSocket* /*srv*/, AsyncWebSocketClient* client,
                      AwsEventType type, void* /*arg*/, uint8_t* /*data*/, size_t /*len*/) {
  if (type == WS_EVT_CONNECT) {
    Serial.printf("[ws] client #%u connected from %s\n",
                  client->id(), client->remoteIP().toString().c_str());
    sendSnapshot(client);
  } else if (type == WS_EVT_DISCONNECT) {
    Serial.printf("[ws] client #%u disconnected\n", client->id());
  }
}

// ---- ESP-NOW init ---------------------------------------------------------

static bool bringUpEspNow() {
  if (esp_now_init() != ESP_OK) {
    Serial.println("hash_evt kind=espnow phase=init status=fail node=master");
    return false;
  }
  HashKwsEspNowRegisterRecv();   // version-specific shim in espnow_recv_glue.cpp
  ensureBroadcastPeer();
  Serial.printf("hash_evt kind=espnow phase=init status=ok node=master mac=%s "
                "broadcast_peer=%d cam_trigger_label=%s\n",
                WiFi.macAddress().c_str(), (int)g_broadcast_peer_added,
                kCategoryLabels[HASH_KWS_CAMERA_TRIGGER_LABEL_IDX]);
  return true;
}

// ---- setup / loop ---------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(50);
  pinMode(LED_RGB_PIN, OUTPUT);
  rgbForLabel(11);  // silence

  Serial.printf("hash_evt kind=boot node=master role=master_aggregator channel=%d agg_mode=%d wifi_mode=%d noise_boost=%.1f\n",
                HASH_KWS_ESPNOW_CHANNEL, HASH_KWS_AGG_MODE, HASH_KWS_WIFI_MODE,
                static_cast<double>(HASH_KWS_AGG_NOISE_BOOST));

  for (uint8_t i = 0; i < HASH_KWS_AGG_NUM_NODES; ++i) {
    g_nodes[i] = PerNodeState{};
    g_nodes[i].node_id = i + 1;
  }

  IPAddress ip = bringUpWifi();

  // mDNS so users can open the dashboard without knowing the IP.
  if (MDNS.begin(MDNS_HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("hash_evt kind=mdns hostname=%s.local\n", MDNS_HOSTNAME);
  } else {
    Serial.println("hash_evt kind=mdns status=fail");
  }
  Serial.printf("hash_evt kind=http phase=ready ip=%s url=http://%s/ mdns=http://%s.local/\n",
                ip.toString().c_str(), ip.toString().c_str(), MDNS_HOSTNAME);

  if (!bringUpEspNow()) {
    // Continue running so the dashboard is at least visible — useful for debug.
  }

  g_aggregator.reset(HASH_KWS_AGG_NUM_NODES, HASH_KWS_AGG_NUM_CLASSES, HASH_KWS_AGG_WINDOW_MS);
  // Noise classes for KWS-12: index 10 = "unknown", 11 = "silence".
  static const uint8_t kHashKwsNoiseClasses[2] = {10, 11};
  g_aggregator.setNoiseClasses(kHashKwsNoiseClasses, 2);
  g_aggregator.setNoiseBoost(static_cast<float>(HASH_KWS_AGG_NOISE_BOOST));
#if HASH_KWS_AGG_MODE == 1
  g_aggregator.setTemperatures(kHashEnsembleTemperatures);
  g_aggregator.setMode(hash_kws_ensemble::Mode::kModeTemperatureScaled);
#elif HASH_KWS_AGG_MODE == 2
  g_aggregator.setLearnedWeights(kHashEnsembleLearnedWeights);
  g_aggregator.setMode(hash_kws_ensemble::Mode::kModeLearnedWeights);
#else
  g_aggregator.setMode(hash_kws_ensemble::Mode::kModeMeanLogits);
#endif

  ws.onEvent(onWsEvent);
  server.addHandler(&ws);
  server.on("/", HTTP_GET, [](AsyncWebServerRequest* request) {
    request->send_P(200, "text/html; charset=utf-8", kHashKwsDashboardHtml);
  });
  server.on("/health", HTTP_GET, [](AsyncWebServerRequest* request) {
    DynamicJsonDocument doc(256);
    doc["ok"] = true;
    doc["uptime_s"] = millis() / 1000UL;
    doc["fusion"]   = g_fusion_total;
    doc["packets"]  = static_cast<uint32_t>(g_packets_received);
    doc["rejected"] = static_cast<uint32_t>(g_packets_rejected);
    String out; serializeJson(doc, out);
    request->send(200, "application/json", out);
  });
  server.begin();
}

void loop() {
  pollAggregator();
  ws.cleanupClients();
  delay(15);
}
