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
#include <string.h>

#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>

#include "hash_ensemble_aggregator.h"
#include "aggregator_params.h"
#include "web_page.h"

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
};
static PerNodeState g_nodes[HASH_KWS_AGG_NUM_NODES];

struct FusionEntry {
  uint8_t  label;
  int16_t  score;
  int16_t  margin;
  uint8_t  voters;
  uint32_t time_ms;
};
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

static void appendCounters(JsonObject obj) {
  obj["fusion"]   = g_fusion_total;
  obj["packets"]  = static_cast<uint32_t>(g_packets_received);
  obj["rejected"] = static_cast<uint32_t>(g_packets_rejected);
  obj["agg_mode"] = HASH_KWS_AGG_MODE;
  obj["uptime_s"] = millis() / 1000UL;
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
}

static void appendFusion(JsonObject obj, const FusionEntry& f) {
  obj["label"]   = f.label;
  obj["score"]   = f.score;
  obj["margin"]  = f.margin;
  obj["voters"]  = f.voters;
  obj["time_ms"] = f.time_ms;
}

static void sendSnapshot(AsyncWebSocketClient* client) {
  JsonDocument doc;
  doc["type"] = "snapshot";
  JsonArray nodes_arr = doc["nodes"].to<JsonArray>();
  for (uint8_t i = 0; i < HASH_KWS_AGG_NUM_NODES; ++i) {
    if (!g_nodes[i].ever_seen) continue;
    appendNode(nodes_arr.add<JsonObject>(), g_nodes[i]);
  }
  JsonArray fusion_arr = doc["fusion"].to<JsonArray>();
  for (int i = 0; i < g_fusion_ring_count; ++i) {
    int idx = (g_fusion_ring_head - 1 - i + kFusionRingSize) % kFusionRingSize;
    appendFusion(fusion_arr.add<JsonObject>(), g_fusion_ring[idx]);
  }
  appendCounters(doc["counters"].to<JsonObject>());
  String out; serializeJson(doc, out);
  client->text(out);
}

static void broadcastNode(const PerNodeState& n) {
  JsonDocument doc;
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
  appendCounters(doc["counters"].to<JsonObject>());
  String out; serializeJson(doc, out);
  ws.textAll(out);
}

static void broadcastFusion(const FusionEntry& f) {
  JsonDocument doc;
  doc["type"]    = "fusion";
  doc["label"]   = f.label;
  doc["score"]   = f.score;
  doc["margin"]  = f.margin;
  doc["voters"]  = f.voters;
  doc["time_ms"] = f.time_ms;
  appendCounters(doc["counters"].to<JsonObject>());
  String out; serializeJson(doc, out);
  ws.textAll(out);
}

// ---- ESP-NOW callback -----------------------------------------------------

#if ESP_ARDUINO_VERSION_MAJOR >= 3
static void onDataRecv(const esp_now_recv_info_t* /*info*/,
                       const uint8_t* data, int len) {
#else
static void onDataRecv(const uint8_t* /*mac*/, const uint8_t* data, int len) {
#endif
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

#if HASH_KWS_MASTER_FORWARD_NODE_EVENTS
  const char* kind_str = (p.kind == 2) ? "emit"
                       : (p.kind == 1) ? "episode"
                       : "infer";
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
  const bool dedup = (out.label == g_last_decision_label) &&
                     ((now - g_last_decision_print_ms) < kDecisionDedupMs);
  if (dedup) return;
  g_last_decision_print_ms = now;
  g_last_decision_label = out.label;

  // Push to ring + counters
  FusionEntry e = { out.label, out.score, out.margin, out.num_voters, now };
  g_fusion_ring[g_fusion_ring_head] = e;
  g_fusion_ring_head = (g_fusion_ring_head + 1) % kFusionRingSize;
  if (g_fusion_ring_count < kFusionRingSize) g_fusion_ring_count++;
  g_fusion_total++;

  rgbForLabel(out.label);
  Serial.printf(
      "hash_evt kind=fusion node=master label=%s score=%d margin=%d voters=%d mode=%d packets=%lu rejected=%lu\n",
      kCategoryLabels[out.label], out.score, out.margin, out.num_voters,
      static_cast<int>(out.mode_used),
      static_cast<unsigned long>(g_packets_received),
      static_cast<unsigned long>(g_packets_rejected));
  broadcastFusion(e);
}

// ---- WiFi setup -----------------------------------------------------------

static bool g_running_in_ap_mode = false;
static uint8_t g_active_channel = HASH_KWS_ESPNOW_CHANNEL;

static IPAddress startApMode() {
  WiFi.mode(WIFI_AP);
  bool ok = WiFi.softAP(HASH_KWS_AP_SSID, HASH_KWS_AP_PASS, HASH_KWS_ESPNOW_CHANNEL);
  esp_wifi_set_channel(HASH_KWS_ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  IPAddress ip = WiFi.softAPIP();
  g_running_in_ap_mode = true;
  g_active_channel = HASH_KWS_ESPNOW_CHANNEL;
  Serial.printf("[wifi] AP \"%s\" pass=\"%s\" ch=%d ip=%s ok=%d\n",
                HASH_KWS_AP_SSID, HASH_KWS_AP_PASS,
                HASH_KWS_ESPNOW_CHANNEL, ip.toString().c_str(), int(ok));
  return ip;
}

static IPAddress bringUpWifi() {
#if HASH_KWS_WIFI_MODE == 1
  if (sizeof(HASH_KWS_STA_SSID) <= 1) {
    Serial.println("[wifi] STA SSID is empty — define HASH_KWS_STA_SSID before flashing.");
    Serial.println("[wifi] Falling back to AP mode for now.");
    return startApMode();
  }
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(HASH_KWS_STA_SSID, HASH_KWS_STA_PASS);
  Serial.printf("[wifi] STA connecting to \"%s\"...\n", HASH_KWS_STA_SSID);
  uint32_t deadline = millis() + HASH_KWS_STA_CONNECT_TIMEOUT_MS;
  while (WiFi.status() != WL_CONNECTED && millis() < deadline) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    g_running_in_ap_mode = false;
    g_active_channel = WiFi.channel();
    Serial.printf("[wifi] STA connected, ip=%s rssi=%d channel=%u\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI(), g_active_channel);
    if (g_active_channel != HASH_KWS_ESPNOW_CHANNEL) {
      Serial.printf(
          "[wifi] !!! WARNING: router channel %u differs from compiled HASH_KWS_ESPNOW_CHANNEL=%d.\n"
          "[wifi] !!! Inference nodes will not be heard. Reflash all 4 sketches with -DHASH_KWS_ESPNOW_CHANNEL=%u\n"
          "[wifi] !!! or move the router/AP to channel %d.\n",
          g_active_channel, HASH_KWS_ESPNOW_CHANNEL, g_active_channel, HASH_KWS_ESPNOW_CHANNEL);
    }
    // ESP-NOW automatically uses the STA channel; no explicit set needed.
    return WiFi.localIP();
  }
  Serial.println("[wifi] STA failed within timeout.");
#if HASH_KWS_AP_FALLBACK
  Serial.println("[wifi] Falling back to AP mode (HASH_KWS_AP_FALLBACK=1).");
  return startApMode();
#else
  Serial.println("[wifi] AP fallback disabled — staying offline. ESP-NOW still active.");
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
  esp_now_register_recv_cb(onDataRecv);
  Serial.printf("hash_evt kind=espnow phase=init status=ok node=master mac=%s\n",
                WiFi.macAddress().c_str());
  return true;
}

// ---- setup / loop ---------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(50);
  pinMode(LED_RGB_PIN, OUTPUT);
  rgbForLabel(11);  // silence

  Serial.printf("hash_evt kind=boot node=master role=master_aggregator channel=%d agg_mode=%d wifi_mode=%d\n",
                HASH_KWS_ESPNOW_CHANNEL, HASH_KWS_AGG_MODE, HASH_KWS_WIFI_MODE);

  for (uint8_t i = 0; i < HASH_KWS_AGG_NUM_NODES; ++i) {
    g_nodes[i] = PerNodeState{};
    g_nodes[i].node_id = i + 1;
  }

  IPAddress ip = bringUpWifi();
  Serial.printf("[http] dashboard at http://%s/\n", ip.toString().c_str());

  if (!bringUpEspNow()) {
    // Continue running so the dashboard is at least visible — useful for debug.
  }

  g_aggregator.reset(HASH_KWS_AGG_NUM_NODES, HASH_KWS_AGG_NUM_CLASSES, HASH_KWS_AGG_WINDOW_MS);
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
    JsonDocument doc;
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
