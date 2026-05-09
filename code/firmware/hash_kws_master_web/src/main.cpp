// main.cpp — 4th ESP32: KWS ensemble aggregator + embedded web dashboard
//
// Receives HashKwsEspNowPacket from 3 inference nodes (ens_a / ens_b / ens_c)
// over ESP-NOW, aggregates logits, and serves a live dashboard at:
//   http://micro_network.local   (mDNS)
//   http://<IP>/                 (direct IP, printed on Serial after boot)
//
// File layout (relative to platformio.ini):
//   src/main.cpp                        ← this file
//   src/hash_ensemble_aggregator.cpp    ← copy from code/firmware/hash_kws_aggregator/
//   hash_ensemble_aggregator.h          ← copy from code/firmware/hash_kws_aggregator/
//   aggregator_params.h                 ← copy from code/deploy/hash_ensemble/reports/
//   web_page.h                          ← already here
//
// Board: ESP32 Dev Module (ESP32 WROOM-32)
//   CPU Frequency   : 240 MHz
//   Flash Size      : 4MB
//   Partition Scheme: Default 4MB with spiffs
//   Upload Speed    : 921600

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <ESPmDNS.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>
#include <algorithm>
#include <string.h>

#include "hash_ensemble_aggregator.h"
#include "aggregator_params.h"
#include "web_page.h"

// ─────────────────────────────────────────────────────────────────────────────
//  User configuration — edit before flashing
// ─────────────────────────────────────────────────────────────────────────────

#define WIFI_SSID     "Big_Frog_Fi"
#define WIFI_PASSWORD "Ceplosergicollection27"

// AP fallback credentials (used when STA connect fails)
#define AP_SSID     "KWS-Master"
#define AP_PASSWORD "kwsmaster1"   // min 8 chars for WPA2

// mDNS hostname → http://micro_network.local
#define MDNS_HOSTNAME "micro_network"

// ESP-NOW channel — must match all inference nodes in AP mode.
// In STA mode the channel is locked to the router; inference nodes must be
// compiled with that same channel (check `netsh wlan show interfaces`).
#ifndef HASH_KWS_ESPNOW_CHANNEL
#define HASH_KWS_ESPNOW_CHANNEL 1
#endif

// Aggregator knobs (compile-time)
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
#define HASH_KWS_AGG_MODE 0   // 0=mean_logits  1=temperature_scaled  2=learned_weights
#endif

// Timing constants
#define WIFI_STA_TIMEOUT_MS  10000u
#define NODE_STALE_MS        4000u
#define FUSION_RING_SIZE     50u
#define LAT_RING_SIZE        30u
#define DEDUP_MS             800u
// ESP32 WROOM-32: built-in blue LED on GPIO2 (active HIGH)
#ifndef LED_BUILTIN
#define LED_BUILTIN 2
#endif
#define LED_BLINK_MS 200u

// ─────────────────────────────────────────────────────────────────────────────
//  Label / mode name tables
// ─────────────────────────────────────────────────────────────────────────────

static const char* const kLabels[HASH_KWS_AGG_NUM_CLASSES] = {
  "yes","no","up","down","left","right","on","off","stop","go","unknown","silence"
};
static const char* const kVariants[HASH_KWS_AGG_NUM_NODES] = {
  "ens_a", "ens_b", "ens_c"
};

// ─────────────────────────────────────────────────────────────────────────────
//  ESP-NOW packet (must stay byte-identical with inference nodes)
// ─────────────────────────────────────────────────────────────────────────────

constexpr uint32_t kEspNowMagic   = 0x4B485731UL;  // "KHW1"
constexpr uint8_t  kEspNowVersion = 1;

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

// ─────────────────────────────────────────────────────────────────────────────
//  Runtime state structs
// ─────────────────────────────────────────────────────────────────────────────

struct LatStats { uint16_t min, med, p95, max; };

struct NodeState {
  uint8_t  label   = 255;
  uint8_t  score   = 0;
  uint8_t  margin  = 0;
  uint32_t packets = 0;
  uint32_t last_ms = 0;
  uint16_t lat_ring[LAT_RING_SIZE] = {};
  uint8_t  lat_head  = 0;
  uint8_t  lat_count = 0;
};

struct FusionRecord {
  uint8_t  label;
  int16_t  score;
  int16_t  margin;
  uint8_t  voters;
  uint32_t time_ms;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Global state
// ─────────────────────────────────────────────────────────────────────────────

static NodeState     g_nodes[HASH_KWS_AGG_NUM_NODES];
static FusionRecord  g_fusion_ring[FUSION_RING_SIZE];
static uint8_t       g_fusion_head  = 0;
static uint8_t       g_fusion_count = 0;

static volatile uint32_t g_pkts_received = 0;
static volatile uint32_t g_pkts_rejected = 0;
static uint32_t          g_fusion_total  = 0;

static volatile bool g_node_dirty[HASH_KWS_AGG_NUM_NODES] = {};

static bool     g_ap_mode    = false;
static uint32_t g_led_off_at = 0;

static hash_kws_ensemble::Aggregator g_aggregator;
static uint32_t g_last_decision_ms    = 0;
static uint8_t  g_last_decision_label = 255;

// ─────────────────────────────────────────────────────────────────────────────
//  Web server + WebSocket
// ─────────────────────────────────────────────────────────────────────────────

static AsyncWebServer server(80);
static AsyncWebSocket ws_("/ws");

// ─────────────────────────────────────────────────────────────────────────────
//  CRC-16/IBM (poly 0xA001)
// ─────────────────────────────────────────────────────────────────────────────

static uint16_t Crc16(const uint8_t* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; ++i) {
    crc ^= (uint16_t)data[i];
    for (int b = 0; b < 8; ++b)
      crc = (crc & 1u) ? (uint16_t)((crc >> 1) ^ 0xA001u) : (uint16_t)(crc >> 1);
  }
  return crc;
}

static bool ValidatePacket(const HashKwsEspNowPacket& p) {
  if (p.magic != kEspNowMagic || p.version != kEspNowVersion) return false;
  if (p.node == 0 || p.node > HASH_KWS_AGG_NUM_NODES)         return false;
  if (p.label >= HASH_KWS_AGG_NUM_CLASSES)                     return false;
  const uint16_t want = Crc16(
      reinterpret_cast<const uint8_t*>(&p), sizeof(p) - sizeof(p.crc16));
  return p.crc16 == want;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Latency ringbuffer
// ─────────────────────────────────────────────────────────────────────────────

static void LatPush(NodeState& n, uint16_t invoke_ms) {
  n.lat_ring[n.lat_head] = invoke_ms;
  n.lat_head = (n.lat_head + 1) % LAT_RING_SIZE;
  if (n.lat_count < LAT_RING_SIZE) n.lat_count++;
}

static LatStats LatCompute(const NodeState& n) {
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

// ─────────────────────────────────────────────────────────────────────────────
//  ArduinoJson v6 / v7 compatibility helpers
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
static JsonObject jMakeObj(T& parent, const char* key) {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  // `template` keyword required when calling a dependent template member function
  return parent[key].template to<JsonObject>();
#else
  return parent.createNestedObject(key);
#endif
}

template<typename T>
static JsonArray jMakeArr(T& parent, const char* key) {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  return parent[key].template to<JsonArray>();
#else
  return parent.createNestedArray(key);
#endif
}

static JsonObject jArrAddObj(JsonArray arr) {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  return arr.add<JsonObject>();
#else
  return arr.createNestedObject();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
//  JSON helpers
// ─────────────────────────────────────────────────────────────────────────────

static void AddCounters(JsonObject root) {
  JsonObject c  = jMakeObj(root, "counters");
  c["fusion"]   = g_fusion_total;
  c["packets"]  = g_pkts_received;
  c["rejected"] = g_pkts_rejected;
  c["agg_mode"] = (int)HASH_KWS_AGG_MODE;
  c["uptime_s"] = millis() / 1000UL;
}

static void FillNodeObj(JsonObject obj, int idx) {
  const NodeState& n = g_nodes[idx];
  LatStats ls = LatCompute(n);
  obj["node"]    = idx + 1;
  obj["variant"] = kVariants[idx];
  if (n.label < HASH_KWS_AGG_NUM_CLASSES)
    obj["label"] = kLabels[n.label];
  else
    obj["label"] = nullptr;
  obj["score"]   = n.score;
  obj["margin"]  = n.margin;
  obj["packets"] = n.packets;
  obj["last_ms"] = n.last_ms;
  JsonObject lat = jMakeObj(obj, "lat");
  lat["min"] = ls.min;
  lat["med"] = ls.med;
  lat["p95"] = ls.p95;
  lat["max"] = ls.max;
}

static String BuildSnapshot() {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  DynamicJsonDocument doc(4096);
#endif
  doc["type"] = "snapshot";
  JsonArray nodesArr   = jMakeArr(doc, "nodes");
  for (int i = 0; i < HASH_KWS_AGG_NUM_NODES; ++i)
    FillNodeObj(jArrAddObj(nodesArr), i);
  JsonArray fusionsArr = jMakeArr(doc, "fusion");
  for (uint8_t i = 0; i < g_fusion_count; ++i) {
    int fi = ((int)g_fusion_head - 1 - i + FUSION_RING_SIZE) % (int)FUSION_RING_SIZE;
    const FusionRecord& f = g_fusion_ring[fi];
    JsonObject fo = jArrAddObj(fusionsArr);
    fo["label"]   = kLabels[f.label];
    fo["score"]   = f.score;
    fo["margin"]  = f.margin;
    fo["voters"]  = f.voters;
    fo["time_ms"] = f.time_ms;
  }
  AddCounters(doc.as<JsonObject>());
  String out;
  serializeJson(doc, out);
  return out;
}

static String BuildNodeMsg(int idx) {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  DynamicJsonDocument doc(512);
#endif
  doc["type"] = "node";
  FillNodeObj(doc.as<JsonObject>(), idx);
  AddCounters(doc.as<JsonObject>());
  String out;
  serializeJson(doc, out);
  return out;
}

static String BuildFusionMsg(const FusionRecord& f) {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  DynamicJsonDocument doc(384);
#endif
  doc["type"]    = "fusion";
  doc["label"]   = kLabels[f.label];
  doc["score"]   = f.score;
  doc["margin"]  = f.margin;
  doc["voters"]  = f.voters;
  doc["time_ms"] = f.time_ms;
  AddCounters(doc.as<JsonObject>());
  String out;
  serializeJson(doc, out);
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  WebSocket event handler
// ─────────────────────────────────────────────────────────────────────────────

static void OnWsEvent(AsyncWebSocket* /*server*/, AsyncWebSocketClient* client,
                      AwsEventType type, void* /*arg*/,
                      uint8_t* /*data*/, size_t /*len*/) {
  if (type == WS_EVT_CONNECT)
    client->text(BuildSnapshot());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Built-in LED blink
// ─────────────────────────────────────────────────────────────────────────────

static void BlinkForLabel(uint8_t label) {
  if (label >= 10) return;
  digitalWrite(LED_BUILTIN, HIGH);
  g_led_off_at = millis() + LED_BLINK_MS;
}

// ─────────────────────────────────────────────────────────────────────────────
//  ESP-NOW receive callback
// ─────────────────────────────────────────────────────────────────────────────

static hash_kws_ensemble::SourceKind KindToSource(uint8_t kind) {
  using hash_kws_ensemble::SourceKind;
  switch (kind) {
    case 1:  return SourceKind::kEpisode;
    case 2:  return SourceKind::kEmit;
    default: return SourceKind::kInfer;
  }
}

#if ESP_ARDUINO_VERSION_MAJOR >= 3
static void OnDataRecv(const esp_now_recv_info_t* /*info*/,
                       const uint8_t* data, int len) {
#else
static void OnDataRecv(const uint8_t* /*mac*/,
                       const uint8_t* data, int len) {
#endif
  if (len != (int)sizeof(HashKwsEspNowPacket)) { g_pkts_rejected++; return; }
  HashKwsEspNowPacket pkt;
  memcpy(&pkt, data, sizeof(pkt));
  if (!ValidatePacket(pkt)) { g_pkts_rejected++; return; }

  const int idx = pkt.node - 1;
  NodeState& n  = g_nodes[idx];
  n.label   = pkt.label;
  n.score   = pkt.score;
  n.margin  = pkt.margin;
  n.last_ms = millis();
  n.packets++;
  LatPush(n, pkt.invoke_ms);
  g_pkts_received++;

  g_aggregator.submit(pkt.node, KindToSource(pkt.kind),
                      pkt.t_ms, millis(),
                      pkt.logits, HASH_KWS_AGG_NUM_CLASSES);
  g_node_dirty[idx] = true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Aggregator poll
// ─────────────────────────────────────────────────────────────────────────────

static void PollAggregator() {
  hash_kws_ensemble::Resolved out;
  g_aggregator.resolve(millis(), &out);
  if (!out.has_decision) return;

  const uint32_t now = millis();
  if (out.label == g_last_decision_label &&
      (now - g_last_decision_ms) < DEDUP_MS) return;

  g_last_decision_label = out.label;
  g_last_decision_ms    = now;
  g_fusion_total++;

  FusionRecord f;
  f.label   = out.label;
  f.score   = out.score;
  f.margin  = out.margin;
  f.voters  = out.num_voters;
  f.time_ms = now;

  g_fusion_ring[g_fusion_head] = f;
  g_fusion_head = (g_fusion_head + 1) % FUSION_RING_SIZE;
  if (g_fusion_count < FUSION_RING_SIZE) g_fusion_count++;

  ws_.textAll(BuildFusionMsg(f));
  BlinkForLabel(out.label);

  Serial.printf(
      "hash_evt kind=fusion node=master label=%s score=%d margin=%d "
      "voters=%d mode=%d packets=%lu rejected=%lu\n",
      kLabels[out.label], out.score, out.margin, out.num_voters,
      (int)HASH_KWS_AGG_MODE,
      (unsigned long)g_pkts_received,
      (unsigned long)g_pkts_rejected);
}

// ─────────────────────────────────────────────────────────────────────────────
//  WiFi: STA → AP fallback
// ─────────────────────────────────────────────────────────────────────────────

static void BringUpWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("hash_evt kind=wifi phase=sta_connect ssid=%s\n", WIFI_SSID);

  const uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - t0) < WIFI_STA_TIMEOUT_MS) {
    delay(250); Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("hash_evt kind=wifi phase=sta_ok ip=%s channel=%d\n",
                  WiFi.localIP().toString().c_str(), WiFi.channel());
  } else {
    g_ap_mode = true;
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASSWORD, HASH_KWS_ESPNOW_CHANNEL);
    Serial.printf("hash_evt kind=wifi phase=ap_fallback ssid=%s ip=%s channel=%d\n",
                  AP_SSID, WiFi.softAPIP().toString().c_str(), HASH_KWS_ESPNOW_CHANNEL);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  ESP-NOW init
// ─────────────────────────────────────────────────────────────────────────────

static void BringUpEspNow() {
  if (esp_now_init() != ESP_OK) {
    Serial.println("hash_evt kind=espnow phase=init status=fail node=master");
    return;
  }
  esp_now_register_recv_cb(OnDataRecv);
  const String mac = g_ap_mode ? WiFi.softAPmacAddress() : WiFi.macAddress();
  Serial.printf("hash_evt kind=espnow phase=init status=ok node=master mac=%s\n",
                mac.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
//  setup() / loop()
// ─────────────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(50);
  Serial.printf("hash_evt kind=boot node=master role=master_web channel=%d agg_mode=%d\n",
                HASH_KWS_ESPNOW_CHANNEL, HASH_KWS_AGG_MODE);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  BringUpWifi();
  BringUpEspNow();

  if (MDNS.begin(MDNS_HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("hash_evt kind=mdns hostname=%s.local\n", MDNS_HOSTNAME);
  } else {
    Serial.println("hash_evt kind=mdns status=fail");
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

  ws_.onEvent(OnWsEvent);
  server.addHandler(&ws_);

  server.on("/", HTTP_GET, [](AsyncWebServerRequest* req) {
    req->send(200, "text/html", kDashboardHtml);
  });
  server.onNotFound([](AsyncWebServerRequest* req) { req->redirect("/"); });
  server.begin();

  const String ip = g_ap_mode ? WiFi.softAPIP().toString() : WiFi.localIP().toString();
  Serial.printf("hash_evt kind=http phase=ready ip=%s url=http://%s/ mdns=http://%s.local/\n",
                ip.c_str(), ip.c_str(), MDNS_HOSTNAME);

  digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
  if (g_led_off_at && millis() >= g_led_off_at) {
    digitalWrite(LED_BUILTIN, LOW);
    g_led_off_at = 0;
  }

  ws_.cleanupClients();

  for (int i = 0; i < HASH_KWS_AGG_NUM_NODES; ++i) {
    if (g_node_dirty[i]) {
      g_node_dirty[i] = false;
      ws_.textAll(BuildNodeMsg(i));
    }
  }

  PollAggregator();
  delay(20);
}
