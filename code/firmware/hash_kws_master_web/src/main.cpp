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
#define WIFI_PASSWORD "Ceploplastic27"

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
#ifndef HASH_KWS_AGG_MIN_VOTERS
// Minimum number of nodes that must have voted within the window for the
// aggregator to emit a fusion decision. Default 2 keeps real ensembling.
// Set to 1 only for single-node debugging.
#define HASH_KWS_AGG_MIN_VOTERS 2
#endif
#ifndef HASH_KWS_AGG_DIAG_MS
// How often to push a `hash_evt kind=agg_diag ...` line to Serial so the
// host can see whether resolve() is firing and why.
#define HASH_KWS_AGG_DIAG_MS 3000u
#endif
#ifndef HASH_KWS_NOISE_MASK
// Bitmask of class indices the aggregator must NEVER elect as top1.
// Default skips index 10 (unknown) and 11 (silence) — they're still
// computed in the mean_logits step (so noise frames don't artificially
// boost some real command), but excluded from the top1/top2 selection.
// The plates (hero + node tiles) also suppress noise labels client-side.
// Set to 0 to disable noise masking entirely.
#define HASH_KWS_NOISE_MASK ((1u << 10) | (1u << 11))
#endif

// ─── Video node (4th MCU running an image classifier) ────────────────────────
// Reuses the same HashKwsEspNowPacket layout (same magic, same logits[12],
// same CRC) so the firmware on the video MCU can be a near-copy of the audio
// nodes. Video packets are routed to g_video state; the audio aggregator
// never sees them.
#ifndef HASH_KWS_VIDEO_NODE_ID
#define HASH_KWS_VIDEO_NODE_ID 4
#endif
#ifndef HASH_KWS_VIDEO_NUM_CLASSES
#define HASH_KWS_VIDEO_NUM_CLASSES 12
#endif
#ifndef HASH_KWS_VIDEO_STALE_MS
#define HASH_KWS_VIDEO_STALE_MS 5000u
#endif
#ifndef HASH_KWS_VIDEO_AGG_WINDOW_MS
// Sliding temporal window for the video smoother. Default 1200 ms — same as
// the audio aggregator for visual symmetry. The smoother averages logits of
// every video packet that arrived within this window.
#define HASH_KWS_VIDEO_AGG_WINDOW_MS 1200u
#endif
#ifndef HASH_KWS_VIDEO_AGG_RING_SIZE
// Upper bound on the number of video packets we keep around for averaging.
// 16 slots @ 1.2s window is plenty for cameras up to ~13 fps; older slots
// are overwritten in round-robin order regardless of age.
#define HASH_KWS_VIDEO_AGG_RING_SIZE 16
#endif
#ifndef HASH_KWS_VIDEO_AGG_DEDUP_MS
// Suppress repeated "same smoothed label" notifications within this many ms,
// like the audio fusion path. Prevents the dashboard from blinking when the
// camera holds on the same scene.
#define HASH_KWS_VIDEO_AGG_DEDUP_MS 800u
#endif

// Timing constants
#define WIFI_STA_TIMEOUT_MS  10000u
#define NODE_STALE_MS        4000u
#define FUSION_RING_SIZE     50u
#define LAT_RING_SIZE        30u
#define DEDUP_MS             800u

// Status LED.
//   - WROOM-32: GPIO2 (blue, active HIGH) is the default. Arduino's
//     LED_BUILTIN was removed from the esp32dev variant in arduino-esp32
//     3.x, so we don't rely on it — the default below covers the original
//     board and any other WROOM-32 derivative with the standard pinout.
//   - ESP32-C3 (Arduino core 3.x): LED_BUILTIN there is a `static const
//     uint8_t` pointing at the RGB-LED sentinel, NOT a real GPIO. Don't
//     try `-DLED_BUILTIN=<n>` — it collides with the variant header
//     ("expected unqualified-id before numeric constant"). Instead
//     override HASH_KWS_LED_PIN via build_flags (we ship `=8` for the
//     C3-MINI-1 env).
//   - Set HASH_KWS_LED_PIN=-1 to disable the LED entirely (modules
//     without an LED, or when the chosen GPIO is needed for something).
#ifndef HASH_KWS_LED_PIN
#define HASH_KWS_LED_PIN 2
#endif
#define HASH_KWS_LED_ENABLED ((HASH_KWS_LED_PIN) >= 0)
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
static const char* const kAggModeNames[3] = {
  "mean_logits", "temperature_scaled", "learned_weights"
};
static inline const char* AggModeName(uint8_t m) {
  return (m < 3) ? kAggModeNames[m] : "unknown";
}

// Default video class table — override at compile time by editing this array.
// Indices must match whatever the video MCU's classifier emits as `pkt.label`.
static const char* const kVideoLabels[HASH_KWS_VIDEO_NUM_CLASSES] = {
  "person", "face", "car", "bicycle", "motorbike", "dog",
  "cat", "bird", "hand", "stop_sign", "unknown", "no_obj"
};
static inline const char* VideoLabelName(uint8_t i) {
  return (i < HASH_KWS_VIDEO_NUM_CLASSES) ? kVideoLabels[i] : "unknown";
}

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
  bool     ever_seen = false;  // true after the first valid packet
};

struct FusionRecord {
  uint8_t  label;
  int16_t  score;
  int16_t  margin;
  uint8_t  voters;
  uint32_t time_ms;
};

struct VideoNodeState {
  uint8_t  label   = 255;
  uint8_t  score   = 0;
  uint8_t  margin  = 0;
  uint32_t packets = 0;
  uint32_t last_ms = 0;
  uint16_t last_seq = 0;
  uint16_t lat_ring[LAT_RING_SIZE] = {};
  uint8_t  lat_head  = 0;
  uint8_t  lat_count = 0;
  bool     ever_seen = false;
};

// One frame of logits inside the video temporal smoother. `valid` means the
// slot has ever held a packet; the `ts_ms` field still has to be checked
// against the sliding window before counting it as a voter.
struct VideoAggSlot {
  uint32_t ts_ms = 0;
  int8_t   logits[HASH_KWS_VIDEO_NUM_CLASSES] = {};
  bool     valid = false;
};

// Output of one VideoAggResolve() pass.
struct VideoAggResult {
  bool    has_decision = false;
  uint8_t label        = 0;
  int16_t score        = 0;   // mean top1 logit, Q8.8 (×256)
  int16_t margin       = 0;   // (top1 - top2) of mean logits, Q8.8
  uint8_t voter_count  = 0;   // number of frames inside the window
};

// ─────────────────────────────────────────────────────────────────────────────
//  Global state
// ─────────────────────────────────────────────────────────────────────────────

static NodeState     g_nodes[HASH_KWS_AGG_NUM_NODES];
static FusionRecord  g_fusion_ring[FUSION_RING_SIZE];
static uint8_t       g_fusion_head  = 0;
static uint8_t       g_fusion_count = 0;

// Packet counters — written from the ESP-NOW receive callback, read from
// the loop task during JSON serialization. Not `volatile`: aligned uint32_t
// stores are atomic on ESP32, and C++20 deprecates `++`/`+=` on volatile
// scalars. Tearing is irrelevant for these metrics.
static uint32_t g_pkts_received = 0;
static uint32_t g_pkts_rejected = 0;
static uint32_t g_fusion_total  = 0;

// Aggregator diagnostics — exposed in the dashboard's counters strip so
// the operator can see at a glance whether resolve() is actually firing.
static uint32_t g_resolves_attempted   = 0;  // PollAggregator() iterations
static uint32_t g_resolves_decided     = 0;  // resolve() returned has_decision
static uint32_t g_resolves_no_voters   = 0;  // 0 voters in window
static uint32_t g_resolves_low_voters  = 0;  // 1..(min-1) voters in window
static uint32_t g_dedup_skipped        = 0;  // decision suppressed by DEDUP_MS
static uint8_t  g_last_voter_count     = 0;
static uint8_t  g_last_mode_used       = HASH_KWS_AGG_MODE;
static uint32_t g_last_diag_ms         = 0;

static volatile bool g_node_dirty[HASH_KWS_AGG_NUM_NODES] = {};

// Video MCU state — independent of the audio aggregator. May never receive
// any packets if the video board isn't powered on; UI handles that gracefully.
static VideoNodeState g_video;
static volatile bool  g_video_dirty = false;

// Video temporal smoother — averages the logits of every video packet inside
// a sliding window. Single-source ensemble in time, not in models.
static VideoAggSlot g_video_agg_ring[HASH_KWS_VIDEO_AGG_RING_SIZE];
static uint8_t      g_video_agg_head        = 0;
static uint32_t     g_video_agg_total       = 0;  // packets fed into the smoother
static uint32_t     g_video_agg_decisions   = 0;  // smoothed labels emitted
static uint32_t     g_video_agg_dedup_skip  = 0;  // suppressed by dedup
static uint8_t      g_video_agg_last_label  = 255;
static uint32_t     g_video_agg_last_dec_ms = 0;
static VideoAggResult g_video_agg_last;

// Forward declarations — definitions live further down in the "Video
// temporal aggregator" section, but OnDataRecv() needs to call them.
static void VideoAggSubmit(uint32_t now_ms, const int8_t* logits);
static void VideoAggResolve(uint32_t now_ms, VideoAggResult* out);
static void PollVideoAggregator();

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

static inline bool IsAudioNode(uint8_t node) {
  return node >= 1 && node <= HASH_KWS_AGG_NUM_NODES;
}
static inline bool IsVideoNode(uint8_t node) {
  return node == HASH_KWS_VIDEO_NODE_ID;
}

static bool ValidatePacket(const HashKwsEspNowPacket& p) {
  if (p.magic != kEspNowMagic || p.version != kEspNowVersion) return false;
  if (p.node == 0)                                             return false;
  const bool audio = IsAudioNode(p.node);
  const bool video = IsVideoNode(p.node);
  if (!audio && !video)                                        return false;
  // Both audio and video reuse the same 12-slot label space in the packet,
  // but each has its own label table — so range-check against the right one.
  if (audio && p.label >= HASH_KWS_AGG_NUM_CLASSES)            return false;
  if (video && p.label >= HASH_KWS_VIDEO_NUM_CLASSES)          return false;
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
  // Liveness summary across all sources.
  const uint32_t now = millis();
  uint8_t audio_online = 0, audio_seen = 0;
  for (int i = 0; i < HASH_KWS_AGG_NUM_NODES; ++i) {
    if (g_nodes[i].ever_seen) {
      audio_seen++;
      if ((now - g_nodes[i].last_ms) < NODE_STALE_MS) audio_online++;
    }
  }
  const bool video_seen   = g_video.ever_seen;
  const bool video_online = video_seen && (now - g_video.last_ms) < HASH_KWS_VIDEO_STALE_MS;

  JsonObject c  = jMakeObj(root, "counters");
  c["fusion"]    = g_fusion_total;
  c["packets"]   = g_pkts_received;
  c["rejected"]  = g_pkts_rejected;
  c["agg_mode"]  = (int)HASH_KWS_AGG_MODE;
  c["agg_mode_name"] = AggModeName((uint8_t)HASH_KWS_AGG_MODE);
  c["min_voters"]    = (int)HASH_KWS_AGG_MIN_VOTERS;
  c["window_ms"]     = (int)HASH_KWS_AGG_WINDOW_MS;
  c["uptime_s"]      = millis() / 1000UL;
  // Connectivity summary so the UI can render a single "X/3 audio + V/1 video"
  // pill instead of the user having to scan all the per-node tiles.
  c["audio_total"]   = (int)HASH_KWS_AGG_NUM_NODES;
  c["audio_online"]  = audio_online;
  c["audio_seen"]    = audio_seen;
  c["video_online"]  = video_online;
  c["video_seen"]    = video_seen;

  JsonObject d = jMakeObj(root, "agg_diag");
  d["resolves"]    = g_resolves_attempted;
  d["decisions"]   = g_resolves_decided;
  d["no_voters"]   = g_resolves_no_voters;
  d["low_voters"]  = g_resolves_low_voters;
  d["dedup_skip"]  = g_dedup_skipped;
  d["last_voters"] = g_last_voter_count;
  d["last_mode"]   = g_last_mode_used;
  d["last_mode_name"] = AggModeName(g_last_mode_used);
}

static void FillNodeObj(JsonObject obj, int idx) {
  const NodeState& n = g_nodes[idx];
  LatStats ls = LatCompute(n);
  const uint32_t now = millis();
  const bool online = n.ever_seen && (now - n.last_ms) < NODE_STALE_MS;
  obj["node"]      = idx + 1;
  obj["variant"]   = kVariants[idx];
  if (n.label < HASH_KWS_AGG_NUM_CLASSES)
    obj["label"] = kLabels[n.label];
  else
    obj["label"] = nullptr;
  obj["score"]     = n.score;
  obj["margin"]    = n.margin;
  obj["packets"]   = n.packets;
  obj["last_ms"]   = n.last_ms;
  obj["ever_seen"] = n.ever_seen;
  obj["online"]    = online;
  obj["age_ms"]    = n.ever_seen ? (now - n.last_ms) : (uint32_t)0;
  JsonObject lat = jMakeObj(obj, "lat");
  lat["min"] = ls.min;
  lat["med"] = ls.med;
  lat["p95"] = ls.p95;
  lat["max"] = ls.max;
}

// Latency stats helper that works on the video node's ring directly (the
// existing LatCompute is bound to NodeState).
static LatStats LatComputeVideo() {
  LatStats out{0, 0, 0, 0};
  if (g_video.lat_count == 0) return out;
  uint16_t buf[LAT_RING_SIZE];
  for (uint8_t i = 0; i < g_video.lat_count; ++i) buf[i] = g_video.lat_ring[i];
  std::sort(buf, buf + g_video.lat_count);
  out.min = buf[0];
  out.med = buf[g_video.lat_count / 2];
  out.p95 = buf[(uint32_t(g_video.lat_count) * 95u) / 100u];
  out.max = buf[g_video.lat_count - 1];
  return out;
}

static void FillVideoObj(JsonObject obj) {
  const VideoNodeState& v = g_video;
  const uint32_t now = millis();
  const bool online = v.ever_seen && (now - v.last_ms) < HASH_KWS_VIDEO_STALE_MS;
  obj["node"]      = (int)HASH_KWS_VIDEO_NODE_ID;
  obj["ever_seen"] = v.ever_seen;
  obj["online"]    = online;
  obj["age_ms"]    = v.ever_seen ? (now - v.last_ms) : (uint32_t)0;
  obj["stale_ms"]  = (int)HASH_KWS_VIDEO_STALE_MS;
  if (v.label < HASH_KWS_VIDEO_NUM_CLASSES)
    obj["label"] = kVideoLabels[v.label];
  else
    obj["label"] = nullptr;
  obj["score"]    = v.score;
  obj["margin"]   = v.margin;
  obj["packets"]  = v.packets;
  obj["last_seq"] = v.last_seq;
  obj["last_ms"]  = v.last_ms;
  LatStats ls = LatComputeVideo();
  JsonObject lat = jMakeObj(obj, "lat");
  lat["min"] = ls.min;
  lat["med"] = ls.med;
  lat["p95"] = ls.p95;
  lat["max"] = ls.max;

  // Smoothed (temporally averaged) output of the video aggregator.
  // `voters` is the number of frames inside the sliding window — when it
  // drops to 0 the dashboard knows the smoother has gone cold.
  JsonObject agg = jMakeObj(obj, "agg");
  agg["window_ms"]   = (int)HASH_KWS_VIDEO_AGG_WINDOW_MS;
  agg["ring_size"]   = (int)HASH_KWS_VIDEO_AGG_RING_SIZE;
  agg["total"]       = g_video_agg_total;
  agg["decisions"]   = g_video_agg_decisions;
  agg["dedup_skip"]  = g_video_agg_dedup_skip;
  agg["voters"]      = g_video_agg_last.voter_count;
  agg["has_decision"] = g_video_agg_last.has_decision;
  if (g_video_agg_last.has_decision) {
    agg["label"]  = VideoLabelName(g_video_agg_last.label);
    agg["score"]  = g_video_agg_last.score;
    agg["margin"] = g_video_agg_last.margin;
  } else {
    agg["label"]  = nullptr;
    agg["score"]  = 0;
    agg["margin"] = 0;
  }
}

static String BuildSnapshot() {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  // Worst case: fusion ring full (50 × ~120 chars) + 3 node objects +
  // video + counters + agg_diag → ~7 KB. 8 KB gives headroom.
  DynamicJsonDocument doc(8192);
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
    fo["label"]      = kLabels[f.label];
    fo["score"]      = f.score;
    fo["margin"]     = f.margin;
    fo["voters"]     = f.voters;
    fo["time_ms"]    = f.time_ms;
    fo["mode"]       = (int)HASH_KWS_AGG_MODE;
    fo["mode_name"]  = AggModeName((uint8_t)HASH_KWS_AGG_MODE);
  }
  // Video subobject is always present in the snapshot — even when the video
  // MCU has never been seen — so the dashboard can render its placeholder
  // card from the very first WS frame.
  JsonObject vo = jMakeObj(doc, "video");
  FillVideoObj(vo);

  AddCounters(doc.as<JsonObject>());
  String out;
  serializeJson(doc, out);
  return out;
}

static String BuildNodeMsg(int idx) {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  // node + counters + agg_diag → ~700 bytes.
  DynamicJsonDocument doc(1024);
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
  // fusion fields + counters + agg_diag → ~700 bytes.
  DynamicJsonDocument doc(1024);
#endif
  doc["type"]      = "fusion";
  doc["label"]     = kLabels[f.label];
  doc["score"]     = f.score;
  doc["margin"]    = f.margin;
  doc["voters"]    = f.voters;
  doc["time_ms"]   = f.time_ms;
  doc["mode"]      = (int)g_last_mode_used;
  doc["mode_name"] = AggModeName(g_last_mode_used);
  AddCounters(doc.as<JsonObject>());
  String out;
  serializeJson(doc, out);
  return out;
}

static String BuildDiagMsg() {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  // counters + agg_diag with connectivity fields → ~600 bytes.
  DynamicJsonDocument doc(768);
#endif
  doc["type"] = "agg_diag";
  AddCounters(doc.as<JsonObject>());
  String out;
  serializeJson(doc, out);
  return out;
}

static String BuildVideoMsg() {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  // video + video.agg + lat + counters + agg_diag → ~900 bytes.
  DynamicJsonDocument doc(1280);
#endif
  doc["type"] = "video";
  JsonObject vo = jMakeObj(doc, "video");
  FillVideoObj(vo);
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
#if HASH_KWS_LED_ENABLED
  digitalWrite(HASH_KWS_LED_PIN, HIGH);
  g_led_off_at = millis() + LED_BLINK_MS;
#else
  (void)label;  // no LED on this board, silently drop
#endif
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

  g_pkts_received++;

  if (IsVideoNode(pkt.node)) {
    // Route to dedicated video state, do NOT submit to the audio aggregator.
    g_video.label   = pkt.label;
    g_video.score   = pkt.score;
    g_video.margin  = pkt.margin;
    g_video.last_ms = millis();
    g_video.last_seq = pkt.seq;
    g_video.packets++;
    g_video.ever_seen = true;
    // Latency ring (mirror of LatPush for NodeState).
    g_video.lat_ring[g_video.lat_head] = pkt.invoke_ms;
    g_video.lat_head = (g_video.lat_head + 1) % LAT_RING_SIZE;
    if (g_video.lat_count < LAT_RING_SIZE) g_video.lat_count++;
    // Feed the temporal smoother with this frame's logits.
    VideoAggSubmit(millis(), pkt.logits);
    g_video_dirty = true;
    return;
  }

  // Audio path (nodes 1..HASH_KWS_AGG_NUM_NODES).
  const int idx = pkt.node - 1;
  NodeState& n  = g_nodes[idx];
  n.label   = pkt.label;
  n.score   = pkt.score;
  n.margin  = pkt.margin;
  n.last_ms = millis();
  n.packets++;
  n.ever_seen = true;
  LatPush(n, pkt.invoke_ms);

  g_aggregator.submit(pkt.node, KindToSource(pkt.kind),
                      pkt.t_ms, millis(),
                      pkt.logits, HASH_KWS_AGG_NUM_CLASSES);
  g_node_dirty[idx] = true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Video temporal aggregator (single source, sliding 1.2 s window)
// ─────────────────────────────────────────────────────────────────────────────

static void VideoAggSubmit(uint32_t now_ms, const int8_t* logits) {
  VideoAggSlot& s = g_video_agg_ring[g_video_agg_head];
  s.ts_ms = now_ms;
  s.valid = true;
  for (uint8_t c = 0; c < HASH_KWS_VIDEO_NUM_CLASSES; ++c) s.logits[c] = logits[c];
  g_video_agg_head = (g_video_agg_head + 1) % HASH_KWS_VIDEO_AGG_RING_SIZE;
  g_video_agg_total++;
}

static void VideoAggResolve(uint32_t now_ms, VideoAggResult* out) {
  out->has_decision = false;
  out->label        = 0;
  out->score        = 0;
  out->margin       = 0;
  out->voter_count  = 0;

  float aggregated[HASH_KWS_VIDEO_NUM_CLASSES] = {0};
  uint8_t n_used = 0;
  for (uint8_t i = 0; i < HASH_KWS_VIDEO_AGG_RING_SIZE; ++i) {
    const VideoAggSlot& s = g_video_agg_ring[i];
    if (!s.valid) continue;
    if ((now_ms - s.ts_ms) > HASH_KWS_VIDEO_AGG_WINDOW_MS) continue;
    for (uint8_t c = 0; c < HASH_KWS_VIDEO_NUM_CLASSES; ++c) {
      aggregated[c] += static_cast<float>(s.logits[c]);
    }
    ++n_used;
  }
  out->voter_count = n_used;
  if (n_used == 0) return;

  const float inv = 1.f / static_cast<float>(n_used);
  uint8_t top1_idx = 0;
  float top1 = -1e9f, top2 = -1e9f;
  for (uint8_t c = 0; c < HASH_KWS_VIDEO_NUM_CLASSES; ++c) {
    const float v = aggregated[c] * inv;
    if (v > top1) {
      top2 = top1;
      top1 = v;
      top1_idx = c;
    } else if (v > top2) {
      top2 = v;
    }
  }
  if (top2 == -1e9f) top2 = top1;

  out->has_decision = true;
  out->label = top1_idx;
  const float top1_q   = top1 * 256.f;
  const float margin_q = (top1 - top2) * 256.f;
  out->score  = (int16_t)(top1_q   < -32767 ? -32767 : (top1_q   > 32767 ? 32767 : top1_q));
  out->margin = (int16_t)(margin_q < 0      ? 0      : (margin_q > 32767 ? 32767 : margin_q));
}

static void PollVideoAggregator() {
  // Always run resolve so the snapshot reflects the current sliding-window
  // state, even when no new packet arrived this tick (the window itself
  // changes as old slots fall off).
  VideoAggResult out;
  VideoAggResolve(millis(), &out);
  g_video_agg_last = out;

  if (!out.has_decision) return;

  const uint32_t now = millis();
  if (out.label == g_video_agg_last_label &&
      (now - g_video_agg_last_dec_ms) < HASH_KWS_VIDEO_AGG_DEDUP_MS) {
    g_video_agg_dedup_skip++;
    return;
  }
  g_video_agg_last_label  = out.label;
  g_video_agg_last_dec_ms = now;
  g_video_agg_decisions++;

  // Mark video as dirty so loop() pushes the new BuildVideoMsg with the
  // freshly-changed smoothed label.
  g_video_dirty = true;

  Serial.printf(
      "hash_evt kind=video_fusion node=%d label=%s score=%d margin=%d "
      "voters=%u window_ms=%u total=%lu\n",
      (int)HASH_KWS_VIDEO_NODE_ID, VideoLabelName(out.label),
      out.score, out.margin, (unsigned)out.voter_count,
      (unsigned)HASH_KWS_VIDEO_AGG_WINDOW_MS,
      (unsigned long)g_video_agg_decisions);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Aggregator poll
// ─────────────────────────────────────────────────────────────────────────────

static void PollAggregator() {
  hash_kws_ensemble::Resolved out;
  g_aggregator.resolve(millis(), &out);

  g_resolves_attempted++;
  g_last_voter_count = out.num_voters;
  g_last_mode_used   = (uint8_t)out.mode_used;

  if (!out.has_decision) {
    if (out.num_voters == 0) {
      g_resolves_no_voters++;
    } else {
      g_resolves_low_voters++;
    }
    return;
  }

  const uint32_t now = millis();
  if (out.label == g_last_decision_label &&
      (now - g_last_decision_ms) < DEDUP_MS) {
    g_dedup_skipped++;
    return;
  }

  g_last_decision_label = out.label;
  g_last_decision_ms    = now;
  g_fusion_total++;
  g_resolves_decided++;

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
      "voters=%d mode=%d mode_name=%s packets=%lu rejected=%lu\n",
      kLabels[out.label], out.score, out.margin, out.num_voters,
      (int)g_last_mode_used, AggModeName(g_last_mode_used),
      (unsigned long)g_pkts_received,
      (unsigned long)g_pkts_rejected);
}

// Periodic diagnostic: tells the host (and the dashboard) whether resolve()
// is firing, how many voters showed up last time, and what was suppressed.
// Runs every HASH_KWS_AGG_DIAG_MS regardless of fusion activity.
static void EmitAggDiagIfDue() {
  const uint32_t now = millis();
  if ((now - g_last_diag_ms) < HASH_KWS_AGG_DIAG_MS) return;
  g_last_diag_ms = now;

  Serial.printf(
      "hash_evt kind=agg_diag node=master "
      "resolves=%lu decisions=%lu no_voters=%lu low_voters=%lu "
      "dedup_skip=%lu last_voters=%u last_mode=%u last_mode_name=%s "
      "min_voters=%d window_ms=%d\n",
      (unsigned long)g_resolves_attempted,
      (unsigned long)g_resolves_decided,
      (unsigned long)g_resolves_no_voters,
      (unsigned long)g_resolves_low_voters,
      (unsigned long)g_dedup_skipped,
      (unsigned)g_last_voter_count,
      (unsigned)g_last_mode_used,
      AggModeName(g_last_mode_used),
      (int)HASH_KWS_AGG_MIN_VOTERS,
      (int)HASH_KWS_AGG_WINDOW_MS);

  ws_.textAll(BuildDiagMsg());
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

#if HASH_KWS_LED_ENABLED
  pinMode(HASH_KWS_LED_PIN, OUTPUT);
  digitalWrite(HASH_KWS_LED_PIN, HIGH);
#endif

  BringUpWifi();
  BringUpEspNow();

  if (MDNS.begin(MDNS_HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("hash_evt kind=mdns hostname=%s.local\n", MDNS_HOSTNAME);
  } else {
    Serial.println("hash_evt kind=mdns status=fail");
  }

  g_aggregator.reset(HASH_KWS_AGG_NUM_NODES, HASH_KWS_AGG_NUM_CLASSES, HASH_KWS_AGG_WINDOW_MS);
  g_aggregator.setMinVoters((uint8_t)HASH_KWS_AGG_MIN_VOTERS);
  g_aggregator.setNoiseMask((uint16_t)HASH_KWS_NOISE_MASK);
  g_last_mode_used = (uint8_t)HASH_KWS_AGG_MODE;
#if HASH_KWS_AGG_MODE == 1
  g_aggregator.setTemperatures(kHashEnsembleTemperatures);
  g_aggregator.setMode(hash_kws_ensemble::Mode::kModeTemperatureScaled);
#elif HASH_KWS_AGG_MODE == 2
  g_aggregator.setLearnedWeights(kHashEnsembleLearnedWeights);
  g_aggregator.setMode(hash_kws_ensemble::Mode::kModeLearnedWeights);
#else
  g_aggregator.setMode(hash_kws_ensemble::Mode::kModeMeanLogits);
#endif
  Serial.printf(
      "hash_evt kind=agg_init node=master mode=%d mode_name=%s "
      "min_voters=%d window_ms=%d num_nodes=%d num_classes=%d "
      "noise_mask=0x%X\n",
      (int)HASH_KWS_AGG_MODE, AggModeName((uint8_t)HASH_KWS_AGG_MODE),
      (int)HASH_KWS_AGG_MIN_VOTERS, (int)HASH_KWS_AGG_WINDOW_MS,
      HASH_KWS_AGG_NUM_NODES, HASH_KWS_AGG_NUM_CLASSES,
      (unsigned)HASH_KWS_NOISE_MASK);

  // Reset video temporal smoother (struct defaults already zero, this just
  // makes the intent explicit and lets us re-init at runtime if ever needed).
  for (uint8_t i = 0; i < HASH_KWS_VIDEO_AGG_RING_SIZE; ++i) {
    g_video_agg_ring[i].valid = false;
  }
  g_video_agg_head        = 0;
  g_video_agg_total       = 0;
  g_video_agg_decisions   = 0;
  g_video_agg_dedup_skip  = 0;
  g_video_agg_last_label  = 255;
  g_video_agg_last_dec_ms = 0;
  Serial.printf(
      "hash_evt kind=video_agg_init node=master video_node=%d "
      "window_ms=%d ring_size=%d num_classes=%d dedup_ms=%d\n",
      (int)HASH_KWS_VIDEO_NODE_ID,
      (int)HASH_KWS_VIDEO_AGG_WINDOW_MS,
      (int)HASH_KWS_VIDEO_AGG_RING_SIZE,
      (int)HASH_KWS_VIDEO_NUM_CLASSES,
      (int)HASH_KWS_VIDEO_AGG_DEDUP_MS);

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

#if HASH_KWS_LED_ENABLED
  digitalWrite(HASH_KWS_LED_PIN, LOW);
#endif
}

void loop() {
  if (g_led_off_at && millis() >= g_led_off_at) {
#if HASH_KWS_LED_ENABLED
    digitalWrite(HASH_KWS_LED_PIN, LOW);
#endif
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
  PollVideoAggregator();
  if (g_video_dirty) {
    g_video_dirty = false;
    ws_.textAll(BuildVideoMsg());
    Serial.printf(
        "hash_evt kind=video node=%d label=%s score=%u margin=%u "
        "packets=%lu seq=%u\n",
        (int)HASH_KWS_VIDEO_NODE_ID,
        (g_video.label < HASH_KWS_VIDEO_NUM_CLASSES)
            ? kVideoLabels[g_video.label] : "unknown",
        (unsigned)g_video.score, (unsigned)g_video.margin,
        (unsigned long)g_video.packets, (unsigned)g_video.last_seq);
  }
  EmitAggDiagIfDue();
  delay(20);
}
