// FRESH_MARKER: hash_kws runtime branch active, USE_HASH_KWS_RUNTIME=1, updated 2026-03-30
#include <TensorFlowLite_ESP32.h>
#include <cstdlib>
#include <cstring>
#include <limits.h>
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "recognize_commands.h"
#include "../../hash_kws_runtime/hash_kws_runner.h"
#include "../../hash_kws_runtime/hash_model_data.h"
#include "../../hash_kws_runtime/hash_recognize_commands.h"
#include <WiFi.h>
#include <esp_now.h>
#include <esp_heap_caps.h>
#include <esp_wifi.h>
#if __has_include(<esp_arduino_version.h>)
#include <esp_arduino_version.h>
#endif
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define HASH_KWS_USE_ESP_NN 1
#define HASH_KWS_ESP_NN_USE_REF_KERNEL 1

#ifndef USE_HASH_KWS_RUNTIME
#define USE_HASH_KWS_RUNTIME 1
#endif

#ifndef HASH_KWS_DEBUG_STREAM
#define HASH_KWS_DEBUG_STREAM 0
#endif

#ifndef HASH_KWS_TELEMETRY_STREAM
#define HASH_KWS_TELEMETRY_STREAM 1
#endif

#ifndef HASH_KWS_NODE_ID
#define HASH_KWS_NODE_ID 1
#endif

#ifndef HASH_KWS_USE_ESPNOW
// Default ON for the distributed demo. Override with -DHASH_KWS_USE_ESPNOW=0
// for single-node debugging.
#define HASH_KWS_USE_ESPNOW 1
#endif

#ifndef HASH_KWS_ESPNOW_CHANNEL
#define HASH_KWS_ESPNOW_CHANNEL 1
#endif

#ifndef HASH_KWS_ESPNOW_FUSION
#define HASH_KWS_ESPNOW_FUSION 1
#endif

#ifndef HASH_KWS_ESPNOW_FUSION_WINDOW_MS
#define HASH_KWS_ESPNOW_FUSION_WINDOW_MS 2000
#endif

#ifndef HASH_KWS_ESPNOW_MIN_SCORE
#define HASH_KWS_ESPNOW_MIN_SCORE 145
#endif

#ifndef HASH_KWS_ESPNOW_MIN_MARGIN
#define HASH_KWS_ESPNOW_MIN_MARGIN 12
#endif

#ifndef HASH_KWS_ESPNOW_FUSION_RESPOND
#define HASH_KWS_ESPNOW_FUSION_RESPOND 1
#endif

#ifndef ESP_ARDUINO_VERSION_MAJOR
#define ESP_ARDUINO_VERSION_MAJOR 2
#endif

#ifndef HASH_KWS_DETECTION_THRESHOLD
#define HASH_KWS_DETECTION_THRESHOLD 150
#endif

#ifndef HASH_KWS_AVERAGE_WINDOW_MS
#define HASH_KWS_AVERAGE_WINDOW_MS 1000
#endif

#ifndef HASH_KWS_SUPPRESSION_MS
#define HASH_KWS_SUPPRESSION_MS 900
#endif

#ifndef HASH_KWS_MINIMUM_COUNT
#define HASH_KWS_MINIMUM_COUNT 1
#endif

#ifndef HASH_KWS_USE_EPISODE_SCHEDULER
#define HASH_KWS_USE_EPISODE_SCHEDULER 1
#endif

#ifndef HASH_KWS_USE_DIRECT_DECODER
#define HASH_KWS_USE_DIRECT_DECODER 0
#endif

#ifndef HASH_KWS_DIRECT_MARGIN
#define HASH_KWS_DIRECT_MARGIN 24
#endif

#ifndef HASH_KWS_ACTIVITY_RECENT_SLICES
#define HASH_KWS_ACTIVITY_RECENT_SLICES 3
#endif

#ifndef HASH_KWS_ACTIVITY_SLICE_MAX_THRESHOLD
#define HASH_KWS_ACTIVITY_SLICE_MAX_THRESHOLD 24
#endif

#ifndef HASH_KWS_ACTIVITY_CONSECUTIVE_HITS
#define HASH_KWS_ACTIVITY_CONSECUTIVE_HITS 2
#endif

#ifndef HASH_KWS_EPISODE_INFER_INTERVAL_MS
#define HASH_KWS_EPISODE_INFER_INTERVAL_MS 240
#endif

#ifndef HASH_KWS_IDLE_PROBE_INTERVAL_MS
#define HASH_KWS_IDLE_PROBE_INTERVAL_MS 6000
#endif

#ifndef HASH_KWS_ENABLE_IDLE_PROBE
#define HASH_KWS_ENABLE_IDLE_PROBE 0
#endif

#ifndef HASH_KWS_EPISODE_TRAILING_SILENCE_MS
#define HASH_KWS_EPISODE_TRAILING_SILENCE_MS 1000
#endif

#ifndef HASH_KWS_EPISODE_MAX_DURATION_MS
#define HASH_KWS_EPISODE_MAX_DURATION_MS 2600
#endif

#ifndef HASH_KWS_EPISODE_MIN_INVOCATIONS
#define HASH_KWS_EPISODE_MIN_INVOCATIONS 1
#endif

#ifndef HASH_KWS_ACTIVITY_REPORT_INTERVAL_MS
#define HASH_KWS_ACTIVITY_REPORT_INTERVAL_MS 120
#endif

// When set to 1, suppress kind=activity events during pure silence. During
// silence we only emit on state change (speech on/off, episode on/off); during
// actual speech or an active episode the interval emission continues, so the
// dashboard still gets real-time telemetry when it matters. Set to 0 to
// restore the old unconditional 120 ms heartbeat.

// Compile-time detection of Espressif esp-nn (SIMD int8 NN kernels for S3).
// If present we wire esp_nn_conv_s8 / esp_nn_depthwise_conv_s8 in pass 4.
#if defined(__has_include)
#  if __has_include(<esp_nn.h>)
#    define HASH_KWS_HAS_ESP_NN 1
#  else
#    define HASH_KWS_HAS_ESP_NN 0
#  endif
#else
#  define HASH_KWS_HAS_ESP_NN 0
#endif

// Runtime fast-path switch. Defaults to ON whenever the header is reachable;
// flip to 0 to bisect drift between ESP-NN and the int-MAC fallback without
// removing the module from the build.
#ifndef HASH_KWS_USE_ESP_NN
#  define HASH_KWS_USE_ESP_NN HASH_KWS_HAS_ESP_NN
#endif

// Diagnostic: when 1, the conv chain runs through a hand-rolled reference
// int8 kernel that uses the same multiplier/shift/bias data as the esp-nn
// path. Useful to bisect "is the math wrong" vs "is the library call wrong".
#ifndef HASH_KWS_ESP_NN_USE_REF_KERNEL
#  define HASH_KWS_ESP_NN_USE_REF_KERNEL 1
#endif

#ifndef HASH_KWS_ACTIVITY_STREAM_ONLY_ACTIVE
#define HASH_KWS_ACTIVITY_STREAM_ONLY_ACTIVE 1
#endif

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
#if !USE_HASH_KWS_RUNTIME
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
#else
hash_kws::HashKwsRunner* hash_runner = nullptr;
hash_kws::HashRecognizeCommands* hash_recognizer = nullptr;
int8_t* hash_model_input_buffer = nullptr;
int8_t* hash_scratch_a = nullptr;
int8_t* hash_scratch_b = nullptr;
int8_t hash_output_scores[hash_kws::kCategoryCount];
int32_t hash_last_debug_time = -1;
int hash_last_emitted_label_index = hash_kws::kSilenceIndex;
bool hash_episode_active = false;
int32_t hash_episode_start_time = 0;
int32_t hash_episode_last_activity_time = 0;
int32_t hash_episode_last_infer_time = INT_MIN;
int32_t hash_last_emit_time = INT_MIN;
int32_t hash_episode_best_time = 0;
int hash_episode_best_label_index = hash_kws::kSilenceIndex;
int hash_episode_best_score = 0;
int hash_episode_best_margin = 0;
int hash_episode_infer_count = 0;
int hash_activity_consecutive_hits = 0;
int32_t hash_last_idle_probe_time = INT_MIN;
int32_t hash_last_activity_report_time = INT_MIN;
bool hash_last_reported_speech_now = false;
bool hash_last_reported_episode_active = false;
#if HASH_KWS_USE_ESPNOW
constexpr uint32_t kHashKwsEspNowMagic = 0x4B485731UL;  // "KHW1"
constexpr uint8_t kHashKwsEspNowVersion = 1;
constexpr uint8_t kHashKwsEspNowKindInfer = 1;
constexpr uint8_t kHashKwsEspNowKindEmit = 2;
constexpr uint8_t kHashKwsEspNowFlagSpeech = 1 << 0;
constexpr uint8_t kHashKwsEspNowFlagEpisode = 1 << 1;
constexpr uint8_t kHashKwsEspNowBroadcastAddress[6] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

struct __attribute__((packed)) HashKwsEspNowPacket {
  uint32_t magic;
  uint8_t version;
  uint8_t node;
  uint16_t seq;
  uint32_t t_ms;
  uint16_t invoke_ms;
  uint8_t kind;
  uint8_t label;
  uint8_t score;
  uint8_t margin;
  uint8_t recent_max;
  uint8_t flags;
  int8_t logits[hash_kws::kCategoryCount];
  uint16_t crc16;
};

bool hash_espnow_ready = false;
uint16_t hash_espnow_seq = 0;
volatile bool hash_peer_packet_available = false;
HashKwsEspNowPacket hash_peer_packet;
uint32_t hash_peer_rx_millis = 0;
portMUX_TYPE hash_peer_packet_mux = portMUX_INITIALIZER_UNLOCKED;
uint16_t hash_espnow_tx_ok = 0;
uint16_t hash_espnow_tx_fail = 0;
int32_t hash_last_fused_emit_time = INT_MIN;
#endif
#endif
FeatureProvider* feature_provider = nullptr;
#if !USE_HASH_KWS_RUNTIME
RecognizeCommands* recognizer = nullptr;
#endif
int32_t previous_time = 0;

#if !USE_HASH_KWS_RUNTIME
// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t* model_input_buffer = nullptr;
#endif
int8_t feature_buffer[kFeatureElementCount];

#if USE_HASH_KWS_RUNTIME
bool IsHashCommandLabel(int label_index);

#if HASH_KWS_USE_ESPNOW
uint16_t HashCrc16(const uint8_t* data, size_t len) {
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

uint16_t PacketCrc(const HashKwsEspNowPacket& packet) {
  return HashCrc16(reinterpret_cast<const uint8_t*>(&packet),
                   sizeof(HashKwsEspNowPacket) - sizeof(packet.crc16));
}

void EmitHashEspNowEvent(const char* phase,
                         const HashKwsEspNowPacket* packet,
                         const char* status) {
  if (packet == nullptr) {
    Serial.printf(
        "hash_evt kind=espnow node=%d phase=%s status=%s tx_ok=%u tx_fail=%u\n",
        HASH_KWS_NODE_ID, phase, status,
        static_cast<unsigned int>(hash_espnow_tx_ok),
        static_cast<unsigned int>(hash_espnow_tx_fail));
    return;
  }
#if HASH_KWS_TELEMETRY_STREAM
  Serial.printf(
      "hash_evt kind=espnow node=%d phase=%s status=%s peer=%d seq=%u label=%s score=%d margin=%d recent_max=%d tx_ok=%u tx_fail=%u\n",
      HASH_KWS_NODE_ID, phase, status, packet->node,
      static_cast<unsigned int>(packet->seq),
      hash_kws::kCategoryLabels[packet->label], packet->score, packet->margin,
      packet->recent_max, static_cast<unsigned int>(hash_espnow_tx_ok),
      static_cast<unsigned int>(hash_espnow_tx_fail));
#endif
}

bool ValidateHashEspNowPacket(const HashKwsEspNowPacket& packet) {
  if (packet.magic != kHashKwsEspNowMagic) {
    return false;
  }
  if (packet.version != kHashKwsEspNowVersion) {
    return false;
  }
  if (packet.node == HASH_KWS_NODE_ID) {
    return false;
  }
  if (packet.label >= hash_kws::kCategoryCount) {
    return false;
  }
  return packet.crc16 == PacketCrc(packet);
}

void RecordHashEspNowSendStatus(esp_now_send_status_t status) {
  if (status == ESP_NOW_SEND_SUCCESS) {
    ++hash_espnow_tx_ok;
  } else {
    ++hash_espnow_tx_fail;
  }
}

#if ESP_ARDUINO_VERSION_MAJOR >= 3
void OnHashEspNowSent(const esp_now_send_info_t* tx_info,
                      esp_now_send_status_t status) {
  RecordHashEspNowSendStatus(status);
}
#else
void OnHashEspNowSent(const uint8_t* mac_addr, esp_now_send_status_t status) {
  RecordHashEspNowSendStatus(status);
}
#endif

void StoreHashEspNowPacket(const uint8_t* data, int len) {
  if (len != static_cast<int>(sizeof(HashKwsEspNowPacket))) {
    return;
  }
  HashKwsEspNowPacket packet;
  std::memcpy(&packet, data, sizeof(packet));
  if (!ValidateHashEspNowPacket(packet)) {
    return;
  }
  portENTER_CRITICAL(&hash_peer_packet_mux);
  hash_peer_packet = packet;
  hash_peer_rx_millis = millis();
  hash_peer_packet_available = true;
  portEXIT_CRITICAL(&hash_peer_packet_mux);
}

#if ESP_ARDUINO_VERSION_MAJOR >= 3
void OnHashEspNowRecv(const esp_now_recv_info_t* info,
                      const uint8_t* data,
                      int len) {
  StoreHashEspNowPacket(data, len);
}
#else
void OnHashEspNowRecv(const uint8_t* mac_addr,
                      const uint8_t* data,
                      int len) {
  StoreHashEspNowPacket(data, len);
}
#endif

void SetupHashEspNow() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_wifi_set_channel(HASH_KWS_ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);

  if (esp_now_init() != ESP_OK) {
    EmitHashEspNowEvent("init", nullptr, "fail");
    hash_espnow_ready = false;
    return;
  }
  esp_now_register_send_cb(OnHashEspNowSent);
  esp_now_register_recv_cb(OnHashEspNowRecv);

  esp_now_peer_info_t peer_info = {};
  std::memcpy(peer_info.peer_addr, kHashKwsEspNowBroadcastAddress,
              sizeof(kHashKwsEspNowBroadcastAddress));
  peer_info.channel = HASH_KWS_ESPNOW_CHANNEL;
  peer_info.encrypt = false;
  peer_info.ifidx = WIFI_IF_STA;
  if (!esp_now_is_peer_exist(kHashKwsEspNowBroadcastAddress)) {
    if (esp_now_add_peer(&peer_info) != ESP_OK) {
      EmitHashEspNowEvent("peer", nullptr, "fail");
      hash_espnow_ready = false;
      return;
    }
  }
  hash_espnow_ready = true;
  EmitHashEspNowEvent("init", nullptr, "ok");
}

void BroadcastHashEspNowPacket(int32_t current_time_ms,
                               uint32_t invoke_duration_ms,
                               int recent_slice_max,
                               bool speech_now,
                               int top1_index,
                               int top1_score_u8,
                               int top2_score_u8,
                               uint8_t kind,
                               const int8_t* scores) {
  if (!hash_espnow_ready || top1_index < 0 ||
      top1_index >= hash_kws::kCategoryCount) {
    return;
  }

  HashKwsEspNowPacket packet = {};
  packet.magic = kHashKwsEspNowMagic;
  packet.version = kHashKwsEspNowVersion;
  packet.node = static_cast<uint8_t>(HASH_KWS_NODE_ID);
  packet.seq = ++hash_espnow_seq;
  packet.t_ms = static_cast<uint32_t>(current_time_ms);
  packet.invoke_ms = static_cast<uint16_t>(
      invoke_duration_ms > 65535 ? 65535 : invoke_duration_ms);
  packet.kind = kind;
  packet.label = static_cast<uint8_t>(top1_index);
  packet.score = static_cast<uint8_t>(top1_score_u8);
  const int margin = top1_score_u8 - top2_score_u8;
  packet.margin = static_cast<uint8_t>(margin < 0 ? 0 : (margin > 255 ? 255 : margin));
  packet.recent_max = static_cast<uint8_t>(
      recent_slice_max < 0 ? 0 : (recent_slice_max > 255 ? 255 : recent_slice_max));
  packet.flags = 0;
  if (speech_now) {
    packet.flags |= kHashKwsEspNowFlagSpeech;
  }
  if (hash_episode_active) {
    packet.flags |= kHashKwsEspNowFlagEpisode;
  }
  if (scores != nullptr) {
    std::memcpy(packet.logits, scores, sizeof(packet.logits));
  }
  packet.crc16 = PacketCrc(packet);
  esp_now_send(kHashKwsEspNowBroadcastAddress,
               reinterpret_cast<const uint8_t*>(&packet), sizeof(packet));
  EmitHashEspNowEvent("tx", &packet, "queued");
}

void MaybeFuseHashEspNowPeer(int32_t current_time_ms,
                             int top1_index,
                             int top1_score_u8,
                             int top2_score_u8) {
#if HASH_KWS_ESPNOW_FUSION
  if (!IsHashCommandLabel(top1_index)) {
    return;
  }
  HashKwsEspNowPacket peer;
  uint32_t peer_rx_millis = 0;
  bool peer_available = false;
  portENTER_CRITICAL(&hash_peer_packet_mux);
  peer_available = hash_peer_packet_available;
  peer = hash_peer_packet;
  peer_rx_millis = hash_peer_rx_millis;
  portEXIT_CRITICAL(&hash_peer_packet_mux);
  if (!peer_available) {
    return;
  }
  const uint32_t peer_age_ms = millis() - peer_rx_millis;
  if (peer_age_ms > HASH_KWS_ESPNOW_FUSION_WINDOW_MS) {
    return;
  }
  if (!IsHashCommandLabel(peer.label)) {
    return;
  }
  if (peer.label != top1_index) {
    return;
  }
  const int local_margin = top1_score_u8 - top2_score_u8;
  if ((top1_score_u8 < HASH_KWS_ESPNOW_MIN_SCORE) ||
      (local_margin < HASH_KWS_ESPNOW_MIN_MARGIN) ||
      (peer.score < HASH_KWS_ESPNOW_MIN_SCORE) ||
      (peer.margin < HASH_KWS_ESPNOW_MIN_MARGIN)) {
    return;
  }
  if ((hash_last_fused_emit_time != INT_MIN) &&
      ((current_time_ms - hash_last_fused_emit_time) < HASH_KWS_SUPPRESSION_MS)) {
    return;
  }
  const int fused_score = (top1_score_u8 + peer.score) / 2;
  Serial.printf(
      "hash_evt kind=fusion node=%d peer=%d label=%s local_score=%d peer_score=%d fused_score=%d local_margin=%d peer_margin=%d peer_age_ms=%lu seq=%u\n",
      HASH_KWS_NODE_ID, peer.node, hash_kws::kCategoryLabels[top1_index],
      top1_score_u8, peer.score, fused_score, local_margin, peer.margin,
      static_cast<unsigned long>(peer_age_ms),
      static_cast<unsigned int>(peer.seq));
#if HASH_KWS_ESPNOW_FUSION_RESPOND
  RespondToCommand(error_reporter, current_time_ms,
                   hash_kws::kCategoryLabels[top1_index],
                   static_cast<uint8_t>(fused_score), true);
#endif
  hash_last_fused_emit_time = current_time_ms;
  hash_last_emit_time = current_time_ms;
#endif
}
#endif

void FindHashTop3(const int8_t* scores,
                  int* top1_index,
                  int* top1_score,
                  int* top2_index,
                  int* top2_score,
                  int* top3_index,
                  int* top3_score) {
  *top1_index = 0;
  *top2_index = 0;
  *top3_index = 0;
  *top1_score = -129;
  *top2_score = -129;
  *top3_score = -129;
  for (int i = 0; i < hash_kws::kCategoryCount; ++i) {
    const int score = static_cast<int>(scores[i]);
    if (score > *top1_score) {
      *top3_score = *top2_score;
      *top3_index = *top2_index;
      *top2_score = *top1_score;
      *top2_index = *top1_index;
      *top1_score = score;
      *top1_index = i;
    } else if (score > *top2_score) {
      *top3_score = *top2_score;
      *top3_index = *top2_index;
      *top2_score = score;
      *top2_index = i;
    } else if (score > *top3_score) {
      *top3_score = score;
      *top3_index = i;
    }
  }
}

int RecentSliceMax(const int8_t* features, int recent_slices) {
  int slice_count = recent_slices;
  if (slice_count < 1) {
    slice_count = 1;
  }
  if (slice_count > kFeatureSliceCount) {
    slice_count = kFeatureSliceCount;
  }
  const int start_index = (kFeatureSliceCount - slice_count) * kFeatureSliceSize;
  int max_value = INT_MIN;
  for (int i = start_index; i < kFeatureElementCount; ++i) {
    const int value = static_cast<int>(features[i]);
    if (value > max_value) {
      max_value = value;
    }
  }
  return max_value;
}

#if HASH_KWS_TELEMETRY_STREAM
void EmitHashTelemetryReady() {
  Serial.printf(
      "hash_evt kind=ready node=%d runtime=hash_kws scheduler=%d direct_decoder=%d threshold=%d margin=%d suppression_ms=%d idle_probe_ms=%d trailing_ms=%d min_episode_invokes=%d\n",
      HASH_KWS_NODE_ID, HASH_KWS_USE_EPISODE_SCHEDULER, HASH_KWS_USE_DIRECT_DECODER,
      HASH_KWS_DETECTION_THRESHOLD, HASH_KWS_DIRECT_MARGIN,
      HASH_KWS_SUPPRESSION_MS, HASH_KWS_IDLE_PROBE_INTERVAL_MS,
      HASH_KWS_EPISODE_TRAILING_SILENCE_MS, HASH_KWS_EPISODE_MIN_INVOCATIONS);
}

void MaybeEmitHashActivity(int32_t current_time_ms,
                           int how_many_new_slices,
                           int recent_slice_max,
                           bool speech_now) {
  const bool interval_elapsed =
      ((hash_last_activity_report_time == INT_MIN) ||
       ((current_time_ms - hash_last_activity_report_time) >=
        HASH_KWS_ACTIVITY_REPORT_INTERVAL_MS));
  const bool speech_changed = (speech_now != hash_last_reported_speech_now);
  const bool episode_changed =
      (hash_episode_active != hash_last_reported_episode_active);
#if HASH_KWS_ACTIVITY_STREAM_ONLY_ACTIVE
  // Keep chatter off the wire during silence: only emit on state change.
  // When the board is actually hearing something, fall through to the
  // interval-based emission so the UI sees a live pulse.
  const bool any_active = speech_now || hash_episode_active;
  if (!any_active) {
    if (!speech_changed && !episode_changed) {
      return;
    }
  } else if (!interval_elapsed && !speech_changed && !episode_changed) {
    return;
  }
#else
  if (!interval_elapsed && !speech_changed && !episode_changed) {
    return;
  }
#endif
  Serial.printf(
      "hash_evt kind=activity node=%d t=%ld slices=%d recent_max=%d speech=%d episode=%d hits=%d\n",
      HASH_KWS_NODE_ID, static_cast<long>(current_time_ms),
      how_many_new_slices, recent_slice_max, speech_now ? 1 : 0,
      hash_episode_active ? 1 : 0,
      hash_activity_consecutive_hits);
  hash_last_activity_report_time = current_time_ms;
  hash_last_reported_speech_now = speech_now;
  hash_last_reported_episode_active = hash_episode_active;
}

void EmitHashEpisodeEvent(const char* phase,
                          int32_t current_time_ms,
                          int duration_ms,
                          int recent_slice_max,
                          int invoke_count,
                          const char* best_label,
                          int best_score,
                          int32_t peak_time_ms) {
  Serial.printf(
      "hash_evt kind=episode node=%d phase=%s t=%ld dur=%d recent_max=%d invokes=%d best=%s best_score=%d peak_t=%ld\n",
      HASH_KWS_NODE_ID, phase, static_cast<long>(current_time_ms), duration_ms,
      recent_slice_max, invoke_count, best_label, best_score,
      static_cast<long>(peak_time_ms));
}

void EmitHashInferEvent(int32_t current_time_ms,
                        int how_many_new_slices,
                        int recent_slice_max,
                        bool speech_now,
                        uint32_t invoke_duration_ms,
                        int top1_index,
                        int top1_score_u8,
                        int top2_index,
                        int top2_score_u8,
                        int top3_index,
                        int top3_score_u8) {
  Serial.printf(
      "hash_evt kind=infer node=%d t=%ld slices=%d recent_max=%d speech=%d invoke_ms=%lu top1=%s top1_score=%d top2=%s top2_score=%d top3=%s top3_score=%d\n",
      HASH_KWS_NODE_ID, static_cast<long>(current_time_ms),
      how_many_new_slices, recent_slice_max, speech_now ? 1 : 0,
      static_cast<unsigned long>(invoke_duration_ms),
      hash_kws::kCategoryLabels[top1_index], top1_score_u8,
      hash_kws::kCategoryLabels[top2_index], top2_score_u8,
      hash_kws::kCategoryLabels[top3_index], top3_score_u8);
}

void EmitHashCommandEvent(int32_t current_time_ms,
                          const char* label,
                          int score,
                          const char* mode) {
  Serial.printf("hash_evt kind=emit node=%d t=%ld label=%s score=%d mode=%s\n",
                HASH_KWS_NODE_ID, static_cast<long>(current_time_ms), label,
                score, mode);
}
#else
void EmitHashTelemetryReady() {}
void MaybeEmitHashActivity(int32_t current_time_ms,
                           int how_many_new_slices,
                           int recent_slice_max,
                           bool speech_now) {}
void EmitHashEpisodeEvent(const char* phase,
                          int32_t current_time_ms,
                          int duration_ms,
                          int recent_slice_max,
                          int invoke_count,
                          const char* best_label,
                          int best_score,
                          int32_t peak_time_ms) {}
void EmitHashInferEvent(int32_t current_time_ms,
                        int how_many_new_slices,
                        int recent_slice_max,
                        bool speech_now,
                        uint32_t invoke_duration_ms,
                        int top1_index,
                        int top1_score_u8,
                        int top2_index,
                        int top2_score_u8,
                        int top3_index,
                        int top3_score_u8) {}
void EmitHashCommandEvent(int32_t current_time_ms,
                          const char* label,
                          int score,
                          const char* mode) {}
#endif

void ResetHashEpisodeState() {
  hash_episode_active = false;
  hash_episode_start_time = 0;
  hash_episode_last_activity_time = 0;
  hash_episode_last_infer_time = INT_MIN;
  hash_episode_best_time = 0;
  hash_episode_best_label_index = hash_kws::kSilenceIndex;
  hash_episode_best_score = 0;
  hash_episode_best_margin = 0;
  hash_episode_infer_count = 0;
}

void StartHashEpisode(int32_t current_time_ms, int recent_slice_max) {
  hash_episode_active = true;
  hash_episode_start_time = current_time_ms;
  hash_episode_last_activity_time = current_time_ms;
  hash_episode_last_infer_time = INT_MIN;
  hash_episode_best_time = current_time_ms;
  hash_episode_best_label_index = hash_kws::kSilenceIndex;
  hash_episode_best_score = 0;
  hash_episode_best_margin = 0;
  hash_episode_infer_count = 0;
#if HASH_KWS_DEBUG_STREAM
  TF_LITE_REPORT_ERROR(error_reporter,
                       "hash_ep start t=%d recent_max=%d",
                       current_time_ms, recent_slice_max);
#endif
  EmitHashEpisodeEvent("start", current_time_ms, 0, recent_slice_max, 0,
                       hash_kws::kCategoryLabels[hash_episode_best_label_index],
                       hash_episode_best_score, hash_episode_best_time);
}

bool IsHashCommandLabel(int label_index) {
  return (label_index != hash_kws::kSilenceIndex) &&
         (label_index != hash_kws::kUnknownIndex);
}

bool HashSuppressionElapsed(int32_t current_time_ms) {
  return (hash_last_emit_time == INT_MIN) ||
         ((current_time_ms - hash_last_emit_time) >= HASH_KWS_SUPPRESSION_MS);
}

bool HashPeakPasses(int label_index, int top1_score_u8, int top2_score_u8) {
  return IsHashCommandLabel(label_index) &&
         (top1_score_u8 >= HASH_KWS_DETECTION_THRESHOLD) &&
         ((top1_score_u8 - top2_score_u8) >= HASH_KWS_DIRECT_MARGIN);
}

void EmitHashPeakCommand(int32_t current_time_ms,
                         int label_index,
                         int score,
                         const char* mode) {
  EmitHashCommandEvent(current_time_ms, hash_kws::kCategoryLabels[label_index],
                       score, mode);
  RespondToCommand(error_reporter, current_time_ms,
                   hash_kws::kCategoryLabels[label_index],
                   static_cast<uint8_t>(score), true);
  hash_last_emit_time = current_time_ms;
  hash_last_emitted_label_index = label_index;
}

void UpdateHashEpisodeBest(int top1_index,
                           int top1_score_u8,
                           int top2_score_u8,
                           int32_t current_time_ms) {
  if (!IsHashCommandLabel(top1_index)) {
    return;
  }
  const int margin = top1_score_u8 - top2_score_u8;
  if ((top1_score_u8 > hash_episode_best_score) ||
      ((top1_score_u8 == hash_episode_best_score) &&
       (margin > hash_episode_best_margin))) {
    hash_episode_best_score = top1_score_u8;
    hash_episode_best_margin = margin;
    hash_episode_best_label_index = top1_index;
    hash_episode_best_time = current_time_ms;
  }
}

void FinalizeHashEpisode(int32_t current_time_ms) {
  const int duration_ms = current_time_ms - hash_episode_start_time;
#if HASH_KWS_DEBUG_STREAM
  TF_LITE_REPORT_ERROR(
      error_reporter,
      "hash_ep end t=%d dur=%d invokes=%d best=%s(%d) peak_t=%d",
      current_time_ms,
      duration_ms,
      hash_episode_infer_count,
      hash_kws::kCategoryLabels[hash_episode_best_label_index],
      hash_episode_best_score,
      hash_episode_best_time);
#endif
  EmitHashEpisodeEvent(
      "end", current_time_ms, duration_ms,
      RecentSliceMax(feature_buffer, HASH_KWS_ACTIVITY_RECENT_SLICES),
      hash_episode_infer_count,
      hash_kws::kCategoryLabels[hash_episode_best_label_index],
      hash_episode_best_score, hash_episode_best_time);
  if (HashPeakPasses(hash_episode_best_label_index, hash_episode_best_score,
                     hash_episode_best_score - hash_episode_best_margin) &&
      HashSuppressionElapsed(current_time_ms)) {
    EmitHashPeakCommand(current_time_ms, hash_episode_best_label_index,
                        hash_episode_best_score, "episode");
  }
  ResetHashEpisodeState();
}

void ReportHashScores(int32_t current_time_ms,
                      int how_many_new_slices,
                      uint32_t invoke_duration_ms,
                      const int8_t* scores,
                      const int8_t* features) {
  int top1_index = 0;
  int top2_index = 0;
  int top3_index = 0;
  int top1_score = -129;
  int top2_score = -129;
  int top3_score = -129;
  FindHashTop3(scores, &top1_index, &top1_score, &top2_index, &top2_score,
               &top3_index, &top3_score);

  int feature_min = INT_MAX;
  int feature_max = INT_MIN;
  int feature_sum_abs = 0;
  for (int i = 0; i < kFeatureElementCount; ++i) {
    const int value = static_cast<int>(features[i]);
    if (value < feature_min) {
      feature_min = value;
    }
    if (value > feature_max) {
      feature_max = value;
    }
    feature_sum_abs += (value < 0) ? -value : value;
  }
  const int feature_mean_abs = feature_sum_abs / kFeatureElementCount;

  TF_LITE_REPORT_ERROR(
      error_reporter,
      "hash_dbg t=%d slices=%d invoke=%dms top1=%s(%d) top2=%s(%d) top3=%s(%d) feat[min=%d max=%d mean_abs=%d]",
      current_time_ms,
      how_many_new_slices,
      static_cast<int>(invoke_duration_ms),
      hash_kws::kCategoryLabels[top1_index],
      top1_score + 128,
      hash_kws::kCategoryLabels[top2_index],
      top2_score + 128,
      hash_kws::kCategoryLabels[top3_index],
      top3_score + 128,
      feature_min,
      feature_max,
      feature_mean_abs);
}
#endif

void* AllocateHashRuntimeBuffer(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void* ptr = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (ptr != nullptr) {
    return ptr;
  }
  ptr = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (ptr != nullptr) {
    return ptr;
  }
  return std::malloc(bytes);
}

void ReportHashMemory(const char* phase) {
  TF_LITE_REPORT_ERROR(
      error_reporter,
      "hash_mem phase=%s free_8bit=%d largest_8bit=%d free_psram=%d largest_psram=%d",
      phase,
      static_cast<int>(heap_caps_get_free_size(MALLOC_CAP_8BIT)),
      static_cast<int>(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT)),
      static_cast<int>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)),
      static_cast<int>(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
}
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Open serial FIRST so the boot diagnostics actually reach the host.
  Serial.begin(115200);
  delay(250);
  Serial.println();
  Serial.println("hash_kws sketch boot");

  // Force max Xtensa-LX7 clock; some toolchains land at 160 MHz by
  // default which triples inference time. setCpuFrequencyMhz is a
  // no-op if the firmware is already at 240.
  setCpuFrequencyMhz(240);
  const uint32_t hash_kws_boot_cpu_mhz = getCpuFrequencyMhz();
  Serial.printf("hash_evt kind=boot node=%d cpu_mhz=%u esp_nn=%d int_mac_pw=1 int_mac_dw=1 int_mac_stem=1\n",
                HASH_KWS_NODE_ID, static_cast<unsigned>(hash_kws_boot_cpu_mhz),
                HASH_KWS_HAS_ESP_NN ? 1 : 0);

#if USE_HASH_KWS_RUNTIME && HASH_KWS_USE_ESPNOW
  SetupHashEspNow();
#endif

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

#if USE_HASH_KWS_RUNTIME
  if (!hash_kws::g_hash_model.available) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "hash_kws model is not available yet. Finish training/export and upload again.");
    return;
  }

  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static hash_kws::HashKwsRunner static_hash_runner(&hash_kws::g_hash_model);
  hash_runner = &static_hash_runner;
  if (!hash_runner->IsReady()) {
    TF_LITE_REPORT_ERROR(error_reporter, "hash_kws runtime model validation failed");
    return;
  }

#if !HASH_KWS_USE_EPISODE_SCHEDULER
  static hash_kws::HashRecognizeCommands static_hash_recognizer(
      error_reporter, HASH_KWS_AVERAGE_WINDOW_MS, HASH_KWS_DETECTION_THRESHOLD,
      HASH_KWS_SUPPRESSION_MS, HASH_KWS_MINIMUM_COUNT);
  hash_recognizer = &static_hash_recognizer;
  hash_last_emit_time = INT_MIN;
  hash_last_emitted_label_index = hash_kws::kSilenceIndex;
#else
  hash_recognizer = nullptr;
  ResetHashEpisodeState();
  hash_last_emit_time = INT_MIN;
  hash_activity_consecutive_hits = 0;
#endif

  const size_t input_bytes =
      static_cast<size_t>(hash_kws::g_hash_model.input_rows) *
      static_cast<size_t>(hash_kws::g_hash_model.input_cols) *
      static_cast<size_t>(hash_kws::g_hash_model.input_channels);
  const size_t scratch_bytes = hash_runner->RequiredSingleScratchBytes();

  ReportHashMemory("before_hash_alloc");
  hash_model_input_buffer =
      static_cast<int8_t*>(AllocateHashRuntimeBuffer(input_bytes));
  hash_scratch_a =
      static_cast<int8_t*>(AllocateHashRuntimeBuffer(scratch_bytes));
  hash_scratch_b =
      static_cast<int8_t*>(AllocateHashRuntimeBuffer(scratch_bytes));
  if ((hash_model_input_buffer == nullptr) || (hash_scratch_a == nullptr) ||
      (hash_scratch_b == nullptr)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "hash_kws allocation failed: input=%d scratch=%d",
                         static_cast<int>(input_bytes),
                         static_cast<int>(scratch_bytes));
    return;
  }
  ReportHashMemory("after_hash_alloc");

  // Pass 4: pre-materialise NHWC int8 weights + per-channel multiplier/shift/bias
  // for the ESP-NN SIMD kernels. Returns false if the header is unreachable, the
  // model contains a layer the fast path doesn't support, or any allocation
  // fails — in all those cases the runner stays on the int-MAC fallback.
  bool hash_kws_esp_nn_active = false;
#if HASH_KWS_USE_ESP_NN
  hash_kws_esp_nn_active = hash_runner->Prepare();
#endif
  Serial.printf("hash_evt kind=boot_post node=%d esp_nn_active=%d ref_kernel=%d\n",
                HASH_KWS_NODE_ID, hash_kws_esp_nn_active ? 1 : 0,
                HASH_KWS_ESP_NN_USE_REF_KERNEL ? 1 : 0);

  TF_LITE_REPORT_ERROR(
      error_reporter,
      "hash_kws ready: classes=%d input=%dx%d scratch_total=%d bytes esp_nn_active=%d",
      hash_runner->num_classes(), hash_kws::g_hash_model.input_rows,
      hash_kws::g_hash_model.input_cols,
      static_cast<int>(hash_runner->RequiredScratchArenaBytes()),
      hash_kws_esp_nn_active ? 1 : 0);
  EmitHashTelemetryReady();

  previous_time = 0;
  return;
#else
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
#endif
}

// The name of this function is important for Arduino compatibility.
void loop() {
#if USE_HASH_KWS_RUNTIME
  if ((feature_provider == nullptr) || (hash_runner == nullptr) ||
      (hash_model_input_buffer == nullptr) ||
      (hash_scratch_a == nullptr) || (hash_scratch_b == nullptr)) {
    return;
  }
#if !HASH_KWS_USE_EPISODE_SCHEDULER
  if (hash_recognizer == nullptr) {
    return;
  }
#endif
#endif
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

#if USE_HASH_KWS_RUNTIME
  const int recent_slice_max =
      RecentSliceMax(feature_buffer, HASH_KWS_ACTIVITY_RECENT_SLICES);
  const bool speech_now =
      (recent_slice_max >= HASH_KWS_ACTIVITY_SLICE_MAX_THRESHOLD);
  if (speech_now) {
    hash_activity_consecutive_hits += 1;
  } else {
    hash_activity_consecutive_hits = 0;
  }
  MaybeEmitHashActivity(current_time, how_many_new_slices, recent_slice_max,
                        speech_now);

#if HASH_KWS_USE_EPISODE_SCHEDULER
  if (hash_episode_active && speech_now) {
    hash_episode_last_activity_time = current_time;
  }

  bool should_invoke = false;
  bool idle_probe_invoke = false;
  if (!hash_episode_active) {
    const bool suppression_elapsed = HashSuppressionElapsed(current_time);
    const bool idle_probe_elapsed =
        ((hash_last_idle_probe_time == INT_MIN) ||
         ((current_time - hash_last_idle_probe_time) >=
          HASH_KWS_IDLE_PROBE_INTERVAL_MS));
    if (speech_now &&
        (hash_activity_consecutive_hits >= HASH_KWS_ACTIVITY_CONSECUTIVE_HITS) &&
        suppression_elapsed) {
      StartHashEpisode(current_time, recent_slice_max);
      should_invoke = true;
      hash_activity_consecutive_hits = 0;
    } else if (HASH_KWS_ENABLE_IDLE_PROBE && idle_probe_elapsed) {
      should_invoke = true;
      idle_probe_invoke = true;
      hash_last_idle_probe_time = current_time;
    }
  } else {
    const bool within_trailing_silence =
        ((current_time - hash_episode_last_activity_time) <=
         HASH_KWS_EPISODE_TRAILING_SILENCE_MS);
    const bool episode_timed_out =
        ((current_time - hash_episode_start_time) >= HASH_KWS_EPISODE_MAX_DURATION_MS);
    const bool needs_more_episode_invokes =
        (hash_episode_infer_count < HASH_KWS_EPISODE_MIN_INVOCATIONS);
    const bool infer_interval_elapsed =
        ((hash_episode_last_infer_time == INT_MIN) ||
         ((current_time - hash_episode_last_infer_time) >=
          HASH_KWS_EPISODE_INFER_INTERVAL_MS));
    if ((speech_now || within_trailing_silence || needs_more_episode_invokes) &&
        infer_interval_elapsed && !episode_timed_out) {
      should_invoke = true;
    } else if ((!within_trailing_silence && !needs_more_episode_invokes) ||
               episode_timed_out) {
      FinalizeHashEpisode(current_time);
      return;
    }
  }

  if (!should_invoke) {
    return;
  }
#endif

  hash_runner->PrepareInputFromMicroFeatures(feature_buffer,
                                             hash_model_input_buffer);
  const uint32_t invoke_started_ms = millis();
  if (!hash_runner->Invoke(hash_model_input_buffer, hash_scratch_a,
                           hash_scratch_b, hash_output_scores)) {
    TF_LITE_REPORT_ERROR(error_reporter, "hash_kws Invoke failed");
    return;
  }
  const uint32_t invoke_duration_ms = millis() - invoke_started_ms;
  hash_last_idle_probe_time = current_time;
  int top1_index = 0;
  int top2_index = 0;
  int top3_index = 0;
  int top1_score = -129;
  int top2_score = -129;
  int top3_score = -129;
  FindHashTop3(hash_output_scores, &top1_index, &top1_score, &top2_index,
               &top2_score, &top3_index, &top3_score);
  const int top1_score_u8 = top1_score + 128;
  const int top2_score_u8 = top2_score + 128;
  const int top3_score_u8 = top3_score + 128;
  EmitHashInferEvent(current_time, how_many_new_slices, recent_slice_max,
                     speech_now, invoke_duration_ms, top1_index,
                     top1_score_u8, top2_index, top2_score_u8, top3_index,
                     top3_score_u8);
#if HASH_KWS_USE_ESPNOW
  BroadcastHashEspNowPacket(current_time, invoke_duration_ms, recent_slice_max,
                            speech_now, top1_index, top1_score_u8,
                            top2_score_u8, kHashKwsEspNowKindInfer,
                            hash_output_scores);
  MaybeFuseHashEspNowPeer(current_time, top1_index, top1_score_u8,
                          top2_score_u8);
#endif

#if HASH_KWS_USE_EPISODE_SCHEDULER
  hash_episode_last_infer_time = current_time;
  if (hash_episode_active) {
    hash_episode_infer_count += 1;
    UpdateHashEpisodeBest(top1_index, top1_score_u8, top2_score_u8,
                          current_time);
  } else if (idle_probe_invoke &&
             HashPeakPasses(top1_index, top1_score_u8, top2_score_u8) &&
             HashSuppressionElapsed(current_time)) {
    EmitHashPeakCommand(current_time, top1_index, top1_score_u8, "idle_probe");
  }
#else
#if HASH_KWS_USE_DIRECT_DECODER
#else
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = hash_recognizer->ProcessLatestResults(
      hash_output_scores, hash_kws::kCategoryCount, current_time,
      &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "HashRecognizeCommands::ProcessLatestResults() failed");
    return;
  }
#endif
#endif
#if HASH_KWS_DEBUG_STREAM
  if ((hash_last_debug_time < 0) || ((current_time - hash_last_debug_time) >= 1000)) {
    ReportHashScores(current_time, how_many_new_slices, invoke_duration_ms,
                     hash_output_scores, feature_buffer);
    hash_last_debug_time = current_time;
  }
#endif
#if HASH_KWS_USE_EPISODE_SCHEDULER
  const bool episode_needs_more_invokes =
      hash_episode_active &&
      (hash_episode_infer_count < HASH_KWS_EPISODE_MIN_INVOCATIONS);
  const bool episode_should_finalize =
      hash_episode_active && !episode_needs_more_invokes &&
      ((((current_time - hash_episode_last_activity_time) >
         HASH_KWS_EPISODE_TRAILING_SILENCE_MS)) ||
       ((current_time - hash_episode_start_time) >=
        HASH_KWS_EPISODE_MAX_DURATION_MS));
  if (episode_should_finalize) {
    FinalizeHashEpisode(current_time);
  }
  return;
#else
#if HASH_KWS_USE_DIRECT_DECODER
  const bool valid_label =
      (top1_index != hash_kws::kSilenceIndex) &&
      (top1_index != hash_kws::kUnknownIndex);
  const bool above_threshold =
      (top1_score_u8 >= HASH_KWS_DETECTION_THRESHOLD);
  const bool separated =
      ((top1_score_u8 - top2_score_u8) >= HASH_KWS_DIRECT_MARGIN);
  const bool suppression_elapsed =
      ((hash_last_emit_time == INT_MIN) ||
       ((current_time - hash_last_emit_time) >= HASH_KWS_SUPPRESSION_MS));
  const bool label_changed =
      (top1_index != hash_last_emitted_label_index);

  if (valid_label && above_threshold && separated &&
      (suppression_elapsed || label_changed)) {
    EmitHashCommandEvent(current_time, hash_kws::kCategoryLabels[top1_index],
                         top1_score_u8, "direct");
    RespondToCommand(error_reporter, current_time,
                     hash_kws::kCategoryLabels[top1_index],
                     static_cast<uint8_t>(top1_score_u8), true);
    hash_last_emit_time = current_time;
    hash_last_emitted_label_index = top1_index;
  }
  return;
#else
  const bool should_report_command =
      is_new_command && (found_command != nullptr) &&
      (std::strcmp(found_command, "silence") != 0) &&
      (std::strcmp(found_command, "unknown") != 0);
  if (should_report_command) {
    EmitHashCommandEvent(current_time, found_command, score, "recognizer");
  }
  RespondToCommand(error_reporter, current_time, found_command, score,
                   should_report_command);
  return;
#endif
#endif
#else
  // Legacy micro_speech path disabled in this build; keep a syntactically
  // valid stub so the preprocessor is happy when USE_HASH_KWS_RUNTIME=0.
  (void)current_time;
  (void)how_many_new_slices;
#endif  // USE_HASH_KWS_RUNTIME
}

