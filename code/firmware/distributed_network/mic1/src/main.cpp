// mic1 - First microphone inference node (ESP32-S3 + INMP441).
//
// Two build configurations live in this single source file, selected by
// USE_HASH_KWS_MODEL in platformio.ini:
//
//   USE_HASH_KWS_MODEL=0  - stub.  Generates fake top-3 from the 12-class
//                           hash-KWS label set every INFERENCE_PERIOD_MS.
//                           Sends them as Distibuded_Network InferencePacket
//                           (74 bytes) to the coordinator.
//
//   USE_HASH_KWS_MODEL=1  - real inference path.  Uses audio_provider /
//                           feature_provider / hash_kws_runtime brought in
//                           via platformio.ini's lib_extra_dirs from
//                           Моедли/board1_model_9128_node1/.  Defaults to
//                           USE_FAKE_MIC=1 so the model can be exercised
//                           without an INMP441 wired up.
//
// Issues fixed compared to the original micro_speech.ino reference:
//
//   1. Serial.begin() is called BEFORE the first Serial.printf().  The
//      reference sketch printed "hash_evt kind=boot ..." into a Serial that
//      had not been started yet, so the very first telemetry line was lost.
//
//   2. WIFI_CHANNEL = 11 (matches coordinator's AP), not 1.
//
//   3. The packet sent over ESP-NOW is the project-wide InferencePacket
//      (74 bytes, top-K = 3 with text labels), not the project-private
//      HashKwsEspNowPacket from the reference.  The coordinator already
//      knows how to decode this format -- no changes to coordinator/ are
//      required to onboard mic1/mic2.
//
//   4. ESP-NOW destination is the coordinator's unicast MAC, not broadcast.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <string.h>

#include "protocol.h"
#include "config.h"

#ifndef USE_HASH_KWS_MODEL
#define USE_HASH_KWS_MODEL 0
#endif

#ifndef LED_RGB_PIN
#define LED_RGB_PIN 48
#endif

static const NodeType NODE_T = NODE_MICROPHONE;

// ---- ESP-NOW send callback ------------------------------------------------
static volatile uint16_t g_tx_ok   = 0;
static volatile uint16_t g_tx_fail = 0;

#if ESP_ARDUINO_VERSION_MAJOR >= 3
static void onDataSent(const wifi_tx_info_t* info, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) ++g_tx_ok; else ++g_tx_fail;
}
#else
static void onDataSent(const uint8_t* mac, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) ++g_tx_ok; else ++g_tx_fail;
}
#endif

// ---- RGB helper -----------------------------------------------------------
static inline void rgbSet(uint8_t r, uint8_t g, uint8_t b) {
    neopixelWrite(LED_RGB_PIN, r, g, b);
}

// ===========================================================================
//  Stub inference path (USE_HASH_KWS_MODEL = 0)
// ===========================================================================
#if !USE_HASH_KWS_MODEL

static void generateFakeTopK(InferencePacket& pkt) {
    // Bias the winner: 60% of the time pick "silence" (index 11) so the
    // dashboard sees a realistic resting state in a quiet room.
    int winner;
    if (random(0, 10) < 6) {
        winner = MIC_SILENCE_INDEX;
    } else {
        winner = random(0, MIC_NUM_CLASSES);
    }

    uint8_t scores[12] = {};
    uint8_t winScore = (uint8_t)random(60, 96);
    scores[winner] = winScore;
    int remaining = 100 - winScore;
    for (int i = 0; i < MIC_NUM_CLASSES && remaining > 0; ++i) {
        if (i == winner) continue;
        int s = (i == MIC_NUM_CLASSES - 1) ? remaining : random(0, remaining + 1);
        scores[i] = (uint8_t)s;
        remaining -= s;
    }

    // Pick top-3 indices by score, descending.
    int idx[12];
    for (int i = 0; i < MIC_NUM_CLASSES; ++i) idx[i] = i;
    for (int i = 0; i < MIC_NUM_CLASSES - 1; ++i) {
        for (int j = i + 1; j < MIC_NUM_CLASSES; ++j) {
            if (scores[idx[j]] > scores[idx[i]]) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
        }
    }

    for (int k = 0; k < TOP_K; ++k) {
        memset(pkt.top[k].label, 0, MAX_LABEL_LEN);
        strncpy(pkt.top[k].label, MIC_LABELS[idx[k]], MAX_LABEL_LEN - 1);
        pkt.top[k].score = scores[idx[k]];
    }
}

#endif  // !USE_HASH_KWS_MODEL

// ===========================================================================
//  Real inference path (USE_HASH_KWS_MODEL = 1)
//
//  The headers below come from the lib_extra_dirs entry in platformio.ini
//  pointing at Моедли/board1_model_9128_node1/code/firmware/.  The shim
//  bug in hash_model_data.h is documented in MICROPHONES.md and should be
//  fixed before enabling this path -- otherwise g_hash_model resolves
//  through include recursion.
// ===========================================================================
#if USE_HASH_KWS_MODEL
extern "C" {
    // Tensor arena lives in PSRAM if available.  The runner pulls it via
    // heap_caps_malloc(MALLOC_CAP_INTERNAL) in setup_hash_runner().
}

#include "hash_kws_runner.h"
#include "hash_model_data.h"
#include "hash_recognize_commands.h"
#include "audio_provider.h"
#include "feature_provider.h"
#include "micro_model_settings.h"

#include <esp_heap_caps.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"

// Declared in audio_provider.cpp; not in the public header.
extern TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter);

namespace {
tflite::ErrorReporter*       g_err = nullptr;
FeatureProvider*             g_fp  = nullptr;
hash_kws::HashKwsRunner*     g_run = nullptr;
int8_t                       g_features[kFeatureElementCount];
int8_t*                      g_input    = nullptr;
int8_t*                      g_scratchA = nullptr;
int8_t*                      g_scratchB = nullptr;
int8_t                       g_scores[hash_kws::kCategoryCount];
int32_t                      g_prev_t   = 0;
}  // namespace

static void* allocPSRAM(size_t bytes) {
    void* p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) p = malloc(bytes);
    return p;
}

static bool setupHashRunner() {
    if (!hash_kws::g_hash_model.available) {
        Serial.println("hash_kws model unavailable - check Моедли/.../hash_model_data.cpp");
        return false;
    }
    static tflite::MicroErrorReporter er;
    g_err = &er;

    static FeatureProvider fp(kFeatureElementCount, g_features);
    g_fp = &fp;

    static hash_kws::HashKwsRunner runner(&hash_kws::g_hash_model);
    g_run = &runner;
    if (!g_run->IsReady()) return false;

    const size_t in_bytes = (size_t)hash_kws::g_hash_model.input_rows
                          * (size_t)hash_kws::g_hash_model.input_cols
                          * (size_t)hash_kws::g_hash_model.input_channels;
    const size_t scr_bytes = g_run->RequiredSingleScratchBytes();

    g_input    = (int8_t*)allocPSRAM(in_bytes);
    g_scratchA = (int8_t*)allocPSRAM(scr_bytes);
    g_scratchB = (int8_t*)allocPSRAM(scr_bytes);
    if (!g_input || !g_scratchA || !g_scratchB) {
        Serial.println("hash_kws: scratch allocation failed");
        return false;
    }

    Serial.println("hash_kws: starting I2S mic capture (BCK=15 WS=16 SD=17)...");
    if (InitAudioRecording(g_err) != kTfLiteOk) {
        Serial.println("hash_kws: InitAudioRecording FAILED - check INMP441 wiring");
        return false;
    }
    Serial.println("hash_kws: audio capture running");
    return true;
}

// Find top-3 indices in g_scores[12], descending by score.
static void findTop3(const int8_t* scores, int* idx) {
    idx[0] = idx[1] = idx[2] = 0;
    int s0 = -129, s1 = -129, s2 = -129;
    for (int i = 0; i < hash_kws::kCategoryCount; ++i) {
        int v = (int)scores[i];
        if (v > s0)      { s2 = s1; idx[2] = idx[1]; s1 = s0; idx[1] = idx[0]; s0 = v; idx[0] = i; }
        else if (v > s1) { s2 = s1; idx[2] = idx[1]; s1 = v; idx[1] = i; }
        else if (v > s2) { s2 = v; idx[2] = i; }
    }
}

#endif  // USE_HASH_KWS_MODEL

// ===========================================================================
//  Arduino entry points
// ===========================================================================

void setup() {
    // 1. Serial first, so every diagnostic line below is actually visible.
    Serial.begin(115200);
    delay(250);
    Serial.println();
    Serial.println("=== mic1 boot ===");

    // 2. Force max CPU clock (some toolchains land at 160 MHz by default).
    setCpuFrequencyMhz(240);
    Serial.printf("cpu=%lu MHz  node=%s\n",
                  (unsigned long)getCpuFrequencyMhz(), NODE_ID);

    // 3. Wi-Fi: pure STA, no AP association, lock onto coordinator channel.
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

    uint8_t mac[6];
    WiFi.macAddress(mac);
    Serial.printf("MAC: %02X:%02X:%02X:%02X:%02X:%02X  channel: %d\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], WIFI_CHANNEL);

    // 4. ESP-NOW init + register coordinator as unicast peer.
    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init FAILED");
        rgbSet(255, 0, 0);
        while (true) delay(1000);
    }
    esp_now_register_send_cb(onDataSent);

    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, COORDINATOR_MAC, 6);
    peer.channel = WIFI_CHANNEL;
    peer.encrypt = false;
    peer.ifidx   = WIFI_IF_STA;
    if (esp_now_add_peer(&peer) != ESP_OK) {
        Serial.println("Add peer FAILED");
        rgbSet(255, 0, 0);
        while (true) delay(1000);
    }

#if USE_HASH_KWS_MODEL
    if (!setupHashRunner()) {
        Serial.println("hash_kws runner setup FAILED - falling back to stub-style heartbeat");
        rgbSet(255, 64, 0);
    } else {
        Serial.println("hash_kws ready");
        rgbSet(0, 255, 0);
    }
#else
    Serial.println("mode=stub (USE_HASH_KWS_MODEL=0)");
    rgbSet(0, 64, 255);
    randomSeed((uint32_t)analogRead(0) ^ micros());
#endif

    Serial.printf("Coordinator: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  COORDINATOR_MAC[0], COORDINATOR_MAC[1], COORDINATOR_MAC[2],
                  COORDINATOR_MAC[3], COORDINATOR_MAC[4], COORDINATOR_MAC[5]);
    Serial.println("Ready.");
}

static void emitInferencePacket(uint16_t inference_ms,
                                const ClassResult* topK) {
    InferencePacket pkt = {};
    strncpy(pkt.node_id, NODE_ID, MAX_NODE_ID_LEN - 1);
    pkt.node_type    = NODE_T;
    pkt.inference_ms = inference_ms;
    pkt.uptime_ms    = millis();
    for (int k = 0; k < TOP_K; ++k) pkt.top[k] = topK[k];

    esp_err_t err = esp_now_send(COORDINATOR_MAC,
                                 (uint8_t*)&pkt, sizeof(pkt));
    Serial.printf("[%lu] send %s  top: %s=%d%%  %s=%d%%  %s=%d%%  %s\n",
                  (unsigned long)pkt.uptime_ms, pkt.node_id,
                  pkt.top[0].label, pkt.top[0].score,
                  pkt.top[1].label, pkt.top[1].score,
                  pkt.top[2].label, pkt.top[2].score,
                  (err == ESP_OK) ? "OK" : "FAIL");
}

void loop() {
#if USE_HASH_KWS_MODEL
    if (!g_run || !g_fp) {
        delay(1000);
        return;
    }
    const int32_t now_t = LatestAudioTimestamp();
    int new_slices = 0;
    if (g_fp->PopulateFeatureData(g_err, g_prev_t, now_t, &new_slices) != kTfLiteOk) {
        delay(20);
        return;
    }
    g_prev_t = now_t;
    if (new_slices == 0) { delay(20); return; }

    g_run->PrepareInputFromMicroFeatures(g_features, g_input);
    const uint32_t t0 = millis();
    if (!g_run->Invoke(g_input, g_scratchA, g_scratchB, g_scores)) {
        Serial.println("Invoke failed");
        return;
    }
    const uint32_t inference_ms = millis() - t0;

    int idx[3];
    findTop3(g_scores, idx);

    ClassResult top[TOP_K] = {};
    for (int k = 0; k < TOP_K; ++k) {
        memset(top[k].label, 0, MAX_LABEL_LEN);
        strncpy(top[k].label, hash_kws::kCategoryLabels[idx[k]], MAX_LABEL_LEN - 1);
        // hash_kws scores are int8 logits; map to 0..100 by clamping +128 then /2.55.
        int u8 = (int)g_scores[idx[k]] + 128;
        if (u8 < 0) u8 = 0;
        if (u8 > 255) u8 = 255;
        top[k].score = (uint8_t)((u8 * 100) / 255);
    }
    emitInferencePacket((uint16_t)(inference_ms > 65535 ? 65535 : inference_ms), top);
#else
    InferencePacket pkt = {};
    generateFakeTopK(pkt);
    ClassResult top[TOP_K];
    for (int k = 0; k < TOP_K; ++k) top[k] = pkt.top[k];
    emitInferencePacket(INFERENCE_PERIOD_MS, top);
    delay(INFERENCE_PERIOD_MS);
#endif
}
