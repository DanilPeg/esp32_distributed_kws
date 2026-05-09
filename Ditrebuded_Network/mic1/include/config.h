#pragma once

// ---- mic1 node configuration ---------------------------------------------
// Logical id printed in the dashboard and stored in NodeRegistry.
static const char* NODE_ID = "mic_01";

// MAC address of the coordinator ESP32-S3.  Run coordinator firmware once,
// note the MAC printed on its serial, and update both mic1 and mic2 if it
// changes.
static const uint8_t COORDINATOR_MAC[6] = {0xF8, 0xB3, 0xB7, 0x22, 0x49, 0xA8};

// Channel must match coordinator/src/main.cpp wifiConnect() result.  In the
// reference setup the AP sits on channel 11 -- if your AP is on a different
// channel, fix WIFI_CHANNEL here AND in cam/, micro/, mic2/ to keep the
// distributed mesh aligned.
#define WIFI_CHANNEL 1

// Time between simulated inferences in stub mode and between scheduled
// inferences when running the full hash-KWS model.
#define INFERENCE_PERIOD_MS 800

// 12-class label set used by the hash-KWS model in
// Моедли/board1_model_9128_node1/code/firmware/hash_kws_runtime/.
// Order must match hash_model_settings.cpp in that bundle.
static const char* MIC_LABELS[12] = {
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "unknown", "silence"
};
static const int MIC_NUM_CLASSES = 12;

// Index of the "silence" class -- used by the stub mode so the bias toward
// silence resembles the real model's behaviour in a quiet room.
#define MIC_SILENCE_INDEX 11
