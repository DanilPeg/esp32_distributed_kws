#pragma once

// ---- mic2 node configuration ---------------------------------------------
// Logical id printed in the dashboard and stored in NodeRegistry.
static const char* NODE_ID = "mic_02";

// MAC address of the coordinator ESP32-S3.  Run coordinator firmware once,
// note the MAC printed on its serial, and update both mic1 and mic2 if it
// changes.
static const uint8_t COORDINATOR_MAC[6] = {0xF8, 0xB3, 0xB7, 0x22, 0x49, 0xA8};

// Must match coordinator/src/main.cpp wifiConnect() result and the channel
// used by mic1/, cam/, micro/.
#define WIFI_CHANNEL 1

// Time between inferences in stub mode and between scheduled inferences
// when running the full hash-KWS model.
#define INFERENCE_PERIOD_MS 800

// 12-class label set used by the hash-KWS model in
// Моедли/board2_model_9261_node2/code/firmware/hash_kws_runtime/.
// Order must match hash_model_settings.cpp in that bundle.
static const char* MIC_LABELS[12] = {
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "unknown", "silence"
};
static const int MIC_NUM_CLASSES = 12;

#define MIC_SILENCE_INDEX 11
