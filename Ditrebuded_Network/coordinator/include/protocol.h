#pragma once

#include <stdint.h>

// ---- Distributed NN Network – shared ESP-NOW packet format ----
// Keep identical across cam/, micro/, coordinator/.

#define MAX_LABEL_LEN   16
#define MAX_NODE_ID_LEN 16
#define TOP_K           3

enum NodeType : uint8_t {
    NODE_CAMERA     = 1,
    NODE_MICROPHONE = 2,
    NODE_GENERIC    = 3
};

struct __attribute__((packed)) ClassResult {
    char    label[MAX_LABEL_LEN];   // null-terminated class name
    uint8_t score;                  // 0-100 %
};

struct __attribute__((packed)) InferencePacket {
    char         node_id[MAX_NODE_ID_LEN];
    NodeType     node_type;
    uint16_t     inference_ms;      // time the stub "spent" on inference
    ClassResult  top[TOP_K];        // top-K classes sorted by score desc
    uint32_t     uptime_ms;         // sender's millis() at send time
};
