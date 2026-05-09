// cam — Stub firmware for camera inference node (ESP32-WROOM-32)
// Sends fake rock/paper/scissors top-3 via ESP-NOW every ~800 ms.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "protocol.h"

#ifndef LED_PIN
#define LED_PIN 2
#endif

// ---- CONFIG ---------------------------------------------------------------
// MAC address of the coordinator ESP32-S3.  Run coordinator once and note the
// address printed on Serial, then paste it here.
static const uint8_t COORDINATOR_MAC[] = {0xB4, 0x3A, 0x45, 0x3F, 0xAC, 0xBC};

#define WIFI_CHANNEL 11

// Simulated inference delay (ms)
#define INFERENCE_DELAY_MS 800

static const char* NODE_ID   = "cam_01";
static const NodeType NODE_T = NODE_CAMERA;

// Camera model classes
static const char* CLASSES[] = {"rock", "paper", "scissors"};
static const int   NUM_CLASSES = 3;

// ---- ESP-NOW send status --------------------------------------------------
static volatile bool lastSendOk = false;

static void onDataSent(const uint8_t* mac, esp_now_send_status_t status) {
    lastSendOk = (status == ESP_NOW_SEND_SUCCESS);
}

// ---- Helpers --------------------------------------------------------------

static void generateFakeTopK(InferencePacket& pkt) {
    int winner = random(0, NUM_CLASSES);

    uint8_t scores[NUM_CLASSES];
    uint8_t winScore = random(60, 96);
    scores[winner] = winScore;

    int remaining = 100 - winScore;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        if (i == winner) continue;
        int s = (i == NUM_CLASSES - 1) ? remaining : random(0, remaining + 1);
        scores[i] = s;
        remaining -= s;
    }
    if (remaining > 0) scores[winner] += remaining;

    // Sort indices by score descending to fill top-K
    int idx[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; ++i) idx[i] = i;
    for (int i = 0; i < NUM_CLASSES - 1; ++i)
        for (int j = i + 1; j < NUM_CLASSES; ++j)
            if (scores[idx[j]] > scores[idx[i]]) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }

    for (int k = 0; k < TOP_K && k < NUM_CLASSES; ++k) {
        memset(pkt.top[k].label, 0, MAX_LABEL_LEN);
        strncpy(pkt.top[k].label, CLASSES[idx[k]], MAX_LABEL_LEN - 1);
        pkt.top[k].score = scores[idx[k]];
    }
}

// ---- Arduino entry points -------------------------------------------------

void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);          // LED ON = setup started

    Serial.begin(115200);
    delay(500);

    Serial.println("\n=== cam stub (ESP32-WROOM-32, ESP-NOW sender) ===");

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

    uint8_t mac[6];
    WiFi.macAddress(mac);
    Serial.printf("My MAC : %02X:%02X:%02X:%02X:%02X:%02X\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    Serial.printf("Channel: %d\n", WIFI_CHANNEL);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init FAILED");
        while (true) delay(1000);
    }
    esp_now_register_send_cb(onDataSent);

    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, COORDINATOR_MAC, 6);
    peer.channel = WIFI_CHANNEL;
    peer.encrypt = false;
    if (esp_now_add_peer(&peer) != ESP_OK) {
        Serial.println("Add peer FAILED");
        while (true) delay(1000);
    }

    Serial.println("Ready. Sending packets every 800 ms...\n");
    randomSeed(analogRead(0) ^ micros());
}

void loop() {
    InferencePacket pkt = {};
    strncpy(pkt.node_id, NODE_ID, MAX_NODE_ID_LEN - 1);
    pkt.node_type    = NODE_T;
    pkt.inference_ms = INFERENCE_DELAY_MS;
    pkt.uptime_ms    = millis();

    generateFakeTopK(pkt);

    digitalWrite(LED_PIN, HIGH);          // LED ON while sending
    esp_err_t err = esp_now_send(COORDINATOR_MAC, (uint8_t*)&pkt, sizeof(pkt));

    Serial.printf("[%lu] send %s  top: %s=%d%%  %s=%d%%  %s=%d%%  %s\n",
                  pkt.uptime_ms,
                  pkt.node_id,
                  pkt.top[0].label, pkt.top[0].score,
                  pkt.top[1].label, pkt.top[1].score,
                  pkt.top[2].label, pkt.top[2].score,
                  (err == ESP_OK) ? "OK" : "FAIL");

    digitalWrite(LED_PIN, LOW);           // LED OFF during delay
    delay(INFERENCE_DELAY_MS);
}
