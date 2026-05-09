// coordinator — Central coordinator for Distributed NN Network (ESP32-S3)
//
// * Joins existing WiFi (STA) for web dashboard access.
// * Receives InferencePacket structs from stub nodes via ESP-NOW.
// * Maintains FIFO processing queue + circular history buffer.
// * Broadcasts results to browsers over WebSocket in real-time.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>

#include "config.h"
#include "protocol.h"
#include "request_queue.h"
#include "node_registry.h"
#include "web_page.h"

// ---- Global objects -------------------------------------------------------

static AsyncWebServer server(WEB_PORT);
static AsyncWebSocket  ws("/ws");

static RequestQueue              reqQueue;
static CircularHistory<HISTORY_SIZE> history;
static NodeRegistry              registry;
static SystemLog                 sysLog;

static uint32_t totalProcessed   = 0;
static uint32_t latencySum       = 0;
static uint32_t latencyCount     = 0;
static uint32_t lastNodeCheckMs  = 0;
static uint32_t lastUptimePushMs = 0;

// ---- JSON helpers ---------------------------------------------------------

static void itemToJson(JsonObject obj, const QueueItem& item) {
    obj["node_id"]      = item.pkt.node_id;
    obj["node_type"]    = (int)item.pkt.node_type;
    obj["inference_ms"] = item.pkt.inference_ms;
    obj["uptime_ms"]    = item.pkt.uptime_ms;
    obj["received_at"]  = item.received_at;
    obj["processed_at"] = item.processed_at;

    JsonArray top = obj["top"].to<JsonArray>();
    for (int k = 0; k < TOP_K; ++k) {
        JsonObject c = top.add<JsonObject>();
        c["label"] = item.pkt.top[k].label;
        c["score"] = item.pkt.top[k].score;
    }
}

static String buildResultMsg(const QueueItem& item) {
    JsonDocument doc;
    doc["type"] = "result";
    JsonObject data = doc["data"].to<JsonObject>();
    itemToJson(data, item);
    String out;
    serializeJson(doc, out);
    return out;
}

static String buildLogMsg(uint32_t ts, const char* text) {
    JsonDocument doc;
    doc["type"] = "log";
    doc["data"]["t"] = ts;
    doc["data"]["m"] = text;
    String out;
    serializeJson(doc, out);
    return out;
}

static String buildQueueMsg() {
    JsonDocument doc;
    doc["type"] = "queue";
    doc["data"]["depth"] = reqQueue.count();
    String out;
    serializeJson(doc, out);
    return out;
}

static String buildNodesMsg() {
    JsonDocument doc;
    doc["type"] = "nodes";
    JsonArray arr = doc["data"]["nodes"].to<JsonArray>();
    registry.forEach([&](const NodeInfo& n) {
        JsonObject o = arr.add<JsonObject>();
        o["id"]       = n.node_id;
        o["type"]     = (int)n.node_type;
        o["online"]   = n.is_online;
        o["packets"]  = n.total_packets;
        o["last_seen"] = n.last_seen_ms;
    });
    String out;
    serializeJson(doc, out);
    return out;
}

static String buildFullState() {
    JsonDocument doc;
    doc["type"] = "state";
    JsonObject data = doc["data"].to<JsonObject>();

    data["uptime"]          = millis();
    data["ip"]              = WiFi.localIP().toString();
    data["queue_depth"]     = reqQueue.count();
    data["total_processed"] = totalProcessed;

    JsonArray nodesArr = data["nodes"].to<JsonArray>();
    registry.forEach([&](const NodeInfo& n) {
        JsonObject o = nodesArr.add<JsonObject>();
        o["id"]        = n.node_id;
        o["type"]      = (int)n.node_type;
        o["online"]    = n.is_online;
        o["packets"]   = n.total_packets;
        o["last_seen"] = n.last_seen_ms;
        o["top_label"] = n.last_top_label;
        o["top_score"] = n.last_top_score;
    });

    int hCount = min(50, history.count());
    JsonArray hArr = data["history"].to<JsonArray>();
    history.forEachNewest(hCount, [&](const QueueItem& item) {
        JsonObject o = hArr.add<JsonObject>();
        itemToJson(o, item);
    });

    int lCount = min(30, sysLog.count());
    JsonArray lArr = data["logs"].to<JsonArray>();
    sysLog.forEachNewest(lCount, [&](const SystemLog::Entry& e) {
        JsonObject o = lArr.add<JsonObject>();
        o["t"] = e.time_ms;
        o["m"] = e.text;
    });

    String out;
    serializeJson(doc, out);
    return out;
}

// ---- WebSocket events -----------------------------------------------------

static void wsBroadcast(const String& msg) {
    ws.textAll(msg);
}

static void onWsEvent(AsyncWebSocket* srv, AsyncWebSocketClient* client,
                       AwsEventType type, void* arg, uint8_t* data, size_t len) {
    if (type == WS_EVT_CONNECT) {
        Serial.printf("WS client #%u connected from %s\n",
                      client->id(), client->remoteIP().toString().c_str());
        String state = buildFullState();
        client->text(state);
    } else if (type == WS_EVT_DISCONNECT) {
        Serial.printf("WS client #%u disconnected\n", client->id());
    }
}

// ---- ESP-NOW receive callback (runs in WiFi task context) -----------------
// Arduino-ESP32 core v3.x changed the callback signature.

#if ESP_ARDUINO_VERSION_MAJOR >= 3
static void onDataRecv(const esp_now_recv_info_t* info,
                        const uint8_t* data, int len) {
    if (len != sizeof(InferencePacket)) return;
    QueueItem item = {};
    memcpy(&item.pkt, data, sizeof(InferencePacket));
    item.received_at  = millis();
    item.processed_at = 0;
    memcpy(item.sender_mac, info->src_addr, 6);
    reqQueue.pushFromISR(item);
}
#else
static void onDataRecv(const uint8_t* mac, const uint8_t* data, int len) {
    if (len != sizeof(InferencePacket)) return;
    QueueItem item = {};
    memcpy(&item.pkt, data, sizeof(InferencePacket));
    item.received_at  = millis();
    item.processed_at = 0;
    memcpy(item.sender_mac, mac, 6);
    reqQueue.pushFromISR(item);
}
#endif

// ---- WiFi setup -----------------------------------------------------------

static void wifiConnect() {
    Serial.printf("Connecting to WiFi \"%s\" ...\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 40) {
        delay(500);
        Serial.print('.');
        tries++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\nWiFi connected!  IP: %s  Channel: %d\n",
                      WiFi.localIP().toString().c_str(), WiFi.channel());
    } else {
        Serial.println("\nWiFi FAILED - web dashboard unavailable, ESP-NOW still active.");
    }
}

// ---- Web server routes ----------------------------------------------------

static void setupWebServer() {
    ws.onEvent(onWsEvent);
    server.addHandler(&ws);

    server.on("/", HTTP_GET, [](AsyncWebServerRequest* req) {
        req->send_P(200, "text/html", INDEX_HTML);
    });

    server.on("/api/state", HTTP_GET, [](AsyncWebServerRequest* req) {
        String json = buildFullState();
        req->send(200, "application/json", json);
    });

    server.begin();
    Serial.printf("Web server started on port %d\n", WEB_PORT);
}

// ---- ESP-NOW setup --------------------------------------------------------

static void setupEspNow() {
    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init FAILED");
        return;
    }
    esp_now_register_recv_cb(onDataRecv);
    Serial.println("ESP-NOW receiver initialized");

    uint8_t mac[6];
    WiFi.macAddress(mac);
    Serial.printf("Coordinator MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    Serial.printf("WiFi channel   : %d\n", WiFi.channel());
}

// ---- RGB LED helpers (WS2812 on GPIO48) -----------------------------------

#ifndef LED_RGB_PIN
#define LED_RGB_PIN 48
#endif

static void rgbSet(uint8_t r, uint8_t g, uint8_t b) {
    neopixelWrite(LED_RGB_PIN, r, g, b);
}

// ---- Arduino entry points -------------------------------------------------

void setup() {
    rgbSet(255, 0, 255);                        // RED = setup started

    Serial.begin(115200);
    delay(1000);
    Serial.println("\n========================================");
    Serial.println("  Distributed NN Network - Coordinator");
    Serial.println("========================================\n");

    rgbSet(255, 165, 0);                      // ORANGE = WiFi connecting
    reqQueue.begin();
    wifiConnect();

    if (WiFi.status() == WL_CONNECTED) {
        rgbSet(0, 0, 255);                    // BLUE = WiFi OK
    } else {
        rgbSet(255, 0, 0);                    // RED = WiFi failed
    }

    setupEspNow();
    setupWebServer();

    sysLog.add("Coordinator started");
    sysLog.addf("IP: %s  Channel: %d", WiFi.localIP().toString().c_str(), WiFi.channel());

    rgbSet(0, 255, 0);                        // GREEN = all OK, running
}

void loop() {
    // 1. Process incoming queue items
    QueueItem item;
    while (reqQueue.pop(item)) {
        item.processed_at = millis();

        history.push(item);

        bool wasKnown = false;
        registry.forEach([&](const NodeInfo& n) {
            if (memcmp(n.mac, item.sender_mac, 6) == 0) wasKnown = true;
        });

        registry.update(item.sender_mac, item.pkt);

        if (!wasKnown) {
            sysLog.addf("New node registered: %s (type %d)", item.pkt.node_id, (int)item.pkt.node_type);
            String nm = buildNodesMsg();
            wsBroadcast(nm);
            String lm = buildLogMsg(millis(), "New node registered");
            wsBroadcast(lm);
        }

        totalProcessed++;
        uint32_t qlat = item.processed_at - item.received_at;
        latencySum += qlat;
        latencyCount++;

        Serial.printf("[PROC] %s  %s=%d%%  qlat=%lu ms\n",
                      item.pkt.node_id,
                      item.pkt.top[0].label, item.pkt.top[0].score,
                      (unsigned long)qlat);

        String msg = buildResultMsg(item);
        wsBroadcast(msg);
    }

    // 2. Periodically check node online/offline status (every 1 s)
    uint32_t now = millis();
    if (now - lastNodeCheckMs > 1000) {
        lastNodeCheckMs = now;

        bool wasBefore[MAX_NODES];
        for (int i = 0; i < MAX_NODES; ++i) {
            const NodeInfo* ni = registry.node(i);
            wasBefore[i] = ni ? ni->is_online : false;
        }

        bool changed = registry.refreshOnlineStatus();
        if (changed) {
            for (int i = 0; i < MAX_NODES; ++i) {
                const NodeInfo* ni = registry.node(i);
                if (!ni) continue;
                if (wasBefore[i] && !ni->is_online) {
                    sysLog.addf("Node OFFLINE: %s", ni->node_id);
                    String lm = buildLogMsg(millis(), (String("Node OFFLINE: ") + ni->node_id).c_str());
                    wsBroadcast(lm);
                } else if (!wasBefore[i] && ni->is_online) {
                    sysLog.addf("Node ONLINE: %s", ni->node_id);
                    String lm = buildLogMsg(millis(), (String("Node ONLINE: ") + ni->node_id).c_str());
                    wsBroadcast(lm);
                }
            }
            String msg = buildNodesMsg();
            wsBroadcast(msg);
        }

        String qm = buildQueueMsg();
        wsBroadcast(qm);
    }

    // 3. Push uptime to web every 5 seconds
    if (now - lastUptimePushMs > 5000) {
        lastUptimePushMs = now;

        JsonDocument doc;
        doc["type"] = "state";
        doc["data"]["uptime"] = now;
        doc["data"]["ip"]     = WiFi.localIP().toString();
        doc["data"]["total_processed"] = totalProcessed;
        doc["data"]["queue_depth"]     = reqQueue.count();
        String out;
        serializeJson(doc, out);
        wsBroadcast(out);
    }

    // 4. Clean up disconnected WebSocket clients
    ws.cleanupClients();

    delay(1);
}
