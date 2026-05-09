#pragma once

// ---- WiFi credentials (coordinator joins this network in STA mode) --------
#define WIFI_SSID       "Big_Frog_Fi"
#define WIFI_PASSWORD   "Ceploplastic27"

// WiFi channel of the router (stubs must be set to the same channel)
#define WIFI_CHANNEL    11

// ---- Web server -----------------------------------------------------------
#define WEB_PORT        80

// ---- Queue / buffer sizes -------------------------------------------------
#define QUEUE_SIZE      64      // FIFO processing queue depth
#define HISTORY_SIZE    200     // circular buffer for dashboard history
#define MAX_LOG_ENTRIES 60      // rolling system log

// ---- Node tracking --------------------------------------------------------
#define MAX_NODES       10      // max simultaneously tracked nodes
#define NODE_TIMEOUT_MS 5000    // node considered offline after this silence

// ---- WebSocket broadcast throttle -----------------------------------------
#define WS_MIN_INTERVAL_MS 50   // don't push faster than this
