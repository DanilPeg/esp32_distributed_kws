// espnow_recv_glue.cpp
//
// ESP-NOW receive callback shim. Lives in a .cpp (NOT .ino) on purpose:
// PlatformIO's Arduino-mode preprocessor auto-generates function prototypes
// for everything in the .ino, regardless of #if/#endif guards, which makes
// a version-conditional callback signature fail to compile on platforms
// where the older type doesn't exist (esp_now_recv_info_t is missing in
// ESP-IDF 4.x). Putting the shim in a .cpp side-steps the auto-prototype
// machinery — only the #if branch we actually want is even seen by the
// compiler.
//
// The .ino exposes a single C-linkage hook `HashKwsHandleRecvPacket` with
// the legacy signature (uint8_t* mac, uint8_t* data, int len). The shim
// here picks the right ESP-NOW callback shape for the active ESP-IDF and
// forwards into that hook.

#include <Arduino.h>
#include <esp_now.h>
#include <esp_idf_version.h>

// Implemented in hash_kws_master_web.ino.
extern "C" void HashKwsHandleRecvPacket(const uint8_t* src_mac,
                                        const uint8_t* data, int len);

#if defined(ESP_IDF_VERSION) && (ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 1, 0))
// ESP-IDF 5.1+: callback receives a per-packet info struct.
static void HashKwsEspNowRecvCb(const esp_now_recv_info_t* info,
                                const uint8_t* data, int len) {
  HashKwsHandleRecvPacket(info ? info->src_addr : nullptr, data, len);
}
#else
// ESP-IDF 4.x (e.g. espressif32@6.10.0 → arduino-esp32 2.0.x → IDF 4.4):
// callback receives the source MAC directly.
static void HashKwsEspNowRecvCb(const uint8_t* src_mac,
                                const uint8_t* data, int len) {
  HashKwsHandleRecvPacket(src_mac, data, len);
}
#endif

// Public entry point — called from setup() in the .ino instead of
// esp_now_register_recv_cb() directly.
extern "C" void HashKwsEspNowRegisterRecv(void) {
  esp_now_register_recv_cb(HashKwsEspNowRecvCb);
}
