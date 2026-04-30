#ifndef DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_SETTINGS_H_
#define DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_SETTINGS_H_

#include <cstdint>

namespace hash_kws {

constexpr int kCategoryCount = 12;
constexpr int kUnknownIndex = 10;
constexpr int kSilenceIndex = 11;

extern const char* kCategoryLabels[kCategoryCount];

}  // namespace hash_kws

#endif  // DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_SETTINGS_H_
