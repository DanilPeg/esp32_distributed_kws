#ifndef DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_DATA_H_
#define DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_DATA_H_

// Real header for the hash-KWS exported model.  The previous content of
// this file was a TEMP_SHIM that did
//     #include "../firmware/hash_kws_runtime/hash_model_data.h"
// which from this very directory resolves back to itself -- a recursive
// include with no guards.  As a consequence, every translation unit that
// referenced hash_kws::g_hash_model (micro_speech.ino,
// hash_kws_runner.cpp, hash_micro_speech.cpp) would either fail to find
// the symbol declaration or pull it in by accident through
// hash_runtime_bridge.cpp's `#include "../../../training/hash_model_data.cpp"`
// trick.  The intended layout is:
//   - hash_model_types.h declares the data structures;
//   - hash_model_data.cpp defines the const instance g_hash_model;
//   - hash_model_data.h (this file) declares the extern.
// The training/ shim is now a thin forwarder to this header.

#include "hash_model_types.h"

namespace hash_kws {

extern const HashDscnnModelData g_hash_model;

}  // namespace hash_kws

#endif  // DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_DATA_H_
