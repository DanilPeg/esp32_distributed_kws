#ifndef DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_DATA_H_
#define DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_DATA_H_

// Real header for the hash-KWS exported model (board2 / 9261).  See
// board1_model_9128_node1/.../hash_model_data.h for the long-form rationale
// behind this fix -- the previous content was a self-referential shim with
// no include guards.

#include "hash_model_types.h"

namespace hash_kws {

extern const HashDscnnModelData g_hash_model;

}  // namespace hash_kws

#endif  // DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_DATA_H_
