#ifndef DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_KWS_RUNNER_H_
#define DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_KWS_RUNNER_H_

#include <cstddef>
#include <cstdint>

#include "hash_model_data.h"

namespace hash_kws {

struct EspNnState;  // forward decl from hash_kws_espnn.h

class HashKwsRunner {
 public:
  explicit HashKwsRunner(const HashDscnnModelData* model = &g_hash_model)
      : model_(model), esp_nn_state_(nullptr) {}

  bool IsReady() const;
  int num_classes() const;
  size_t RequiredSingleScratchBytes() const;
  size_t RequiredScratchArenaBytes() const;

  // One-shot weight materialisation for the ESP-NN fast path. Safe no-op if
  // ESP-NN is unavailable or the model has unsupported layers; the runner
  // simply stays on the int-MAC fallback.
  bool Prepare();
  bool IsEspNnReady() const;

  // Converts the existing firmware feature layout [time][frequency]
  // into the model layout [channel=1][frequency][time].
  void PrepareInputFromMicroFeatures(const int8_t* feature_slices,
                                     int8_t* model_input) const;

  bool Invoke(const int8_t* model_input,
              int8_t* scratch_a,
              int8_t* scratch_b,
              int8_t* output_scores) const;

 private:
  const HashDscnnModelData* model_;
  EspNnState* esp_nn_state_;
};

}  // namespace hash_kws

#endif  // DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_KWS_RUNNER_H_
