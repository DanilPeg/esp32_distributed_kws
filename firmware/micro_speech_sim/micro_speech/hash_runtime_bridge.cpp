// Speed the hash-KWS runtime up as much as the default Arduino toolchain
// will allow: force -O3 + aggressive loop unrolling for every translation
// unit pulled in by this bridge. This applies to hash_kws_runner.cpp,
// hash_recognize_commands.cpp and the exported hash_model_data.cpp.
#pragma GCC optimize ("O3", "unroll-loops", "tree-vectorize")

// Pass 4: route into esp-nn library (set 0) for SIMD speedup, or into our
// self-contained RefConvS8 (set 1) for a correct-but-slow fallback.
// Defines in the .ino do NOT propagate here because this is a separate
// translation unit, so the switch lives in this bridge.
//
// RefConvS8 path has been verified correct end-to-end (stem+DW+PW match
// int-MAC output byte-for-byte, top1 tracks speech). Library path has an
// output-shift bug on S3 SIMD kernels — bypass scratch-size miscalc
// hypothesis: allocate 4x more scratch than esp_nn_get_*_scratch_size
// reports, in case the function returns words rather than bytes.
#define HASH_KWS_ESP_NN_USE_REF_KERNEL 0
#define HASH_KWS_ESP_NN_SCRATCH_MULT   1

#include "../../hash_kws_runtime/hash_model_settings.cpp"
// TEMP_SOURCE: use exported model dropped into code/training/
#include "../../../training/hash_model_data.cpp"
#include "../../hash_kws_runtime/hash_recognize_commands.cpp"
#include "../../hash_kws_runtime/hash_kws_runner.cpp"
#include "../../hash_kws_runtime/hash_kws_espnn.cpp"
