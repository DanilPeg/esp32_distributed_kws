#ifndef DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_KWS_ESPNN_H_
#define DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_KWS_ESPNN_H_

#include <cstddef>
#include <cstdint>

#include "hash_model_types.h"

#if defined(__has_include)
#  if __has_include(<esp_nn.h>)
#    define HASH_KWS_ESPNN_HEADER_AVAILABLE 1
#  else
#    define HASH_KWS_ESPNN_HEADER_AVAILABLE 0
#  endif
#else
#  define HASH_KWS_ESPNN_HEADER_AVAILABLE 0
#endif

// Diagnostic switch: when set to 1 the conv chain runs through a hand-rolled
// reference int8 kernel instead of esp_nn_conv_s8 / esp_nn_depthwise_conv_s8.
// Same multiplier/shift/bias math, same NHWC/OHWI layouts. If the reference
// kernel produces correct labels but esp-nn does not, the bug is in the
// library call (signature/struct/layout mismatch); if both produce the same
// wrong output, the bug is in our quant conversion.
#ifndef HASH_KWS_ESP_NN_USE_REF_KERNEL
#  define HASH_KWS_ESP_NN_USE_REF_KERNEL 0
#endif

// Per-layer "force my ref kernel" overrides. Use these to bisect which
// library SIMD kernel produces wrong output. Set one at a time to 1, leave
// the others at 0, reflash, and check labels:
//   FORCE_REF_STEM=1  → stem via RefConvS8, dw/pw via library SIMD.
//   FORCE_REF_DW=1    → all dw via RefDepthwiseConvS8, stem/pw via library.
//   FORCE_REF_PW=1    → all pw via RefConvS8, stem/dw via library.
// If labels become correct after flipping one of them, that layer's library
// kernel is the broken one.
#ifndef HASH_KWS_ESP_NN_FORCE_REF_STEM
#  define HASH_KWS_ESP_NN_FORCE_REF_STEM 0
#endif
#ifndef HASH_KWS_ESP_NN_FORCE_REF_DW
#  define HASH_KWS_ESP_NN_FORCE_REF_DW 0
#endif
#ifndef HASH_KWS_ESP_NN_FORCE_REF_PW
#  define HASH_KWS_ESP_NN_FORCE_REF_PW 0
#endif

namespace hash_kws {

// Per-channel quant params for one conv/depthwise layer.
// multiplier is Q31 (TFLite Micro convention), shift is right-shift after the
// multiplier (positive = right shift, negative = left shift), bias is in the
// pre-multiplier int32 domain.
struct EspNnLayerQuant {
  int32_t* multiplier;  // [out_channels]
  int32_t* shift;       // [out_channels]
  int32_t* bias_q32;    // [out_channels]
  int32_t  out_offset;  // 0 for symmetric output
};

// Materialised per-layer NHWC weights + per-channel quant.
struct EspNnConvLayer {
  // Filter layout: HWIO for pointwise/stem (out_h * out_w * in_ch * out_ch
  // contiguous, with H=W=1 for pointwise). For depthwise: HWC1 (kh*kw*ch*1).
  int8_t* filter;
  EspNnLayerQuant quant;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
  int in_channels;
  int out_channels;
  bool is_depthwise;
};

struct EspNnState {
  bool ready;
  int  num_blocks;
  EspNnConvLayer stem;
  EspNnConvLayer depthwise[kHashMaxBlocks];
  EspNnConvLayer pointwise[kHashMaxBlocks];
  // Library scratch buffers required by the esp-nn SIMD kernels. Without
  // these the library falls back to a non-SIMD reference path and prints
  // "scratch_buffer not set!" on every call.
  void*  conv_scratch;
  size_t conv_scratch_bytes;
  void*  dw_scratch;
  size_t dw_scratch_bytes;
  // Pre-padded input buffer for the stem. Upstream esp-nn S3 SIMD kernel is
  // named *_filter_aligned_input_padded_esp32s3.S, which strongly implies it
  // expects the input to already be zero-padded spatially (so the SIMD inner
  // loop can avoid edge checks). We allocate a (H + 2*pad_h) × (W + 2*pad_w)
  // buffer once, zero the borders at Prepare time, and on each invoke copy
  // model_input into the interior before calling esp_nn_conv_s8 with
  // padding=0.
  int8_t* stem_padded_input;
  int     stem_padded_h;
  int     stem_padded_w;
  // Same idea for depthwise blocks. All four DW layers share one scratch
  // because only one layer is in flight at a time; sized for the max dims.
  int8_t* dw_padded_input;
  int     dw_padded_h;
  int     dw_padded_w;
};

// Returns true if the esp-nn header is reachable in this build.
bool EspNnHeaderAvailable();

// Materialises NHWC weights and per-channel quant for the whole model into
// internal SRAM. Returns false on validation/allocation failure. *out_state is
// always set to a non-null pointer that can be passed to EspNnRelease, even on
// failure (so the caller can clean up partial allocations).
bool EspNnPrepare(const HashDscnnModelData& model, EspNnState** out_state);
void EspNnRelease(EspNnState* state);

// Runs the conv chain (stem + dw/pw blocks) and AvgPool through ESP-NN.
// model_input is the existing CHW buffer (C=1 → layout-agnostic with HWC).
// scratch_a/scratch_b are reused as HWC NHWC tensors during conv. The pooled
// int8 [classifier.in_dim] is written to pooled_out; pooled_scale_out gets
// the matching activation output_scale (caller feeds these into the float
// classifier+softmax tail).
// Returns false on missing prep, incoherent state, or unsupported layer
// (e.g. residual block — caller should fall back to the int-MAC path).
bool EspNnInvokeUpToPooled(const HashDscnnModelData& model,
                           const EspNnState& state,
                           const int8_t* model_input,
                           int8_t* scratch_a,
                           int8_t* scratch_b,
                           int8_t* pooled_out,
                           float* pooled_scale_out);

// Legacy convenience signature (unused by the runner, kept for symmetry).
bool EspNnInvoke(const HashDscnnModelData& model,
                 const EspNnState& state,
                 const int8_t* model_input,
                 int8_t* scratch_a,
                 int8_t* scratch_b,
                 int8_t* output_scores);

}  // namespace hash_kws

#endif  // DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_KWS_ESPNN_H_
