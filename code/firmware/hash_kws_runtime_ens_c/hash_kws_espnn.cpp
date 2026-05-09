#include "hash_kws_espnn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if HASH_KWS_ESPNN_HEADER_AVAILABLE
#  include <esp_heap_caps.h>
#  include <esp_nn.h>
// Workaround for https://github.com/espressif/esp-nn/issues/23 — the S3
// SIMD-optimized conv kernel (esp_nn_conv_s8_filter_aligned_input_padded_esp32s3)
// returns wrong values (saturation, accuracy collapse). Issue is open since
// 2025-08, no fix shipped. Override the dispatch macros to route into the
// generic-optimized or ANSI-C reference implementation instead. esp_nn.h
// (included above) already pulled the _ansi prototypes via
// esp_nn_ansi_headers.h, so the symbols link from libespressif__esp-nn.a.
#  undef esp_nn_conv_s8
#  undef esp_nn_depthwise_conv_s8
#  undef esp_nn_get_conv_scratch_size
#  undef esp_nn_set_conv_scratch_buf
#  undef esp_nn_get_depthwise_conv_scratch_size
#  undef esp_nn_set_depthwise_conv_scratch_buf
#  define esp_nn_conv_s8                       esp_nn_conv_s8_ansi
#  define esp_nn_depthwise_conv_s8             esp_nn_depthwise_conv_s8_ansi
#  define esp_nn_get_conv_scratch_size         esp_nn_get_conv_scratch_size_ansi
#  define esp_nn_set_conv_scratch_buf          esp_nn_set_conv_scratch_buf_ansi
#  define esp_nn_get_depthwise_conv_scratch_size  esp_nn_get_depthwise_conv_scratch_size_ansi
#  define esp_nn_set_depthwise_conv_scratch_buf   esp_nn_set_depthwise_conv_scratch_buf_ansi
#endif

namespace hash_kws {
namespace {

// --- Hash math: re-implement the int8 weight derivation locally so this
// translation unit is self-contained. The constants must match
// hash_kws_runner.cpp; they're well-defined model contracts, not impl detail.

constexpr int kEspConvHashOc    = 1337;
constexpr int kEspConvHashIc    = 7919;
constexpr int kEspConvHashKh    = 2971;
constexpr int kEspConvHashKw    = 6151;
constexpr int kEspConvHashLayer = 104729;

constexpr int kEspDwHashCh    = 1337;
constexpr int kEspDwHashKh    = 7919;
constexpr int kEspDwHashKw    = 2971;
constexpr int kEspDwHashLayer = 104729;

constexpr int kEspSignHashA = 4099;
constexpr int kEspSignHashB = 6151;
constexpr int kEspSignHashC = 14887;

inline int EspWrapPositiveMod(int value, int modulus) {
  int r = value % modulus;
  if (r < 0) r += modulus;
  return r;
}

inline int8_t EspHashStemWeightInt8(const HashConvLayerData& layer,
                                    int output_channel,
                                    int kernel_row,
                                    int kernel_col) {
  const int raw =
      (output_channel * kEspConvHashOc) + (0 * kEspConvHashIc) +
      (kernel_row * kEspConvHashKh) + (kernel_col * kEspConvHashKw) +
      (layer.layer_id * kEspConvHashLayer);
  const int bucket = EspWrapPositiveMod(raw, layer.codebook_size);
  int v = layer.codebook[bucket];
  if (layer.signed_hash) {
    const int seed =
        (output_channel * kEspSignHashA) + (0 * kEspSignHashB) +
        (kernel_row * kEspSignHashC) + (kernel_col * (kEspSignHashA + kEspSignHashB)) +
        (layer.layer_id * (kEspSignHashC + 11));
    if (EspWrapPositiveMod(seed, 2) == 0) v = -v;
  }
  if (v < -128) v = -128;
  if (v >  127) v =  127;
  return static_cast<int8_t>(v);
}

inline int8_t EspHashPointwiseWeightInt8(const HashConvLayerData& layer,
                                         int output_channel,
                                         int input_channel) {
  const int raw =
      (output_channel * kEspConvHashOc) + (input_channel * kEspConvHashIc) +
      (layer.layer_id * kEspConvHashLayer);
  const int bucket = EspWrapPositiveMod(raw, layer.codebook_size);
  int v = layer.codebook[bucket];
  if (layer.signed_hash) {
    const int seed =
        (output_channel * kEspSignHashA) + (input_channel * kEspSignHashB) +
        (layer.layer_id * (kEspSignHashC + 11));
    if (EspWrapPositiveMod(seed, 2) == 0) v = -v;
  }
  if (v < -128) v = -128;
  if (v >  127) v =  127;
  return static_cast<int8_t>(v);
}

inline int8_t EspHashDepthwiseWeightInt8(const HashDepthwiseLayerData& layer,
                                         int channel,
                                         int kernel_row,
                                         int kernel_col) {
  const int raw =
      (channel * kEspDwHashCh) + (kernel_row * kEspDwHashKh) +
      (kernel_col * kEspDwHashKw) + (layer.layer_id * kEspDwHashLayer);
  const int bucket = EspWrapPositiveMod(raw, layer.codebook_size);
  int v = layer.codebook[bucket];
  if (layer.signed_hash) {
    const int seed =
        (channel * kEspSignHashA) + (kernel_row * kEspSignHashB) +
        (kernel_col * kEspSignHashC) + (layer.layer_id * (kEspSignHashA + 29));
    if (EspWrapPositiveMod(seed, 2) == 0) v = -v;
  }
  if (v < -128) v = -128;
  if (v >  127) v =  127;
  return static_cast<int8_t>(v);
}

inline int EspOutputDim(int input, int kernel, int stride, int padding) {
  return ((input + (2 * padding) - kernel) / stride) + 1;
}

// --- Internal-SRAM allocators with zero-init for arrays we'll fill below.

void* AllocInternal(size_t bytes) {
#if HASH_KWS_ESPNN_HEADER_AVAILABLE
  if (bytes == 0) return nullptr;
  void* p = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (p != nullptr) std::memset(p, 0, bytes);
  return p;
#else
  (void)bytes;
  return nullptr;
#endif
}

// Internal-first, PSRAM-fallback. Used for large read-mostly buffers (padded
// inputs, filter weights) that don't need to live in internal SRAM — the
// library reads them once per invoke, so PSRAM latency is tolerable, and we
// need to leave enough internal heap for the TF audio provider ring buffer.
void* AllocAny(size_t bytes) {
#if HASH_KWS_ESPNN_HEADER_AVAILABLE
  if (bytes == 0) return nullptr;
  void* p = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (p == nullptr) {
    p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (p == nullptr) {
    p = heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
  }
  if (p != nullptr) std::memset(p, 0, bytes);
  return p;
#else
  (void)bytes;
  return nullptr;
#endif
}

void FreeInternal(void* p) {
#if HASH_KWS_ESPNN_HEADER_AVAILABLE
  if (p != nullptr) heap_caps_free(p);
#else
  (void)p;
#endif
}

// TFLite Micro / esp-nn convention: shift is the frexp exponent directly.
// Reconstruction: real ≈ (mantissa_q31 / 2^31) * 2^shift.
// In the runtime: shift > 0 means LEFT-shift the accumulator by `shift` before
// SRDHM with mantissa_q31; shift < 0 means RIGHT-shift the SRDHM result by
// |shift|. Both esp-nn and TFLM split shift into LEFT_SHIFT / RIGHT_SHIFT.
void QuantizeMultiplier(double real_multiplier, int32_t* out_mult, int32_t* out_shift) {
  if (real_multiplier <= 0.0) {
    *out_mult = 0;
    *out_shift = 0;
    return;
  }
  int shift_exp = 0;
  const double sig = std::frexp(real_multiplier, &shift_exp);  // sig in [0.5, 1.0)
  int64_t mantissa = static_cast<int64_t>(std::round(sig * (1LL << 31)));
  if (mantissa == (1LL << 31)) {
    mantissa /= 2;
    ++shift_exp;
  }
  int32_t shift = shift_exp;
  // For typical conv quant ratios real_multiplier is in [1e-5, 1e-1] giving
  // shift in roughly [-17, -3]. The clamps below are safety only.
  if (shift > 30) {
    // multiplier is so large the accumulator would overflow on left-shift;
    // clamp to max representable.
    shift = 30;
    mantissa = (1LL << 31) - 1;
  }
  if (shift < -31) {
    // multiplier is effectively zero.
    shift = -31;
    mantissa = 0;
  }
  *out_mult = static_cast<int32_t>(mantissa);
  *out_shift = shift;
}

bool AllocConvLayerQuant(EspNnLayerQuant* q, int channels) {
  q->multiplier = static_cast<int32_t*>(AllocInternal(sizeof(int32_t) * channels));
  q->shift      = static_cast<int32_t*>(AllocInternal(sizeof(int32_t) * channels));
  q->bias_q32   = static_cast<int32_t*>(AllocInternal(sizeof(int32_t) * channels));
  q->out_offset = 0;
  return q->multiplier && q->shift && q->bias_q32;
}

void FreeConvLayerQuant(EspNnLayerQuant* q) {
  FreeInternal(q->multiplier); q->multiplier = nullptr;
  FreeInternal(q->shift);      q->shift = nullptr;
  FreeInternal(q->bias_q32);   q->bias_q32 = nullptr;
}

void FillPerChannelQuant(EspNnLayerQuant* q,
                         const float* post_scale,
                         const float* post_bias,
                         float codebook_scale,
                         float input_scale,
                         float output_scale,
                         int channels) {
  for (int c = 0; c < channels; ++c) {
    const float filter_scale = post_scale[c] * codebook_scale;
    const double real_mul = (static_cast<double>(input_scale) *
                             static_cast<double>(filter_scale)) /
                            static_cast<double>(output_scale);
    QuantizeMultiplier(real_mul, &q->multiplier[c], &q->shift[c]);

    const double bias_real  = static_cast<double>(post_bias[c]);
    const double bias_denom = static_cast<double>(input_scale) *
                              static_cast<double>(filter_scale);
    long long bias_q = (bias_denom > 0.0)
                           ? static_cast<long long>(std::llround(bias_real / bias_denom))
                           : 0LL;
    if (bias_q >  2147483647LL) bias_q =  2147483647LL;
    if (bias_q < -2147483648LL) bias_q = -2147483648LL;
    q->bias_q32[c] = static_cast<int32_t>(bias_q);
  }
}

// --- HWC weight materialisation -----------------------------------------

bool BuildStemFilter(const HashConvLayerData& layer, EspNnConvLayer* dst) {
  // esp-nn conv expects filter in OHWI layout:
  //   idx = oc*KH*KW*IC + kh*KW*IC + kw*IC + ic
  // For the stem IC=1, so it collapses to oc*KH*KW + kh*KW + kw.
  const int ic = layer.in_channels;
  const int oc = layer.out_channels;
  const int filter_bytes = layer.kernel_h * layer.kernel_w * ic * oc;
  dst->filter = static_cast<int8_t*>(AllocInternal(filter_bytes));
  if (dst->filter == nullptr) return false;
  for (int o = 0; o < oc; ++o) {
    for (int kh = 0; kh < layer.kernel_h; ++kh) {
      for (int kw = 0; kw < layer.kernel_w; ++kw) {
        for (int i = 0; i < ic; ++i) {
          const int idx = (((o * layer.kernel_h) + kh) * layer.kernel_w + kw) * ic + i;
          dst->filter[idx] = EspHashStemWeightInt8(layer, o, kh, kw);
        }
      }
    }
  }
  dst->kernel_h = layer.kernel_h;
  dst->kernel_w = layer.kernel_w;
  dst->stride_h = layer.stride_h;
  dst->stride_w = layer.stride_w;
  dst->padding_h = layer.padding_h;
  dst->padding_w = layer.padding_w;
  dst->in_channels = ic;
  dst->out_channels = oc;
  dst->is_depthwise = false;
  return true;
}

bool BuildPointwiseFilter(const HashConvLayerData& layer, EspNnConvLayer* dst) {
  // esp-nn conv expects OHWI; for 1x1 KH=KW=1 → idx = oc*IC + ic.
  const int ic = layer.in_channels;
  const int oc = layer.out_channels;
  const int filter_bytes = ic * oc;
  dst->filter = static_cast<int8_t*>(AllocInternal(filter_bytes));
  if (dst->filter == nullptr) return false;
  for (int o = 0; o < oc; ++o) {
    for (int i = 0; i < ic; ++i) {
      dst->filter[o * ic + i] = EspHashPointwiseWeightInt8(layer, o, i);
    }
  }
  dst->kernel_h = 1;
  dst->kernel_w = 1;
  dst->stride_h = 1;
  dst->stride_w = 1;
  dst->padding_h = 0;
  dst->padding_w = 0;
  dst->in_channels = ic;
  dst->out_channels = oc;
  dst->is_depthwise = false;
  return true;
}

bool BuildDepthwiseFilter(const HashDepthwiseLayerData& layer, EspNnConvLayer* dst) {
  // HWC1; idx = (kh * KW + kw) * C + c.
  const int c = layer.channels;
  const int filter_bytes = layer.kernel_h * layer.kernel_w * c;
  dst->filter = static_cast<int8_t*>(AllocInternal(filter_bytes));
  if (dst->filter == nullptr) return false;
  for (int kh = 0; kh < layer.kernel_h; ++kh) {
    for (int kw = 0; kw < layer.kernel_w; ++kw) {
      for (int ch = 0; ch < c; ++ch) {
        const int idx = ((kh * layer.kernel_w) + kw) * c + ch;
        dst->filter[idx] = EspHashDepthwiseWeightInt8(layer, ch, kh, kw);
      }
    }
  }
  dst->kernel_h = layer.kernel_h;
  dst->kernel_w = layer.kernel_w;
  dst->stride_h = layer.stride_h;
  dst->stride_w = layer.stride_w;
  dst->padding_h = layer.padding_h;
  dst->padding_w = layer.padding_w;
  dst->in_channels = c;
  dst->out_channels = c;
  dst->is_depthwise = true;
  return true;
}

void FreeConvLayer(EspNnConvLayer* layer) {
  FreeInternal(layer->filter);
  layer->filter = nullptr;
  FreeConvLayerQuant(&layer->quant);
}

#if HASH_KWS_ESPNN_HEADER_AVAILABLE

// --- Reference int8 conv (diagnostic). Uses the same data layouts and
// multiplier/shift/bias convention as esp-nn so we can isolate library
// behaviour from quant conversion. Slow but mathematically transparent.

inline int32_t RefSrdHm(int32_t a, int32_t b) {
  // Saturating Rounding Doubling High Mul: round(a * b / 2^31).
  if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
  int64_t prod = static_cast<int64_t>(a) * static_cast<int64_t>(b);
  int64_t nudge = (prod >= 0) ? (1LL << 30) : (1LL - (1LL << 30));
  return static_cast<int32_t>((prod + nudge) >> 31);
}

inline int32_t RefDivByPotRound(int32_t x, int32_t exp) {
  if (exp <= 0) return x;
  int32_t mask = (1 << exp) - 1;
  int32_t rem  = x & mask;
  int32_t res  = x >> exp;
  int32_t threshold = (mask >> 1) + (res < 0 ? 1 : 0);
  if (rem > threshold) ++res;
  return res;
}

inline int32_t RefRequantize(int32_t value, int32_t mult, int32_t shift,
                             int32_t out_offset, int32_t amin, int32_t amax) {
  int32_t left_shift  = shift > 0 ? shift : 0;
  int32_t right_shift = shift > 0 ? 0 : -shift;
  int32_t prod = RefSrdHm(value * (1 << left_shift), mult);
  int32_t res  = RefDivByPotRound(prod, right_shift);
  res += out_offset;
  if (res < amin) res = amin;
  if (res > amax) res = amax;
  return res;
}

void RefConvS8(const data_dims_t* in_d, const int8_t* in_data,
               const data_dims_t* flt_d, const int8_t* flt_data,
               const int32_t* bias_data,
               const data_dims_t* out_d, int8_t* out_data,
               const conv_params_t* cp,
               const quant_data_t* qd) {
  const int IH = in_d->height,  IW = in_d->width,  IC = in_d->channels;
  const int OH = out_d->height, OW = out_d->width, OC = out_d->channels;
  const int KH = flt_d->height, KW = flt_d->width;
  static bool s_pass4_refconv_params_dumped = false;
  if (!s_pass4_refconv_params_dumped) {
    s_pass4_refconv_params_dumped = true;
    Serial.printf("hash_dbg refconv params stride_w=%ld stride_h=%ld pad_w=%ld pad_h=%ld IH=%d IW=%d OH=%d OW=%d KH=%d KW=%d IC=%d OC=%d in_off=%ld out_off=%ld act_min=%ld act_max=%ld\n",
                  static_cast<long>(cp->stride.width), static_cast<long>(cp->stride.height),
                  static_cast<long>(cp->padding.width), static_cast<long>(cp->padding.height),
                  IH, IW, OH, OW, KH, KW, IC, OC,
                  static_cast<long>(cp->in_offset), static_cast<long>(cp->out_offset),
                  static_cast<long>(cp->activation.min), static_cast<long>(cp->activation.max));
  }
  for (int oh = 0; oh < OH; ++oh) {
    for (int ow = 0; ow < OW; ++ow) {
      for (int oc = 0; oc < OC; ++oc) {
        int32_t acc = bias_data ? bias_data[oc] : 0;
        for (int kh = 0; kh < KH; ++kh) {
          const int ih = oh * cp->stride.height + kh - cp->padding.height;
          if (ih < 0 || ih >= IH) continue;
          for (int kw = 0; kw < KW; ++kw) {
            const int iw = ow * cp->stride.width + kw - cp->padding.width;
            if (iw < 0 || iw >= IW) continue;
            for (int ic = 0; ic < IC; ++ic) {
              const int in_idx  = (ih * IW + iw) * IC + ic;
              const int flt_idx = ((oc * KH + kh) * KW + kw) * IC + ic;
              acc += (static_cast<int32_t>(in_data[in_idx]) + cp->in_offset)
                     * static_cast<int32_t>(flt_data[flt_idx]);
            }
          }
        }
        const int32_t res =
            RefRequantize(acc, qd->mult[oc], qd->shift[oc],
                          cp->out_offset, cp->activation.min, cp->activation.max);
        const int write_idx = (oh * OW + ow) * OC + oc;
        if (oh == 0 && ow == 0 && oc == 0) {
          static bool s_pass4_refconv_000 = false;
          if (!s_pass4_refconv_000) {
            s_pass4_refconv_000 = true;
            Serial.printf("hash_dbg refconv (0,0,0) acc=%ld mult=%ld shift=%ld res=%ld write_idx=%d\n",
                          static_cast<long>(acc),
                          static_cast<long>(qd->mult[oc]), static_cast<long>(qd->shift[oc]),
                          static_cast<long>(res), write_idx);
          }
        }
        out_data[write_idx] = static_cast<int8_t>(res);
      }
    }
  }
}

void RefDepthwiseConvS8(const data_dims_t* in_d, const int8_t* in_data,
                        const data_dims_t* flt_d, const int8_t* flt_data,
                        const int32_t* bias_data,
                        const data_dims_t* out_d, int8_t* out_data,
                        const dw_conv_params_t* cp,
                        const quant_data_t* qd) {
  const int IH = in_d->height,  IW = in_d->width,  IC = in_d->channels;
  const int OH = out_d->height, OW = out_d->width;
  const int KH = flt_d->height, KW = flt_d->width;
  for (int oh = 0; oh < OH; ++oh) {
    for (int ow = 0; ow < OW; ++ow) {
      for (int ic = 0; ic < IC; ++ic) {
        int32_t acc = bias_data ? bias_data[ic] : 0;
        for (int kh = 0; kh < KH; ++kh) {
          const int ih = oh * cp->stride.height + kh - cp->padding.height;
          if (ih < 0 || ih >= IH) continue;
          for (int kw = 0; kw < KW; ++kw) {
            const int iw = ow * cp->stride.width + kw - cp->padding.width;
            if (iw < 0 || iw >= IW) continue;
            const int in_idx  = (ih * IW + iw) * IC + ic;
            const int flt_idx = (kh * KW + kw) * IC + ic;
            acc += (static_cast<int32_t>(in_data[in_idx]) + cp->in_offset)
                   * static_cast<int32_t>(flt_data[flt_idx]);
          }
        }
        const int32_t res =
            RefRequantize(acc, qd->mult[ic], qd->shift[ic],
                          cp->out_offset, cp->activation.min, cp->activation.max);
        out_data[(oh * OW + ow) * IC + ic] = static_cast<int8_t>(res);
      }
    }
  }
}

inline void DispatchConv(const data_dims_t* in_d, const int8_t* in_data,
                         const data_dims_t* flt_d, const int8_t* flt_data,
                         const int32_t* bias_data,
                         const data_dims_t* out_d, int8_t* out_data,
                         const conv_params_t* cp,
                         const quant_data_t* qd) {
#if HASH_KWS_ESP_NN_USE_REF_KERNEL
  RefConvS8(in_d, in_data, flt_d, flt_data, bias_data, out_d, out_data, cp, qd);
#else
  esp_nn_conv_s8(in_d, in_data, flt_d, flt_data, bias_data, out_d, out_data, cp, qd);
#endif
}

inline void DispatchDepthwise(const data_dims_t* in_d, const int8_t* in_data,
                              const data_dims_t* flt_d, const int8_t* flt_data,
                              const int32_t* bias_data,
                              const data_dims_t* out_d, int8_t* out_data,
                              const dw_conv_params_t* cp,
                              const quant_data_t* qd) {
#if HASH_KWS_ESP_NN_USE_REF_KERNEL
  RefDepthwiseConvS8(in_d, in_data, flt_d, flt_data, bias_data, out_d, out_data, cp, qd);
#else
  esp_nn_depthwise_conv_s8(in_d, in_data, flt_d, flt_data, bias_data, out_d, out_data, cp, qd);
#endif
}

void RunStemEspNn(const HashConvLayerData& spec,
                  const EspNnConvLayer& prep,
                  const int8_t* input_hwc, int input_h, int input_w,
                  int8_t* output_hwc) {
  const int out_h = EspOutputDim(input_h, spec.kernel_h, spec.stride_h, spec.padding_h);
  const int out_w = EspOutputDim(input_w, spec.kernel_w, spec.stride_w, spec.padding_w);

  // esp-nn data_dims: input/output use channels=ch, extra=batch; filter uses
  // channels=in_ch, extra=out_ch. Filter layout is OHWI. Use named-field
  // assignment because the actual installed esp_nn_defs.h may order fields
  // differently than the canonical header (positional init silently swaps
  // width/height in that case → output is offset / transposed).
  data_dims_t in_dims  = {};
  in_dims.width = input_w;  in_dims.height = input_h;
  in_dims.channels = spec.in_channels; in_dims.extra = 1;
  data_dims_t flt_dims = {};
  flt_dims.width = spec.kernel_w; flt_dims.height = spec.kernel_h;
  flt_dims.channels = spec.in_channels; flt_dims.extra = spec.out_channels;
  data_dims_t out_dims = {};
  out_dims.width = out_w; out_dims.height = out_h;
  out_dims.channels = spec.out_channels; out_dims.extra = 1;

  conv_params_t cp = {};
  cp.in_offset       = 0;
  cp.out_offset      = 0;
  cp.stride.width    = spec.stride_w;
  cp.stride.height   = spec.stride_h;
  cp.padding.width   = spec.padding_w;
  cp.padding.height  = spec.padding_h;
  cp.dilation.width  = 1;
  cp.dilation.height = 1;
  cp.activation.min  = 0;
  cp.activation.max  = 127;

  // NOTE: keeping positional init deliberately — earlier runs with
  // { prep.quant.shift, prep.quant.multiplier } produced correct stem output,
  // and switching to named-field init broke stem. The actual struct order in
  // the installed esp-nn header appears to match this positional layout.
  quant_data_t qd = { prep.quant.shift, prep.quant.multiplier };

  DispatchConv(&in_dims, input_hwc,
               &flt_dims, prep.filter,
               prep.quant.bias_q32,
               &out_dims, output_hwc,
               &cp, &qd);
}

void RunDepthwiseEspNn(const HashDepthwiseLayerData& spec,
                       const EspNnConvLayer& prep,
                       const int8_t* input_hwc, int input_h, int input_w,
                       int8_t* output_hwc) {
  const int out_h = EspOutputDim(input_h, spec.kernel_h, spec.stride_h, spec.padding_h);
  const int out_w = EspOutputDim(input_w, spec.kernel_w, spec.stride_w, spec.padding_w);

  // Depthwise filter layout in esp-nn: HWC with channels = in_ch * ch_mult.
  // data_dims: channels=C, extra=1. Named-field for header field-order safety.
  data_dims_t in_dims  = {};
  in_dims.width = input_w; in_dims.height = input_h;
  in_dims.channels = spec.channels; in_dims.extra = 1;
  data_dims_t flt_dims = {};
  flt_dims.width = spec.kernel_w; flt_dims.height = spec.kernel_h;
  flt_dims.channels = spec.channels; flt_dims.extra = 1;
  data_dims_t out_dims = {};
  out_dims.width = out_w; out_dims.height = out_h;
  out_dims.channels = spec.channels; out_dims.extra = 1;

  dw_conv_params_t cp = {};
  cp.in_offset       = 0;
  cp.out_offset      = 0;
  cp.ch_mult         = 1;
  cp.stride.width    = spec.stride_w;
  cp.stride.height   = spec.stride_h;
  cp.padding.width   = spec.padding_w;
  cp.padding.height  = spec.padding_h;
  cp.dilation.width  = 1;
  cp.dilation.height = 1;
  cp.activation.min  = 0;
  cp.activation.max  = 127;

  // NOTE: keeping positional init deliberately — earlier runs with
  // { prep.quant.shift, prep.quant.multiplier } produced correct stem output,
  // and switching to named-field init broke stem. The actual struct order in
  // the installed esp-nn header appears to match this positional layout.
  quant_data_t qd = { prep.quant.shift, prep.quant.multiplier };

  DispatchDepthwise(&in_dims, input_hwc,
                    &flt_dims, prep.filter,
                    prep.quant.bias_q32,
                    &out_dims, output_hwc,
                    &cp, &qd);
}

// Forward decl of the S3 SIMD conv kernel and its scratch-buffer setter
// + sizing function. We call _esp32s3 directly for 1×1 pointwise (mult8
// kernel, distinct from the buggy filter_aligned_input_padded one). The
// _esp32s3 backend has its own static scratch buffer pointer separate
// from the _ansi backend, so we need to set both.
extern "C" void esp_nn_conv_s8_esp32s3(const data_dims_t *input_dims,
                                       const int8_t *input_data,
                                       const data_dims_t *filter_dims,
                                       const int8_t *filter_data,
                                       const int32_t *bias,
                                       const data_dims_t *output_dims,
                                       int8_t *output_data,
                                       const conv_params_t *conv_params,
                                       const quant_data_t *quant_data);
extern "C" int  esp_nn_get_conv_scratch_size_esp32s3(const data_dims_t *input_dims,
                                                     const data_dims_t *filter_dims,
                                                     const data_dims_t *output_dims,
                                                     const conv_params_t *conv_params);
extern "C" void esp_nn_set_conv_scratch_buf_esp32s3(const void *buf);

void RunPointwiseEspNn(const HashConvLayerData& spec,
                       const EspNnConvLayer& prep,
                       const int8_t* input_hwc, int input_h, int input_w,
                       int8_t* output_hwc) {
  data_dims_t in_dims  = {};
  in_dims.width = input_w; in_dims.height = input_h;
  in_dims.channels = spec.in_channels; in_dims.extra = 1;
  data_dims_t flt_dims = {};
  flt_dims.width = 1; flt_dims.height = 1;
  flt_dims.channels = spec.in_channels; flt_dims.extra = spec.out_channels;
  data_dims_t out_dims = {};
  out_dims.width = input_w; out_dims.height = input_h;
  out_dims.channels = spec.out_channels; out_dims.extra = 1;

  conv_params_t cp = {};
  cp.in_offset       = 0;
  cp.out_offset      = 0;
  cp.stride.width    = 1;
  cp.stride.height   = 1;
  cp.padding.width   = 0;
  cp.padding.height  = 0;
  cp.dilation.width  = 1;
  cp.dilation.height = 1;
  cp.activation.min  = 0;
  cp.activation.max  = 127;

  quant_data_t qd = { prep.quant.shift, prep.quant.multiplier };

  // For pointwise (1×1) we explicitly call the S3 SIMD kernel — it routes
  // to esp_nn_conv_s8_mult8_1x1_esp32s3 internally, which (per the file
  // naming in esp-nn) is separate from the buggy filter_aligned_input_padded
  // kernel that Issue #23 reports. PW dominates compute (~80% of model)
  // so SIMD here is the biggest available win.
#if HASH_KWS_ESPNN_HEADER_AVAILABLE && !HASH_KWS_ESP_NN_USE_REF_KERNEL
  esp_nn_conv_s8_esp32s3(&in_dims, input_hwc,
                         &flt_dims, prep.filter,
                         prep.quant.bias_q32,
                         &out_dims, output_hwc,
                         &cp, &qd);
#else
  DispatchConv(&in_dims, input_hwc,
               &flt_dims, prep.filter,
               prep.quant.bias_q32,
               &out_dims, output_hwc,
               &cp, &qd);
#endif
}

#endif  // HASH_KWS_ESPNN_HEADER_AVAILABLE

void AveragePoolHwc(const int8_t* hwc, int rows, int cols, int channels,
                    float input_scale, int8_t* pooled, float pooled_scale) {
  const int spatial = rows * cols;
  for (int c = 0; c < channels; ++c) {
    int32_t sum = 0;
    for (int s = 0; s < spatial; ++s) {
      sum += static_cast<int32_t>(hwc[s * channels + c]);
    }
    const float mean_val =
        (static_cast<float>(sum) / static_cast<float>(spatial)) * input_scale;
    const float scale = pooled_scale > 0.0f ? pooled_scale : 1.0f;
    int q = static_cast<int>(std::lround(mean_val / scale));
    if (q < -128) q = -128;
    if (q >  127) q =  127;
    pooled[c] = static_cast<int8_t>(q);
  }
}

}  // namespace

bool EspNnHeaderAvailable() {
#if HASH_KWS_ESPNN_HEADER_AVAILABLE
  return true;
#else
  return false;
#endif
}

bool EspNnPrepare(const HashDscnnModelData& model, EspNnState** out_state) {
  *out_state = static_cast<EspNnState*>(AllocInternal(sizeof(EspNnState)));
  if (*out_state == nullptr) return false;
  EspNnState* s = *out_state;
  s->ready = false;
  s->num_blocks = model.num_blocks;

#if !HASH_KWS_ESPNN_HEADER_AVAILABLE
  return false;
#else
  if (!BuildStemFilter(model.stem, &s->stem)) return false;
  if (!AllocConvLayerQuant(&s->stem.quant, model.stem.out_channels)) return false;
  FillPerChannelQuant(&s->stem.quant,
                      model.stem.post_scale,
                      model.stem.post_bias,
                      model.stem.codebook_scale,
                      model.activations[0].input_scale,
                      model.activations[0].output_scale,
                      model.stem.out_channels);

  for (int b = 0; b < model.num_blocks; ++b) {
    const int dw_stage = 1 + (2 * b);
    const int pw_stage = dw_stage + 1;

    if (!BuildDepthwiseFilter(model.depthwise[b], &s->depthwise[b])) return false;
    if (!AllocConvLayerQuant(&s->depthwise[b].quant, model.depthwise[b].channels)) return false;
    FillPerChannelQuant(&s->depthwise[b].quant,
                        model.depthwise[b].post_scale,
                        model.depthwise[b].post_bias,
                        model.depthwise[b].codebook_scale,
                        model.activations[dw_stage].input_scale,
                        model.activations[dw_stage].output_scale,
                        model.depthwise[b].channels);

    if (!BuildPointwiseFilter(model.pointwise[b], &s->pointwise[b])) return false;
    if (!AllocConvLayerQuant(&s->pointwise[b].quant, model.pointwise[b].out_channels)) return false;
    FillPerChannelQuant(&s->pointwise[b].quant,
                        model.pointwise[b].post_scale,
                        model.pointwise[b].post_bias,
                        model.pointwise[b].codebook_scale,
                        model.activations[pw_stage].input_scale,
                        model.activations[pw_stage].output_scale,
                        model.pointwise[b].out_channels);
  }

  // Compute the worst-case esp-nn scratch buffer sizes by iterating the
  // model dims (same walk as Invoke). Without these set the library falls
  // back to a non-SIMD path and spams "scratch_buffer not set!".
  size_t max_conv_scratch = 0;
  size_t max_dw_scratch   = 0;
  {
    int rows = model.input_rows;
    int cols = model.input_cols;

    auto fill_conv_params = [](const HashConvLayerData& l, conv_params_t* cp) {
      cp->in_offset       = 0;
      cp->out_offset      = 0;
      cp->stride.width    = l.stride_w;
      cp->stride.height   = l.stride_h;
      cp->padding.width   = l.padding_w;
      cp->padding.height  = l.padding_h;
      cp->dilation.width  = 1;
      cp->dilation.height = 1;
      cp->activation.min  = 0;
      cp->activation.max  = 127;
    };
    auto fill_dw_params = [](const HashDepthwiseLayerData& l, dw_conv_params_t* cp) {
      cp->in_offset       = 0;
      cp->out_offset      = 0;
      cp->ch_mult         = 1;
      cp->stride.width    = l.stride_w;
      cp->stride.height   = l.stride_h;
      cp->padding.width   = l.padding_w;
      cp->padding.height  = l.padding_h;
      cp->dilation.width  = 1;
      cp->dilation.height = 1;
      cp->activation.min  = 0;
      cp->activation.max  = 127;
    };

    auto fill_dims = [](data_dims_t* d, int w, int h, int ch, int extra) {
      *d = data_dims_t{};
      d->width = w; d->height = h; d->channels = ch; d->extra = extra;
    };

    {
      const HashConvLayerData& l = model.stem;
      const int oh = EspOutputDim(rows, l.kernel_h, l.stride_h, l.padding_h);
      const int ow = EspOutputDim(cols, l.kernel_w, l.stride_w, l.padding_w);
      data_dims_t in_d, flt_d, out_d;
      fill_dims(&in_d,  cols, rows, l.in_channels, 1);
      fill_dims(&flt_d, l.kernel_w, l.kernel_h, l.in_channels, l.out_channels);
      fill_dims(&out_d, ow, oh, l.out_channels, 1);
      conv_params_t cp = {}; fill_conv_params(l, &cp);
      const size_t sz = esp_nn_get_conv_scratch_size(&in_d, &flt_d, &out_d, &cp);
      if (sz > max_conv_scratch) max_conv_scratch = sz;
      rows = oh; cols = ow;
    }

    for (int b = 0; b < model.num_blocks; ++b) {
      {
        const HashDepthwiseLayerData& l = model.depthwise[b];
        const int oh = EspOutputDim(rows, l.kernel_h, l.stride_h, l.padding_h);
        const int ow = EspOutputDim(cols, l.kernel_w, l.stride_w, l.padding_w);
        data_dims_t in_d, flt_d, out_d;
        fill_dims(&in_d,  cols, rows, l.channels, 1);
        fill_dims(&flt_d, l.kernel_w, l.kernel_h, l.channels, 1);
        fill_dims(&out_d, ow, oh, l.channels, 1);
        dw_conv_params_t cp = {}; fill_dw_params(l, &cp);
        const size_t sz = esp_nn_get_depthwise_conv_scratch_size(&in_d, &flt_d, &out_d, &cp);
        if (sz > max_dw_scratch) max_dw_scratch = sz;
        rows = oh; cols = ow;
      }
      {
        const HashConvLayerData& l = model.pointwise[b];
        data_dims_t in_d, flt_d, out_d;
        fill_dims(&in_d,  cols, rows, l.in_channels, 1);
        fill_dims(&flt_d, 1, 1, l.in_channels, l.out_channels);
        fill_dims(&out_d, cols, rows, l.out_channels, 1);
        conv_params_t cp = {}; fill_conv_params(l, &cp);
        const size_t sz_ansi = esp_nn_get_conv_scratch_size(&in_d, &flt_d, &out_d, &cp);
        const size_t sz_s3   = esp_nn_get_conv_scratch_size_esp32s3(&in_d, &flt_d, &out_d, &cp);
        const size_t sz = sz_ansi > sz_s3 ? sz_ansi : sz_s3;
        if (sz > max_conv_scratch) max_conv_scratch = sz;
        // 1x1 doesn't change spatial dims.
      }
    }
  }

  // H3 hypothesis: esp_nn_get_conv_scratch_size may return size in int32
  // words rather than bytes, causing us to under-allocate. Apply a safety
  // multiplier so the library can't overflow into neighbouring memory.
#ifndef HASH_KWS_ESP_NN_SCRATCH_MULT
#  define HASH_KWS_ESP_NN_SCRATCH_MULT 1
#endif
  if (max_conv_scratch > 0) {
    const size_t alloc_bytes = max_conv_scratch * HASH_KWS_ESP_NN_SCRATCH_MULT;
    s->conv_scratch = AllocInternal(alloc_bytes);
    if (s->conv_scratch == nullptr) return false;
    s->conv_scratch_bytes = alloc_bytes;
    esp_nn_set_conv_scratch_buf(s->conv_scratch);  // _ansi backend (stem path)
    esp_nn_set_conv_scratch_buf_esp32s3(s->conv_scratch);  // S3 SIMD backend (PW path)
  }
  if (max_dw_scratch > 0) {
    const size_t alloc_bytes = max_dw_scratch * HASH_KWS_ESP_NN_SCRATCH_MULT;
    s->dw_scratch = AllocInternal(alloc_bytes);
    if (s->dw_scratch == nullptr) return false;
    s->dw_scratch_bytes = alloc_bytes;
    esp_nn_set_depthwise_conv_scratch_buf(s->dw_scratch);
  }

  // Pre-padded input buffers were needed only for the buggy S3 SIMD kernel
  // (https://github.com/espressif/esp-nn/issues/23). Now that we route into
  // the _ansi reference implementation it handles padding internally and
  // these buffers are no longer required — keeps RAM footprint tight.
  s->stem_padded_input = nullptr;
  s->dw_padded_input   = nullptr;

  s->ready = true;

  // One-shot dump of stem channel-0 quant for sanity vs hand-calc.
  Serial.printf("hash_dbg pass4 prep stem ch0 mult=%ld shift=%ld bias_q32=%ld input_scale=%.6f post_scale=%.6f post_bias=%.6f codebook_scale=%.6f output_scale=%.6f\n",
                static_cast<long>(s->stem.quant.multiplier[0]),
                static_cast<long>(s->stem.quant.shift[0]),
                static_cast<long>(s->stem.quant.bias_q32[0]),
                static_cast<double>(model.activations[0].input_scale),
                static_cast<double>(model.stem.post_scale[0]),
                static_cast<double>(model.stem.post_bias[0]),
                static_cast<double>(model.stem.codebook_scale),
                static_cast<double>(model.activations[0].output_scale));
  // Also dump first 9 stem filter bytes (channel 0 OHWI weights).
  Serial.print("hash_dbg pass4 prep stem ch0 filter9=[");
  for (int i = 0; i < 9; ++i) {
    Serial.printf("%d", static_cast<int>(s->stem.filter[i]));
    if (i + 1 < 9) Serial.print(",");
  }
  Serial.println("]");
  // DW block 0 quant sanity dump.
  Serial.printf("hash_dbg pass4 prep dw0 ch0 mult=%ld shift=%ld bias_q32=%ld input_scale=%.6f post_scale=%.6f post_bias=%.6f codebook_scale=%.6f output_scale=%.6f\n",
                static_cast<long>(s->depthwise[0].quant.multiplier[0]),
                static_cast<long>(s->depthwise[0].quant.shift[0]),
                static_cast<long>(s->depthwise[0].quant.bias_q32[0]),
                static_cast<double>(model.activations[1].input_scale),
                static_cast<double>(model.depthwise[0].post_scale[0]),
                static_cast<double>(model.depthwise[0].post_bias[0]),
                static_cast<double>(model.depthwise[0].codebook_scale),
                static_cast<double>(model.activations[1].output_scale));
  return true;
#endif
}

void EspNnRelease(EspNnState* state) {
  if (state == nullptr) return;
  FreeConvLayer(&state->stem);
  for (int b = 0; b < state->num_blocks; ++b) {
    FreeConvLayer(&state->depthwise[b]);
    FreeConvLayer(&state->pointwise[b]);
  }
  FreeInternal(state->conv_scratch);
  FreeInternal(state->dw_scratch);
  FreeInternal(state->stem_padded_input);
  FreeInternal(state->dw_padded_input);
  FreeInternal(state);
}

bool EspNnInvokeUpToPooled(const HashDscnnModelData& model,
                           const EspNnState& state,
                           const int8_t* model_input,
                           int8_t* scratch_a,
                           int8_t* scratch_b,
                           int8_t* pooled_out,
                           float* pooled_scale_out) {
#if !HASH_KWS_ESPNN_HEADER_AVAILABLE
  (void)model; (void)state; (void)model_input;
  (void)scratch_a; (void)scratch_b; (void)pooled_out; (void)pooled_scale_out;
  return false;
#else
  if (!state.ready) return false;

  // Re-assert the esp-nn scratch buffers in case any other esp-nn consumer
  // in the firmware (e.g. TFLM kernels behind the legacy path) overwrote
  // the global pointers. Cheap; just forwarding pointers we already hold.
  if (state.conv_scratch != nullptr) {
    esp_nn_set_conv_scratch_buf(state.conv_scratch);            // _ansi
    esp_nn_set_conv_scratch_buf_esp32s3(state.conv_scratch);    // S3 (PW path)
  }
  if (state.dw_scratch != nullptr) {
    esp_nn_set_depthwise_conv_scratch_buf(state.dw_scratch);
  }

  // Block residuals are not yet wired through ESP-NN. The 91% baseline has
  // block_residual=false everywhere; for the future 93% bundle we'll need a
  // fused-add path. Refuse here so the caller can take the int-MAC fallback.
  for (int b = 0; b < model.num_blocks; ++b) {
    if (model.block_residual[b]) return false;
  }

  // C=1 input → CHW and HWC are identical.
  int rows = model.input_rows;
  int cols = model.input_cols;
  // Use the ansi library variants directly; they handle padding internally
  // so we no longer need the stem_padded_input / dw_padded_input scratch
  // buffers we allocated for the buggy S3 SIMD kernel.
  RunStemEspNn(model.stem, state.stem, model_input, rows, cols, scratch_a);
  rows = EspOutputDim(rows, model.stem.kernel_h, model.stem.stride_h, model.stem.padding_h);
  cols = EspOutputDim(cols, model.stem.kernel_w, model.stem.stride_w, model.stem.padding_w);

  // Diagnostic: post-stem channel-0 stats. HWC layout: channel 0 at indices
  // 0, OC, 2*OC, ..., (H*W-1)*OC. Sum and value at [0,0] are layout-independent
  // statistics for channel 0 — comparable to the int-MAC path's CHW dump.
  // Skip silence frames so we capture comparable inputs across both paths.
  bool esp_stem_input_has_signal = false;
  for (int i = 0; i < 64 && !esp_stem_input_has_signal; ++i) {
    if (model_input[i] > -64) esp_stem_input_has_signal = true;
  }
  static bool s_pass4_stem_dumped = false;
  if (!s_pass4_stem_dumped && esp_stem_input_has_signal) {
    s_pass4_stem_dumped = true;
    const int OC = model.stem.out_channels;
    int32_t ch0_sum = 0;
    int32_t ch0_max = -128;
    int     ch0_nz  = 0;
    for (int p = 0; p < rows * cols; ++p) {
      const int v = scratch_a[p * OC + 0];
      ch0_sum += v;
      if (v > ch0_max) ch0_max = v;
      if (v != 0) ++ch0_nz;
    }
    Serial.printf("hash_dbg pass4 stem path=esp_nn rows=%d cols=%d oc=%d ch0_at00=%d ch0_sum=%ld ch0_max=%d ch0_nonzero=%d\n",
                  rows, cols, OC, static_cast<int>(scratch_a[0]),
                  static_cast<long>(ch0_sum), static_cast<int>(ch0_max), ch0_nz);
    Serial.print("hash_dbg pass4 stem path=esp_nn model_input_first16=[");
    for (int i = 0; i < 16; ++i) {
      Serial.printf("%d", static_cast<int>(model_input[i]));
      if (i + 1 < 16) Serial.print(",");
    }
    Serial.println("]");
    // Hand-computed MAC at output (0, 0, oc=0) using the materialised ESP-NN
    // stem filter (OHWI) and direct model_input access. Mirror of the int-MAC
    // hand_mac dump — should match exactly if filter/input access are sane.
    int32_t hand_mac_esp = 0;
    for (int kh = 0; kh < model.stem.kernel_h; ++kh) {
      const int ih = 0 - model.stem.padding_h + kh;
      if (ih < 0 || ih >= model.input_rows) continue;
      for (int kw = 0; kw < model.stem.kernel_w; ++kw) {
        const int iw = 0 - model.stem.padding_w + kw;
        if (iw < 0 || iw >= model.input_cols) continue;
        // OHWI filter idx with oc=0, IC=1, ic=0:
        // ((0 * KH + kh) * KW + kw) * 1 + 0 = kh * KW + kw
        const int flt_idx = kh * model.stem.kernel_w + kw;
        int8_t w = state.stem.filter[flt_idx];
        int32_t in_v = model_input[ih * model.input_cols + iw];
        hand_mac_esp += in_v * w;
      }
    }
    Serial.printf("hash_dbg pass4 hand_mac stem path=esp_nn oc=0 pos=[0,0] mac=%ld\n",
                  static_cast<long>(hand_mac_esp));
    // Hand-MAC for DW block 0, channel 0, position (0, 0) reading the ESP-NN
    // stem output (scratch_a in HWC layout). Compares 1:1 with int-MAC's
    // dw0 hand_mac when stem outputs match.
    const auto& dw0 = model.depthwise[0];
    const int H_dw_esp = rows;
    const int W_dw_esp = cols;
    int32_t dw0_hand_mac_esp = 0;
    for (int kh = 0; kh < dw0.kernel_h; ++kh) {
      const int ih = 0 - dw0.padding_h + kh;
      if (ih < 0 || ih >= H_dw_esp) continue;
      for (int kw = 0; kw < dw0.kernel_w; ++kw) {
        const int iw = 0 - dw0.padding_w + kw;
        if (iw < 0 || iw >= W_dw_esp) continue;
        int8_t w = EspHashDepthwiseWeightInt8(dw0, 0, kh, kw);
        // HWC: scratch_a[(ih * W_dw + iw) * OC + ch], ch=0, OC=64 here.
        int32_t in_v = scratch_a[(ih * W_dw_esp + iw) * OC + 0];
        dw0_hand_mac_esp += in_v * w;
      }
    }
    Serial.printf("hash_dbg pass4 hand_mac dw0 path=esp_nn ch=0 pos=[0,0] mac=%ld\n",
                  static_cast<long>(dw0_hand_mac_esp));
    // First row of channel 0 from ESP-NN stem (HWC: scratch_a[i*OC+0]).
    Serial.print("hash_dbg pass4 stem path=esp_nn ch0_row0=[");
    for (int i = 0; i < cols; ++i) {
      Serial.printf("%d", static_cast<int>(scratch_a[i * OC + 0]));
      if (i + 1 < cols) Serial.print(",");
    }
    Serial.println("]");
  }

  for (int b = 0; b < model.num_blocks; ++b) {
    RunDepthwiseEspNn(model.depthwise[b], state.depthwise[b],
                      scratch_a, rows, cols, scratch_b);
    rows = EspOutputDim(rows, model.depthwise[b].kernel_h,
                        model.depthwise[b].stride_h,
                        model.depthwise[b].padding_h);
    cols = EspOutputDim(cols, model.depthwise[b].kernel_w,
                        model.depthwise[b].stride_w,
                        model.depthwise[b].padding_w);

    if (b == 0) {
      bool dw_has_signal = false;
      for (int i = 0; i < 64 && !dw_has_signal; ++i) {
        if (model_input[i] > -64) dw_has_signal = true;
      }
      static bool s_pass4_dw0_dumped = false;
      if (!s_pass4_dw0_dumped && dw_has_signal) {
        s_pass4_dw0_dumped = true;
        const int OC_dw = model.depthwise[0].channels;
        int32_t sum = 0; int32_t maxv = -128; int nz = 0;
        for (int p = 0; p < rows * cols; ++p) {
          const int v = scratch_b[p * OC_dw + 0];  // HWC: ch0 at p*OC.
          sum += v;
          if (v > maxv) maxv = v;
          if (v != 0) ++nz;
        }
        Serial.printf("hash_dbg pass4 dw0 path=esp_nn ch0_at00=%d ch0_sum=%ld ch0_max=%d ch0_nonzero=%d\n",
                      static_cast<int>(scratch_b[0]),
                      static_cast<long>(sum), static_cast<int>(maxv), nz);
        Serial.print("hash_dbg pass4 dw0 path=esp_nn ch0_row0=[");
        for (int i = 0; i < cols; ++i) {
          Serial.printf("%d", static_cast<int>(scratch_b[i * OC_dw + 0]));
          if (i + 1 < cols) Serial.print(",");
        }
        Serial.println("]");
      }
    }

    RunPointwiseEspNn(model.pointwise[b], state.pointwise[b],
                      scratch_b, rows, cols, scratch_a);
  }

  const float pooled_scale =
      model.activations[2 * model.num_blocks].output_scale;
  AveragePoolHwc(scratch_a, rows, cols, model.classifier.in_dim,
                 pooled_scale, pooled_out, pooled_scale);
  if (pooled_scale_out != nullptr) *pooled_scale_out = pooled_scale;
  return true;
#endif
}

bool EspNnInvoke(const HashDscnnModelData& model,
                 const EspNnState& state,
                 const int8_t* model_input,
                 int8_t* scratch_a,
                 int8_t* scratch_b,
                 int8_t* /*output_scores*/) {
  // Convenience legacy signature; not used by the runner. Real entry is
  // EspNnInvokeUpToPooled because the classifier+softmax tail belongs to
  // the runner where its float helpers already live.
  float pooled_scale = 0.0f;
  return EspNnInvokeUpToPooled(model, state, model_input,
                               scratch_a, scratch_b,
                               scratch_b, &pooled_scale);
}

}  // namespace hash_kws
