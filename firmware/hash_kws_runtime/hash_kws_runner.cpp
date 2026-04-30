#include "hash_kws_runner.h"
#include "hash_kws_espnn.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <Arduino.h>

namespace hash_kws {

namespace {

constexpr int kConvHashOc = 1337;
constexpr int kConvHashIc = 7919;
constexpr int kConvHashKh = 2971;
constexpr int kConvHashKw = 6151;
constexpr int kConvHashLayer = 104729;

constexpr int kDwHashCh = 1337;
constexpr int kDwHashKh = 7919;
constexpr int kDwHashKw = 2971;
constexpr int kDwHashLayer = 104729;

constexpr int kLinearHashA = 1337;
constexpr int kLinearHashB = 7919;
constexpr int kLinearHashC = 2971;

constexpr int kSignHashA = 4099;
constexpr int kSignHashB = 6151;
constexpr int kSignHashC = 14887;

inline int WrapPositiveMod(int value, int modulus) {
  int result = value % modulus;
  if (result < 0) {
    result += modulus;
  }
  return result;
}

inline float HashSign(bool enabled, int value) {
  if (!enabled) {
    return 1.0f;
  }
  return ((value & 1) == 0) ? -1.0f : 1.0f;
}

inline int OutputDim(int input, int kernel, int stride, int padding) {
  return ((input + (2 * padding) - kernel) / stride) + 1;
}

inline int8_t QuantizeToInt8(float value, float scale) {
  if (scale <= 0.0f) {
    scale = 1.0f;
  }
  int quantized = static_cast<int>(std::lround(value / scale));
  if (quantized < -128) {
    quantized = -128;
  }
  if (quantized > 127) {
    quantized = 127;
  }
  return static_cast<int8_t>(quantized);
}

inline float DequantizeCodebookValue(int8_t value, float scale) {
  if (scale <= 0.0f) {
    scale = 1.0f;
  }
  return static_cast<float>(value) * scale;
}

float HashWeight(const HashConvLayerData& layer,
                 int output_channel,
                 int input_channel,
                 int kernel_row,
                 int kernel_col) {
  const int raw_index =
      (output_channel * kConvHashOc) + (input_channel * kConvHashIc) +
      (kernel_row * kConvHashKh) + (kernel_col * kConvHashKw) +
      (layer.layer_id * kConvHashLayer);
  const int bucket = WrapPositiveMod(raw_index, layer.codebook_size);
  const int sign_seed =
      (output_channel * kSignHashA) + (input_channel * kSignHashB) +
      (kernel_row * kSignHashC) + (kernel_col * (kSignHashA + kSignHashB)) +
      (layer.layer_id * (kSignHashC + 11));
  return DequantizeCodebookValue(layer.codebook[bucket], layer.codebook_scale) *
         HashSign(layer.signed_hash, sign_seed);
}

float HashDepthwiseWeight(const HashDepthwiseLayerData& layer,
                          int channel,
                          int kernel_row,
                          int kernel_col) {
  const int raw_index =
      (channel * kDwHashCh) + (kernel_row * kDwHashKh) +
      (kernel_col * kDwHashKw) + (layer.layer_id * kDwHashLayer);
  const int bucket = WrapPositiveMod(raw_index, layer.codebook_size);
  const int sign_seed =
      (channel * kSignHashA) + (kernel_row * kSignHashB) +
      (kernel_col * kSignHashC) + (layer.layer_id * (kSignHashA + 29));
  return DequantizeCodebookValue(layer.codebook[bucket], layer.codebook_scale) *
         HashSign(layer.signed_hash, sign_seed);
}

float HashLinearWeight(const HashLinearLayerData& layer, int output_index, int input_index) {
  const int raw_index =
      (output_index * kLinearHashA) + (input_index * kLinearHashB) +
      (layer.layer_id * kLinearHashC);
  const int bucket = WrapPositiveMod(raw_index, layer.codebook_size);
  const int sign_seed =
      (output_index * kSignHashA) + (input_index * kSignHashB) +
      (layer.layer_id * kSignHashC);
  return DequantizeCodebookValue(layer.codebook[bucket], layer.codebook_scale) *
         HashSign(layer.signed_hash, sign_seed);
}

void FillStemKernelWeights(const HashConvLayerData& layer,
                           int output_channel,
                           float* weights_3x3) {
  int index = 0;
  for (int kernel_row = 0; kernel_row < 3; ++kernel_row) {
    for (int kernel_col = 0; kernel_col < 3; ++kernel_col) {
      weights_3x3[index++] =
          HashWeight(layer, output_channel, 0, kernel_row, kernel_col);
    }
  }
}

void FillDepthwiseKernelWeights(const HashDepthwiseLayerData& layer,
                                int channel,
                                float* weights_3x3) {
  int index = 0;
  for (int kernel_row = 0; kernel_row < 3; ++kernel_row) {
    for (int kernel_col = 0; kernel_col < 3; ++kernel_col) {
      weights_3x3[index++] =
          HashDepthwiseWeight(layer, channel, kernel_row, kernel_col);
    }
  }
}

void FillPointwiseWeights(const HashConvLayerData& layer,
                          int output_channel,
                          float* weights_1x1) {
  for (int input_channel = 0; input_channel < layer.in_channels; ++input_channel) {
    weights_1x1[input_channel] = HashWeight(layer, output_channel, input_channel, 0, 0);
  }
}


// Integer-math companion to FillPointwiseWeights. Packs the hash-derived
// weight into a signed int8, with the sign from HashSign baked in and
// clamped to [-128, 127]. The per-layer scale is just codebook_scale.
inline int8_t HashPointwiseWeightInt8(const HashConvLayerData& layer,
                                     int output_channel,
                                     int input_channel) {
  const int raw_index =
      (output_channel * kConvHashOc) + (input_channel * kConvHashIc) +
      (layer.layer_id * kConvHashLayer);
  const int bucket = WrapPositiveMod(raw_index, layer.codebook_size);
  int value = layer.codebook[bucket];
  if (layer.signed_hash) {
    const int sign_seed =
        (output_channel * kSignHashA) + (input_channel * kSignHashB) +
        (layer.layer_id * (kSignHashC + 11));
    if ((WrapPositiveMod(sign_seed, 2)) == 0) {
      value = -value;
    }
  }
  if (value < -128) value = -128;
  if (value > 127) value = 127;
  return static_cast<int8_t>(value);
}

void FillPointwiseWeightsInt8(const HashConvLayerData& layer,
                              int output_channel,
                              int8_t* weights_1x1) {
  for (int input_channel = 0; input_channel < layer.in_channels; ++input_channel) {
    weights_1x1[input_channel] =
        HashPointwiseWeightInt8(layer, output_channel, input_channel);
  }
}

void FillLinearWeights(const HashLinearLayerData& layer,
                       int output_index,
                       float* weights) {
  for (int input_index = 0; input_index < layer.in_dim; ++input_index) {
    weights[input_index] = HashLinearWeight(layer, output_index, input_index);
  }
}


// Integer-math companion for stem conv weights.
inline int8_t HashStemWeightInt8(const HashConvLayerData& layer,
                                 int output_channel,
                                 int kernel_row,
                                 int kernel_col) {
  const int raw_index =
      (output_channel * kConvHashOc) + (0 * kConvHashIc) +
      (kernel_row * kConvHashKh) + (kernel_col * kConvHashKw) +
      (layer.layer_id * kConvHashLayer);
  const int bucket = WrapPositiveMod(raw_index, layer.codebook_size);
  int value = layer.codebook[bucket];
  if (layer.signed_hash) {
    const int sign_seed =
        (output_channel * kSignHashA) + (0 * kSignHashB) +
        (kernel_row * kSignHashC) + (kernel_col * (kSignHashA + kSignHashB)) +
        (layer.layer_id * (kSignHashC + 11));
    if (WrapPositiveMod(sign_seed, 2) == 0) {
      value = -value;
    }
  }
  if (value < -128) value = -128;
  if (value > 127) value = 127;
  return static_cast<int8_t>(value);
}

void FillStemKernelWeightsInt8(const HashConvLayerData& layer,
                               int output_channel,
                               int8_t* weights_3x3) {
  int index = 0;
  for (int kernel_row = 0; kernel_row < 3; ++kernel_row) {
    for (int kernel_col = 0; kernel_col < 3; ++kernel_col) {
      weights_3x3[index++] =
          HashStemWeightInt8(layer, output_channel, kernel_row, kernel_col);
    }
  }
}

// Integer-math companion for depthwise conv weights.
inline int8_t HashDepthwiseWeightInt8(const HashDepthwiseLayerData& layer,
                                      int channel,
                                      int kernel_row,
                                      int kernel_col) {
  const int raw_index =
      (channel * kDwHashCh) + (kernel_row * kDwHashKh) +
      (kernel_col * kDwHashKw) + (layer.layer_id * kDwHashLayer);
  const int bucket = WrapPositiveMod(raw_index, layer.codebook_size);
  int value = layer.codebook[bucket];
  if (layer.signed_hash) {
    const int sign_seed =
        (channel * kSignHashA) + (kernel_row * kSignHashB) +
        (kernel_col * kSignHashC) + (layer.layer_id * (kSignHashA + 29));
    if (WrapPositiveMod(sign_seed, 2) == 0) {
      value = -value;
    }
  }
  if (value < -128) value = -128;
  if (value > 127) value = 127;
  return static_cast<int8_t>(value);
}

void FillDepthwiseKernelWeightsInt8(const HashDepthwiseLayerData& layer,
                                    int channel,
                                    int8_t* weights_3x3) {
  int index = 0;
  for (int kernel_row = 0; kernel_row < 3; ++kernel_row) {
    for (int kernel_col = 0; kernel_col < 3; ++kernel_col) {
      weights_3x3[index++] =
          HashDepthwiseWeightInt8(layer, channel, kernel_row, kernel_col);
    }
  }
}

void RunStemConv3x3(const HashConvLayerData& layer,
                    const HashActivationQuantParams& quant,
                    const int8_t* __restrict__ input,
                    int input_rows,
                    int input_cols,
                    int8_t* __restrict__ output) {
  const int output_rows =
      OutputDim(input_rows, layer.kernel_h, layer.stride_h, layer.padding_h);
  const int output_cols =
      OutputDim(input_cols, layer.kernel_w, layer.stride_w, layer.padding_w);
  const float mac_scale = quant.input_scale * layer.codebook_scale;
  for (int output_channel = 0; output_channel < layer.out_channels; ++output_channel) {
    int8_t weights_3x3[9];
    FillStemKernelWeightsInt8(layer, output_channel, weights_3x3);
    const float combined_scale = layer.post_scale[output_channel] * mac_scale;
    const float post_bias = layer.post_bias[output_channel];
    for (int output_row = 0; output_row < output_rows; ++output_row) {
      for (int output_col = 0; output_col < output_cols; ++output_col) {
        int32_t accum = 0;
        int weight_index = 0;
        for (int kernel_row = 0; kernel_row < 3; ++kernel_row) {
          const int input_row =
              (output_row * layer.stride_h) + kernel_row - layer.padding_h;
          for (int kernel_col = 0; kernel_col < 3; ++kernel_col) {
            const int input_col =
                (output_col * layer.stride_w) + kernel_col - layer.padding_w;
            if ((input_row >= 0) && (input_row < input_rows) &&
                (input_col >= 0) && (input_col < input_cols)) {
              const int input_index = (input_row * input_cols) + input_col;
              accum += static_cast<int32_t>(input[input_index]) *
                       static_cast<int32_t>(weights_3x3[weight_index]);
            }
            ++weight_index;
          }
        }
        float activated = (combined_scale * static_cast<float>(accum)) + post_bias;
        if (activated < 0.0f) {
          activated = 0.0f;
        }
        const int output_index =
            ((output_channel * output_rows) + output_row) * output_cols + output_col;
        output[output_index] = QuantizeToInt8(activated, quant.output_scale);
      }
    }
  }
}

void RunPointwiseConv1x1(const HashConvLayerData& layer,
                         const HashActivationQuantParams& quant,
                         const int8_t* __restrict__ input,
                         int input_rows,
                         int input_cols,
                         int8_t* __restrict__ output) {
  // Integer-math fast path. Uses the fact that each hash-derived weight
  // is just ±1 * codebook_scale * codebook[bucket]. We bake the sign
  // into an int8 weight and the codebook_scale into a single float
  // that multiplies the int32 accumulator once per output pixel.
  int8_t weights_1x1[kHashMaxChannels];
  const float mac_scale = quant.input_scale * layer.codebook_scale;
  for (int output_channel = 0; output_channel < layer.out_channels; ++output_channel) {
    FillPointwiseWeightsInt8(layer, output_channel, weights_1x1);
    const float combined_scale = layer.post_scale[output_channel] * mac_scale;
    const float post_bias = layer.post_bias[output_channel];
    for (int output_row = 0; output_row < input_rows; ++output_row) {
      for (int output_col = 0; output_col < input_cols; ++output_col) {
        int32_t accum = 0;
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
        for (int input_channel = 0; input_channel < layer.in_channels; ++input_channel) {
          const int input_index =
              ((input_channel * input_rows) + output_row) * input_cols + output_col;
          accum += static_cast<int32_t>(input[input_index]) *
                   static_cast<int32_t>(weights_1x1[input_channel]);
        }
        float activated = (combined_scale * static_cast<float>(accum)) + post_bias;
        if (activated < 0.0f) {
          activated = 0.0f;
        }
        const int output_index =
            ((output_channel * input_rows) + output_row) * input_cols + output_col;
        output[output_index] = QuantizeToInt8(activated, quant.output_scale);
      }
    }
  }
}

void RunPointwiseResidualConv1x1(const HashConvLayerData& layer,
                                 const HashActivationQuantParams& quant,
                                 const int8_t* __restrict__ input,
                                 int input_rows,
                                 int input_cols,
                                 const int8_t* __restrict__ residual_input,
                                 float residual_input_scale,
                                 int8_t* __restrict__ output) {
  int8_t weights_1x1[kHashMaxChannels];
  const float mac_scale = quant.input_scale * layer.codebook_scale;
  for (int output_channel = 0; output_channel < layer.out_channels; ++output_channel) {
    FillPointwiseWeightsInt8(layer, output_channel, weights_1x1);
    const float combined_scale = layer.post_scale[output_channel] * mac_scale;
    const float post_bias = layer.post_bias[output_channel];
    for (int output_row = 0; output_row < input_rows; ++output_row) {
      for (int output_col = 0; output_col < input_cols; ++output_col) {
        int32_t accum = 0;
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
        for (int input_channel = 0; input_channel < layer.in_channels; ++input_channel) {
          const int input_index =
              ((input_channel * input_rows) + output_row) * input_cols + output_col;
          accum += static_cast<int32_t>(input[input_index]) *
                   static_cast<int32_t>(weights_1x1[input_channel]);
        }
        float activated = (combined_scale * static_cast<float>(accum)) + post_bias;
        const int output_index =
            ((output_channel * input_rows) + output_row) * input_cols + output_col;
        const float residual_value =
            static_cast<float>(residual_input[output_index]) * residual_input_scale;
        activated += residual_value;
        if (activated < 0.0f) {
          activated = 0.0f;
        }
        output[output_index] = QuantizeToInt8(activated, quant.output_scale);
      }
    }
  }
}

void RunHashConv2D(const HashConvLayerData& layer,
                   const HashActivationQuantParams& quant,
                   const int8_t* input,
                   int input_rows,
                   int input_cols,
                   int8_t* output) {
  if ((layer.kernel_h == 3) && (layer.kernel_w == 3) && (layer.in_channels == 1) &&
      (layer.out_channels <= kHashMaxChannels)) {
    RunStemConv3x3(layer, quant, input, input_rows, input_cols, output);
    return;
  }
  if ((layer.kernel_h == 1) && (layer.kernel_w == 1) && (layer.in_channels <= kHashMaxChannels) &&
      (layer.out_channels <= kHashMaxChannels) && (layer.stride_h == 1) &&
      (layer.stride_w == 1) && (layer.padding_h == 0) && (layer.padding_w == 0)) {
    RunPointwiseConv1x1(layer, quant, input, input_rows, input_cols, output);
    return;
  }

  const int output_rows =
      OutputDim(input_rows, layer.kernel_h, layer.stride_h, layer.padding_h);
  const int output_cols =
      OutputDim(input_cols, layer.kernel_w, layer.stride_w, layer.padding_w);

  for (int output_channel = 0; output_channel < layer.out_channels; ++output_channel) {
    const float post_scale = layer.post_scale[output_channel];
    const float post_bias = layer.post_bias[output_channel];
    for (int output_row = 0; output_row < output_rows; ++output_row) {
      for (int output_col = 0; output_col < output_cols; ++output_col) {
        float accum = 0.0f;
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
        for (int input_channel = 0; input_channel < layer.in_channels; ++input_channel) {
          for (int kernel_row = 0; kernel_row < layer.kernel_h; ++kernel_row) {
            const int input_row =
                (output_row * layer.stride_h) + kernel_row - layer.padding_h;
            if ((input_row < 0) || (input_row >= input_rows)) {
              continue;
            }
            for (int kernel_col = 0; kernel_col < layer.kernel_w; ++kernel_col) {
              const int input_col =
                  (output_col * layer.stride_w) + kernel_col - layer.padding_w;
              if ((input_col < 0) || (input_col >= input_cols)) {
                continue;
              }
              const int input_index =
                  ((input_channel * input_rows) + input_row) * input_cols + input_col;
              const float input_value =
                  static_cast<float>(input[input_index]) * quant.input_scale;
              accum +=
                  input_value *
                  HashWeight(layer, output_channel, input_channel, kernel_row, kernel_col);
            }
          }
        }
        float activated = (post_scale * accum) + post_bias;
        if (activated < 0.0f) {
          activated = 0.0f;
        }
        const int output_index =
            ((output_channel * output_rows) + output_row) * output_cols + output_col;
        output[output_index] = QuantizeToInt8(activated, quant.output_scale);
      }
    }
  }
}

void RunHashDepthwiseConv2D(const HashDepthwiseLayerData& layer,
                            const HashActivationQuantParams& quant,
                            const int8_t* __restrict__ input,
                            int input_rows,
                            int input_cols,
                            int8_t* __restrict__ output) {
  if ((layer.kernel_h == 3) && (layer.kernel_w == 3) && (layer.channels <= kHashMaxChannels) &&
      (layer.stride_h == 1) && (layer.stride_w == 1) && (layer.padding_h == 1) &&
      (layer.padding_w == 1)) {
    const float mac_scale = quant.input_scale * layer.codebook_scale;
    for (int channel = 0; channel < layer.channels; ++channel) {
      int8_t weights_3x3[9];
      FillDepthwiseKernelWeightsInt8(layer, channel, weights_3x3);
      const float combined_scale = layer.post_scale[channel] * mac_scale;
      const float post_bias = layer.post_bias[channel];
      for (int output_row = 0; output_row < input_rows; ++output_row) {
        for (int output_col = 0; output_col < input_cols; ++output_col) {
          int32_t accum = 0;
          int weight_index = 0;
          for (int kernel_row = 0; kernel_row < 3; ++kernel_row) {
            const int input_row = output_row + kernel_row - 1;
            for (int kernel_col = 0; kernel_col < 3; ++kernel_col) {
              const int input_col = output_col + kernel_col - 1;
              if ((input_row >= 0) && (input_row < input_rows) &&
                  (input_col >= 0) && (input_col < input_cols)) {
                const int input_index =
                    ((channel * input_rows) + input_row) * input_cols + input_col;
                accum += static_cast<int32_t>(input[input_index]) *
                         static_cast<int32_t>(weights_3x3[weight_index]);
              }
              ++weight_index;
            }
          }
          float activated = (combined_scale * static_cast<float>(accum)) + post_bias;
          if (activated < 0.0f) {
            activated = 0.0f;
          }
          const int output_index =
              ((channel * input_rows) + output_row) * input_cols + output_col;
          output[output_index] = QuantizeToInt8(activated, quant.output_scale);
        }
      }
    }
    return;
  }

  // Fallback: non-3x3 or non-unit stride/padding path (unoptimised float).
  const int output_rows =
      OutputDim(input_rows, layer.kernel_h, layer.stride_h, layer.padding_h);
  const int output_cols =
      OutputDim(input_cols, layer.kernel_w, layer.stride_w, layer.padding_w);

  for (int channel = 0; channel < layer.channels; ++channel) {
    const float post_scale = layer.post_scale[channel];
    const float post_bias = layer.post_bias[channel];
    for (int output_row = 0; output_row < output_rows; ++output_row) {
      for (int output_col = 0; output_col < output_cols; ++output_col) {
        float accum = 0.0f;
        for (int kernel_row = 0; kernel_row < layer.kernel_h; ++kernel_row) {
          const int input_row =
              (output_row * layer.stride_h) + kernel_row - layer.padding_h;
          if ((input_row < 0) || (input_row >= input_rows)) continue;
          for (int kernel_col = 0; kernel_col < layer.kernel_w; ++kernel_col) {
            const int input_col =
                (output_col * layer.stride_w) + kernel_col - layer.padding_w;
            if ((input_col < 0) || (input_col >= input_cols)) continue;
            const int input_index =
                ((channel * input_rows) + input_row) * input_cols + input_col;
            const float input_value =
                static_cast<float>(input[input_index]) * quant.input_scale;
            accum += input_value * HashDepthwiseWeight(layer, channel, kernel_row, kernel_col);
          }
        }
        float activated = (post_scale * accum) + post_bias;
        if (activated < 0.0f) activated = 0.0f;
        const int output_index =
            ((channel * output_rows) + output_row) * output_cols + output_col;
        output[output_index] = QuantizeToInt8(activated, quant.output_scale);
      }
    }
  }
}

void RunHashLinear(const HashLinearLayerData& layer,
                   const int8_t* input,
                   float input_scale,
                   float* logits) {
  float weights[kHashMaxChannels];
  for (int output_index = 0; output_index < layer.out_dim; ++output_index) {
    FillLinearWeights(layer, output_index, weights);
    float accum = layer.bias[output_index];
    for (int input_index = 0; input_index < layer.in_dim; ++input_index) {
      accum += (static_cast<float>(input[input_index]) * input_scale) *
               weights[input_index];
    }
    logits[output_index] = accum;
  }
}

void AveragePoolChannels(const int8_t* input,
                         int channels,
                         int rows,
                         int cols,
                         float input_scale,
                         int8_t* pooled_output,
                         float pooled_output_scale) {
  const int spatial_size = rows * cols;
  for (int channel = 0; channel < channels; ++channel) {
    int32_t sum = 0;
    const int base_index = channel * spatial_size;
    for (int i = 0; i < spatial_size; ++i) {
      sum += static_cast<int32_t>(input[base_index + i]);
    }
    const float mean_value = (static_cast<float>(sum) / static_cast<float>(spatial_size)) * input_scale;
    pooled_output[channel] = QuantizeToInt8(mean_value, pooled_output_scale);
  }
}

void SoftmaxToCenteredInt8(const float* logits, int count, int8_t* output_scores) {
  float max_logit = logits[0];
  for (int i = 1; i < count; ++i) {
    max_logit = std::max(max_logit, logits[i]);
  }

  float sum = 0.0f;
  float probabilities[kHashMaxClasses];
  for (int i = 0; i < count; ++i) {
    probabilities[i] = std::exp(logits[i] - max_logit);
    sum += probabilities[i];
  }

  for (int i = 0; i < count; ++i) {
    const float normalized = probabilities[i] / sum;
    int quantized = static_cast<int>(std::lround(normalized * 255.0f)) - 128;
    if (quantized < -128) {
      quantized = -128;
    }
    if (quantized > 127) {
      quantized = 127;
    }
    output_scores[i] = static_cast<int8_t>(quantized);
  }
}

bool ValidateModel(const HashDscnnModelData* model) {
  if (model == nullptr) {
    return false;
  }
  if (!model->available) {
    return false;
  }
  if (model->input_channels != kHashInputChannels) {
    return false;
  }
  if ((model->num_blocks <= 0) || (model->num_blocks > kHashMaxBlocks)) {
    return false;
  }
  if ((model->stem.out_channels <= 0) || (model->stem.out_channels > kHashMaxChannels)) {
    return false;
  }
  if ((model->classifier.in_dim <= 0) || (model->classifier.in_dim > kHashMaxChannels)) {
    return false;
  }
  if ((model->num_classes <= 0) || (model->num_classes > kHashMaxClasses)) {
    return false;
  }
  for (int block = 0; block < model->num_blocks; ++block) {
    if ((model->depthwise[block].channels <= 0) ||
        (model->depthwise[block].channels > kHashMaxChannels)) {
      return false;
    }
    if ((model->pointwise[block].in_channels <= 0) ||
        (model->pointwise[block].in_channels > kHashMaxChannels) ||
        (model->pointwise[block].out_channels <= 0) ||
        (model->pointwise[block].out_channels > kHashMaxChannels)) {
      return false;
    }
    if (model->block_residual[block]) {
      if ((model->depthwise[block].stride_h != 1) || (model->depthwise[block].stride_w != 1) ||
          (model->pointwise[block].kernel_h != 1) || (model->pointwise[block].kernel_w != 1) ||
          (model->pointwise[block].stride_h != 1) || (model->pointwise[block].stride_w != 1) ||
          (model->pointwise[block].padding_h != 0) || (model->pointwise[block].padding_w != 0) ||
          (model->pointwise[block].in_channels != model->pointwise[block].out_channels) ||
          (model->depthwise[block].channels != model->pointwise[block].out_channels)) {
        return false;
      }
    }
  }
  return true;
}

size_t MaxActivationElements(const HashDscnnModelData& model) {
  int rows = model.input_rows;
  int cols = model.input_cols;
  size_t max_elements = 0;

  rows = OutputDim(rows, model.stem.kernel_h, model.stem.stride_h, model.stem.padding_h);
  cols = OutputDim(cols, model.stem.kernel_w, model.stem.stride_w, model.stem.padding_w);
  max_elements = std::max(max_elements, static_cast<size_t>(model.stem.out_channels * rows * cols));

  for (int block = 0; block < model.num_blocks; ++block) {
    rows = OutputDim(rows, model.depthwise[block].kernel_h, model.depthwise[block].stride_h, model.depthwise[block].padding_h);
    cols = OutputDim(cols, model.depthwise[block].kernel_w, model.depthwise[block].stride_w, model.depthwise[block].padding_w);
    max_elements = std::max(max_elements, static_cast<size_t>(model.depthwise[block].channels * rows * cols));

    rows = OutputDim(rows, model.pointwise[block].kernel_h, model.pointwise[block].stride_h, model.pointwise[block].padding_h);
    cols = OutputDim(cols, model.pointwise[block].kernel_w, model.pointwise[block].stride_w, model.pointwise[block].padding_w);
    max_elements = std::max(max_elements, static_cast<size_t>(model.pointwise[block].out_channels * rows * cols));
  }

  max_elements = std::max(max_elements, static_cast<size_t>(model.classifier.in_dim));
  return max_elements;
}

}  // namespace

bool HashKwsRunner::IsReady() const { return ValidateModel(model_); }

bool HashKwsRunner::Prepare() {
  if (esp_nn_state_ != nullptr) return IsEspNnReady();
  if (!IsReady()) return false;
  if (!EspNnHeaderAvailable()) return false;
  if (!EspNnPrepare(*model_, &esp_nn_state_)) {
    if (esp_nn_state_ != nullptr) {
      EspNnRelease(esp_nn_state_);
      esp_nn_state_ = nullptr;
    }
    return false;
  }
  return IsEspNnReady();
}

bool HashKwsRunner::IsEspNnReady() const {
  return (esp_nn_state_ != nullptr) && esp_nn_state_->ready;
}

int HashKwsRunner::num_classes() const {
  return (model_ != nullptr) ? model_->num_classes : 0;
}

size_t HashKwsRunner::RequiredSingleScratchBytes() const {
  if (model_ == nullptr) {
    return 0;
  }
  return MaxActivationElements(*model_) * sizeof(int8_t);
}

size_t HashKwsRunner::RequiredScratchArenaBytes() const {
  return 2 * RequiredSingleScratchBytes();
}

void HashKwsRunner::PrepareInputFromMicroFeatures(const int8_t* feature_slices,
                                                  int8_t* model_input) const {
  if ((feature_slices == nullptr) || (model_input == nullptr)) {
    return;
  }
  for (int time_index = 0; time_index < kHashInputCols; ++time_index) {
    for (int freq_index = 0; freq_index < kHashInputRows; ++freq_index) {
      const int source_index = (time_index * kHashInputRows) + freq_index;
      const int dest_index = (freq_index * kHashInputCols) + time_index;
      model_input[dest_index] = feature_slices[source_index];
    }
  }
}

bool HashKwsRunner::Invoke(const int8_t* model_input,
                           int8_t* scratch_a,
                           int8_t* scratch_b,
                           int8_t* output_scores) const {
  if (!IsReady() || (model_input == nullptr) || (scratch_a == nullptr) ||
      (scratch_b == nullptr) || (output_scores == nullptr)) {
    return false;
  }

  // ESP-NN fast path: full conv chain + AvgPool inside the SIMD module,
  // pooled int8 lands in scratch_b. Falls back to the int-MAC path below
  // if the model has unsupported layers (e.g. residual blocks today).
  if (IsEspNnReady()) {
    // Pass 4 diagnostic: run the int-MAC stem on the SAME model_input first
    // and dump channel-0 stats. scratch_b is safe to use as a temp here —
    // EspNnInvokeUpToPooled below will write to scratch_a (stem) and then
    // scratch_b (DW block 0) before AvgPool reads it. The first non-silence
    // invoke triggers it; subsequent invokes are unaffected.
    {
      bool dual_input_signal = false;
      for (int i = 0; i < 64 && !dual_input_signal; ++i) {
        if (model_input[i] > -64) dual_input_signal = true;
      }
      static bool s_pass4_dual_stem_dumped = false;
      if (!s_pass4_dual_stem_dumped && dual_input_signal) {
        s_pass4_dual_stem_dumped = true;
        RunHashConv2D(model_->stem, model_->activations[0],
                      model_input, model_->input_rows, model_->input_cols,
                      scratch_b);
        const int OH = OutputDim(model_->input_rows,
                                 model_->stem.kernel_h,
                                 model_->stem.stride_h,
                                 model_->stem.padding_h);
        const int OW = OutputDim(model_->input_cols,
                                 model_->stem.kernel_w,
                                 model_->stem.stride_w,
                                 model_->stem.padding_w);
        const int spatial = OH * OW;
        int32_t sum = 0; int32_t maxv = -128; int nz = 0;
        for (int p = 0; p < spatial; ++p) {
          const int v = scratch_b[p];
          sum += v;
          if (v > maxv) maxv = v;
          if (v != 0) ++nz;
        }
        Serial.printf("hash_dbg pass4 dual_stem int_mac_on_same_input ch0_at00=%d ch0_sum=%ld ch0_max=%d ch0_nonzero=%d\n",
                      static_cast<int>(scratch_b[0]),
                      static_cast<long>(sum),
                      static_cast<int>(maxv), nz);
        // Hand-computed MAC at output (0, 0, oc=0) via int-MAC's hash helpers
        // and direct model_input access. This is the gold-standard reference.
        int32_t hand_mac = 0;
        for (int kh = 0; kh < model_->stem.kernel_h; ++kh) {
          const int ih = 0 - model_->stem.padding_h + kh;
          if (ih < 0 || ih >= model_->input_rows) continue;
          for (int kw = 0; kw < model_->stem.kernel_w; ++kw) {
            const int iw = 0 - model_->stem.padding_w + kw;
            if (iw < 0 || iw >= model_->input_cols) continue;
            int8_t w = HashStemWeightInt8(model_->stem, 0, kh, kw);
            int32_t in_v = model_input[ih * model_->input_cols + iw];
            hand_mac += in_v * w;
          }
        }
        Serial.printf("hash_dbg pass4 hand_mac stem path=int_mac oc=0 pos=[0,0] mac=%ld\n",
                      static_cast<long>(hand_mac));
        // Hand-MAC for DW block 0, channel 0, position (0, 0) using the
        // int-MAC stem output sitting in scratch_b (CHW layout). This is
        // the gold-standard reference for what the next conv (DW) should
        // accumulate before bias / requantize.
        const auto& dw0_im = model_->depthwise[0];
        const int H_dw_im = OutputDim(model_->input_rows, model_->stem.kernel_h,
                                      model_->stem.stride_h, model_->stem.padding_h);
        const int W_dw_im = OutputDim(model_->input_cols, model_->stem.kernel_w,
                                      model_->stem.stride_w, model_->stem.padding_w);
        int32_t dw0_hand_mac_im = 0;
        for (int kh = 0; kh < dw0_im.kernel_h; ++kh) {
          const int ih = 0 - dw0_im.padding_h + kh;
          if (ih < 0 || ih >= H_dw_im) continue;
          for (int kw = 0; kw < dw0_im.kernel_w; ++kw) {
            const int iw = 0 - dw0_im.padding_w + kw;
            if (iw < 0 || iw >= W_dw_im) continue;
            int8_t w = HashDepthwiseWeightInt8(dw0_im, 0, kh, kw);
            // CHW: scratch_b[ch * H_dw * W_dw + ih * W_dw + iw], ch=0.
            int32_t in_v = scratch_b[(0 * H_dw_im + ih) * W_dw_im + iw];
            dw0_hand_mac_im += in_v * w;
          }
        }
        Serial.printf("hash_dbg pass4 hand_mac dw0 path=int_mac ch=0 pos=[0,0] mac=%ld\n",
                      static_cast<long>(dw0_hand_mac_im));
        // First row of channel 0 from int-MAC stem (CHW: scratch_b[0..W-1]).
        Serial.print("hash_dbg pass4 stem path=int_mac ch0_row0=[");
        for (int i = 0; i < W_dw_im; ++i) {
          Serial.printf("%d", static_cast<int>(scratch_b[i]));
          if (i + 1 < W_dw_im) Serial.print(",");
        }
        Serial.println("]");
        // Run int-MAC DW block 0 on scratch_b → scratch_a, dump ch0 stats.
        // DW block 0 is stride 1 padding 1 kernel 3, so output dims match input.
        RunHashDepthwiseConv2D(model_->depthwise[0], model_->activations[1],
                               scratch_b, H_dw_im, W_dw_im, scratch_a);
        const int dw_spatial = H_dw_im * W_dw_im;
        int32_t dw_sum = 0; int32_t dw_max = -128; int dw_nz = 0;
        for (int p = 0; p < dw_spatial; ++p) {
          int v = scratch_a[p];  // CHW: ch0 at scratch_a[0..spatial-1].
          dw_sum += v;
          if (v > dw_max) dw_max = v;
          if (v != 0) ++dw_nz;
        }
        Serial.printf("hash_dbg pass4 dw0 path=int_mac ch0_at00=%d ch0_sum=%ld ch0_max=%d ch0_nonzero=%d\n",
                      static_cast<int>(scratch_a[0]),
                      static_cast<long>(dw_sum), static_cast<int>(dw_max), dw_nz);
        Serial.print("hash_dbg pass4 dw0 path=int_mac ch0_row0=[");
        for (int i = 0; i < W_dw_im; ++i) {
          Serial.printf("%d", static_cast<int>(scratch_a[i]));
          if (i + 1 < W_dw_im) Serial.print(",");
        }
        Serial.println("]");
        // scratch_a will be overwritten by EspNnInvokeUpToPooled's stem next.
      }
    }

    float pooled_scale_fast = 0.0f;
    const bool fast_ok = EspNnInvokeUpToPooled(*model_, *esp_nn_state_,
                                               model_input,
                                               scratch_a, scratch_b,
                                               scratch_b,
                                               &pooled_scale_fast);
    if (fast_ok) {
      float logits_fast[kHashMaxClasses];
      RunHashLinear(model_->classifier, scratch_b, pooled_scale_fast, logits_fast);
      // Diagnostic one-shot: dump pooled int8 + logits on the first invoke
      // that has actual speech in the input (so we can compare same-input
      // captures across HASH_KWS_USE_ESP_NN=0 / =1 builds).
      bool esp_input_has_signal = false;
      for (int i = 0; i < 64 && !esp_input_has_signal; ++i) {
        if (model_input[i] > -64) esp_input_has_signal = true;
      }
      static bool s_pass4_first_invoke_dumped = false;
      if (!s_pass4_first_invoke_dumped && esp_input_has_signal) {
        s_pass4_first_invoke_dumped = true;
        Serial.printf("hash_dbg pass4 path=esp_nn pooled_scale=%.6f\n", pooled_scale_fast);
        Serial.print("hash_dbg pooled_int8=[");
        for (int i = 0; i < model_->classifier.in_dim; ++i) {
          Serial.printf("%d", static_cast<int>(scratch_b[i]));
          if (i + 1 < model_->classifier.in_dim) Serial.print(",");
        }
        Serial.println("]");
        Serial.print("hash_dbg logits=[");
        for (int i = 0; i < model_->num_classes; ++i) {
          Serial.printf("%.4f", logits_fast[i]);
          if (i + 1 < model_->num_classes) Serial.print(",");
        }
        Serial.println("]");
      }
      SoftmaxToCenteredInt8(logits_fast, model_->num_classes, output_scores);
      return true;
    }
    // else: fall through to int-MAC path below.
  }

  int rows = model_->input_rows;
  int cols = model_->input_cols;

  RunHashConv2D(model_->stem, model_->activations[0], model_input, rows, cols, scratch_a);
  // Diagnostic: post-stem channel-0 stats in CHW layout. Channel 0 lives at
  // scratch_a[0..H*W-1]. Sum/max are layout-independent so they compare 1:1
  // with the ESP-NN dump.
  // Skip silence frames so both paths can be compared on similar inputs.
  bool int_mac_input_has_signal = false;
  for (int i = 0; i < 64 && !int_mac_input_has_signal; ++i) {
    if (model_input[i] > -64) int_mac_input_has_signal = true;
  }
  {
    static bool s_pass4_stem_int_mac_dumped = false;
    if (!s_pass4_stem_int_mac_dumped && int_mac_input_has_signal) {
      s_pass4_stem_int_mac_dumped = true;
      // Dump int-MAC's stem channel-0 weights so we can directly compare
      // with the ESP-NN BuildStemFilter output (filter9 dump).
      int8_t int_mac_stem_w[9];
      FillStemKernelWeightsInt8(model_->stem, /*output_channel=*/0, int_mac_stem_w);
      Serial.print("hash_dbg pass4 stem path=int_mac ch0_filter9=[");
      for (int i = 0; i < 9; ++i) {
        Serial.printf("%d", static_cast<int>(int_mac_stem_w[i]));
        if (i + 1 < 9) Serial.print(",");
      }
      Serial.println("]");
      // Also dump first few input values for cross-check.
      Serial.print("hash_dbg pass4 stem path=int_mac model_input_first16=[");
      for (int i = 0; i < 16; ++i) {
        Serial.printf("%d", static_cast<int>(model_input[i]));
        if (i + 1 < 16) Serial.print(",");
      }
      Serial.println("]");
      const int OH = OutputDim(rows, model_->stem.kernel_h, model_->stem.stride_h, model_->stem.padding_h);
      const int OW = OutputDim(cols, model_->stem.kernel_w, model_->stem.stride_w, model_->stem.padding_w);
      const int spatial = OH * OW;
      int32_t ch0_sum = 0;
      int32_t ch0_max = -128;
      int     ch0_nz  = 0;
      for (int p = 0; p < spatial; ++p) {
        const int v = scratch_a[0 * spatial + p];
        ch0_sum += v;
        if (v > ch0_max) ch0_max = v;
        if (v != 0) ++ch0_nz;
      }
      Serial.printf("hash_dbg pass4 stem path=int_mac rows=%d cols=%d oc=%d ch0_at00=%d ch0_sum=%ld ch0_max=%d ch0_nonzero=%d\n",
                    OH, OW, model_->stem.out_channels, static_cast<int>(scratch_a[0]),
                    static_cast<long>(ch0_sum), static_cast<int>(ch0_max), ch0_nz);
    }
  }
  rows = OutputDim(rows, model_->stem.kernel_h, model_->stem.stride_h, model_->stem.padding_h);
  cols = OutputDim(cols, model_->stem.kernel_w, model_->stem.stride_w, model_->stem.padding_w);

  for (int block = 0; block < model_->num_blocks; ++block) {
    const int depthwise_stage = 1 + (2 * block);
    const int pointwise_stage = depthwise_stage + 1;
    RunHashDepthwiseConv2D(model_->depthwise[block],
                           model_->activations[depthwise_stage],
                           scratch_a,
                           rows,
                           cols,
                           scratch_b);
    rows = OutputDim(rows, model_->depthwise[block].kernel_h,
                     model_->depthwise[block].stride_h,
                     model_->depthwise[block].padding_h);
    cols = OutputDim(cols, model_->depthwise[block].kernel_w,
                     model_->depthwise[block].stride_w,
                     model_->depthwise[block].padding_w);

    if (model_->block_residual[block]) {
      RunPointwiseResidualConv1x1(model_->pointwise[block],
                                  model_->activations[pointwise_stage],
                                  scratch_b,
                                  rows,
                                  cols,
                                  scratch_a,
                                  model_->activations[depthwise_stage].input_scale,
                                  scratch_a);
    } else {
      RunHashConv2D(model_->pointwise[block],
                    model_->activations[pointwise_stage],
                    scratch_b,
                    rows,
                    cols,
                    scratch_a);
    }
    rows = OutputDim(rows, model_->pointwise[block].kernel_h,
                     model_->pointwise[block].stride_h,
                     model_->pointwise[block].padding_h);
    cols = OutputDim(cols, model_->pointwise[block].kernel_w,
                     model_->pointwise[block].stride_w,
                     model_->pointwise[block].padding_w);
  }

  const float pooled_scale = model_->activations[2 * model_->num_blocks].output_scale;
  AveragePoolChannels(scratch_a,
                      model_->classifier.in_dim,
                      rows,
                      cols,
                      pooled_scale,
                      scratch_b,
                      pooled_scale);

  float logits[kHashMaxClasses];
  RunHashLinear(model_->classifier, scratch_b, pooled_scale, logits);
  // Diagnostic one-shot mirror of the ESP-NN dump above: when the int-MAC
  // path is taken first (e.g. HASH_KWS_USE_ESP_NN=0 build) we still want a
  // pooled+logits snapshot for comparison.
  static bool s_pass4_int_mac_first_invoke_dumped = false;
  if (!s_pass4_int_mac_first_invoke_dumped) {
    s_pass4_int_mac_first_invoke_dumped = true;
    Serial.printf("hash_dbg pass4 path=int_mac pooled_scale=%.6f\n", pooled_scale);
    Serial.print("hash_dbg pooled_int8=[");
    for (int i = 0; i < model_->classifier.in_dim; ++i) {
      Serial.printf("%d", static_cast<int>(scratch_b[i]));
      if (i + 1 < model_->classifier.in_dim) Serial.print(",");
    }
    Serial.println("]");
    Serial.print("hash_dbg logits=[");
    for (int i = 0; i < model_->num_classes; ++i) {
      Serial.printf("%.4f", logits[i]);
      if (i + 1 < model_->num_classes) Serial.print(",");
    }
    Serial.println("]");
  }
  SoftmaxToCenteredInt8(logits, model_->num_classes, output_scores);
  return true;
}

}  // namespace hash_kws
