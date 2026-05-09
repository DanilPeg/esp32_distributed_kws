#include "hash_kws_runner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

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

  int rows = model_->input_rows;
  int cols = model_->input_cols;

  RunHashConv2D(model_->stem, model_->activations[0], model_input, rows, cols, scratch_a);
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
  SoftmaxToCenteredInt8(logits, model_->num_classes, output_scores);
  return true;
}

}  // namespace hash_kws
