#ifndef DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_TYPES_H_
#define DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_TYPES_H_

#include <cstddef>
#include <cstdint>

namespace hash_kws {

constexpr int kHashInputRows = 40;
constexpr int kHashInputCols = 49;
constexpr int kHashInputChannels = 1;
constexpr int kHashMaxBlocks = 4;
constexpr int kHashMaxActivationStages = 1 + (2 * kHashMaxBlocks);
constexpr int kHashMaxChannels = 128;
constexpr int kHashMaxClasses = 16;

struct HashActivationQuantParams {
  float input_scale;
  float output_scale;
};

struct HashConvLayerData {
  const int8_t* codebook;
  float codebook_scale;
  const float* post_scale;
  const float* post_bias;
  int codebook_size;
  int in_channels;
  int out_channels;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
  int layer_id;
  bool signed_hash;
};

struct HashDepthwiseLayerData {
  const int8_t* codebook;
  float codebook_scale;
  const float* post_scale;
  const float* post_bias;
  int codebook_size;
  int channels;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
  int layer_id;
  bool signed_hash;
};

struct HashLinearLayerData {
  const int8_t* codebook;
  float codebook_scale;
  const float* bias;
  int codebook_size;
  int in_dim;
  int out_dim;
  int layer_id;
  bool signed_hash;
};

struct HashDscnnModelData {
  bool available;
  int input_rows;
  int input_cols;
  int input_channels;
  int stem_out_channels;
  int num_blocks;
  int num_classes;
  HashConvLayerData stem;
  HashDepthwiseLayerData depthwise[kHashMaxBlocks];
  HashConvLayerData pointwise[kHashMaxBlocks];
  bool block_residual[kHashMaxBlocks];
  HashLinearLayerData classifier;
  HashActivationQuantParams activations[kHashMaxActivationStages];
};

}  // namespace hash_kws

#endif  // DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_MODEL_TYPES_H_
