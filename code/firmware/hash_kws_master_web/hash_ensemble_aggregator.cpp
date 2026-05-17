#include "hash_ensemble_aggregator.h"

#include <math.h>
#include <string.h>

namespace hash_kws_ensemble {

namespace {

constexpr int kFixedShift = 8;  // Q8.8 fixed-point for transport-friendly scores

// Convert int8 logits -> float view, scale by 1/T if requested.
void load_logits(const Vote& v, uint8_t num_classes, const float scale, float* out) {
  for (uint8_t c = 0; c < num_classes; ++c) {
    out[c] = static_cast<float>(v.logits[c]) * scale;
  }
}

// Numerically stable softmax in place.
void softmax_in_place(float* x, uint8_t n) {
  float m = x[0];
  for (uint8_t i = 1; i < n; ++i) if (x[i] > m) m = x[i];
  float sum = 0.f;
  for (uint8_t i = 0; i < n; ++i) {
    x[i] = expf(x[i] - m);
    sum += x[i];
  }
  if (sum <= 0.f) {
    const float inv = 1.f / static_cast<float>(n);
    for (uint8_t i = 0; i < n; ++i) x[i] = inv;
    return;
  }
  const float inv_sum = 1.f / sum;
  for (uint8_t i = 0; i < n; ++i) x[i] *= inv_sum;
}

// Find top-1, top-2 from a float vector.
void top2(const float* x, uint8_t n, uint8_t* top1_idx, float* top1_val,
          uint8_t* top2_idx, float* top2_val) {
  *top1_idx = 0; *top1_val = x[0];
  *top2_idx = 0; *top2_val = -INFINITY;
  for (uint8_t i = 1; i < n; ++i) {
    if (x[i] > *top1_val) {
      *top2_val = *top1_val;
      *top2_idx = *top1_idx;
      *top1_val = x[i];
      *top1_idx = i;
    } else if (x[i] > *top2_val) {
      *top2_val = x[i];
      *top2_idx = i;
    }
  }
  if (*top2_val == -INFINITY) {
    *top2_val = *top1_val;
    *top2_idx = *top1_idx;
  }
}

inline bool slot_in_window(const Vote& v, uint32_t now_ms, uint32_t window_ms) {
  return v.has_data && (now_ms - v.host_arrival_ms) <= window_ms;
}

}  // namespace

void Aggregator::reset(uint8_t num_nodes, uint8_t num_classes, uint32_t window_ms) {
  if (num_nodes > HASH_KWS_AGG_MAX_NODES) num_nodes = HASH_KWS_AGG_MAX_NODES;
  if (num_classes > HASH_KWS_AGG_MAX_CLASSES) num_classes = HASH_KWS_AGG_MAX_CLASSES;
  num_nodes_ = num_nodes;
  num_classes_ = num_classes;
  window_ms_ = window_ms;
  mode_ = Mode::kModeMeanLogits;
  have_temperatures_ = false;
  have_learned_weights_ = false;
  noise_boost_ = 0.0f;
  num_noise_classes_ = 0;
  for (uint8_t i = 0; i < HASH_KWS_AGG_MAX_NODES; ++i) {
    temperatures_[i] = 1.f;
    learned_weights_[i] = 1.f / static_cast<float>(num_nodes_ ? num_nodes_ : 1);
    rings_[i].head = 0;
    rings_[i].count = 0;
    for (uint8_t s = 0; s < HASH_KWS_AGG_RING_DEPTH; ++s) {
      Vote& v = rings_[i].slots[s];
      v.has_data = 0;
      v.node_id = 0;
      v.source = SourceKind::kInfer;
      v.device_time_ms = 0;
      v.host_arrival_ms = 0;
      for (uint8_t c = 0; c < HASH_KWS_AGG_MAX_CLASSES; ++c) v.logits[c] = 0;
    }
  }
  for (uint8_t i = 0; i < HASH_KWS_AGG_MAX_CLASSES; ++i) {
    noise_classes_[i] = 0;
  }
}

void Aggregator::setTemperatures(const float* temperatures) {
  if (!temperatures || num_nodes_ == 0) return;
  for (uint8_t i = 0; i < num_nodes_; ++i) {
    temperatures_[i] = (temperatures[i] > 1e-3f) ? temperatures[i] : 1.f;
  }
  have_temperatures_ = true;
}

void Aggregator::setNoiseBoost(float boost) {
  noise_boost_ = boost;
}

void Aggregator::setNoiseClasses(const uint8_t* class_indices, uint8_t count) {
  if (!class_indices) {
    num_noise_classes_ = 0;
    return;
  }
  if (count > HASH_KWS_AGG_MAX_CLASSES) count = HASH_KWS_AGG_MAX_CLASSES;
  num_noise_classes_ = count;
  for (uint8_t i = 0; i < count; ++i) {
    noise_classes_[i] = class_indices[i];
  }
}

void Aggregator::setLearnedWeights(const float* weights) {
  if (!weights || num_nodes_ == 0) return;
  float sum = 0.f;
  for (uint8_t i = 0; i < num_nodes_; ++i) {
    learned_weights_[i] = weights[i] > 0.f ? weights[i] : 0.f;
    sum += learned_weights_[i];
  }
  if (sum <= 0.f) {
    const float inv = 1.f / static_cast<float>(num_nodes_);
    for (uint8_t i = 0; i < num_nodes_; ++i) learned_weights_[i] = inv;
  } else {
    const float inv_sum = 1.f / sum;
    for (uint8_t i = 0; i < num_nodes_; ++i) learned_weights_[i] *= inv_sum;
  }
  have_learned_weights_ = true;
}

bool Aggregator::submit(uint8_t node_id, SourceKind source,
                        uint32_t device_time_ms, uint32_t host_arrival_ms,
                        const int8_t* logits, uint8_t num_classes) {
  if (!logits) return false;
  if (num_classes != num_classes_) return false;
  if (node_id == 0 || node_id > num_nodes_) return false;

  NodeRing& ring = rings_[node_id - 1];
  Vote& slot = ring.slots[ring.head];
  slot.node_id = node_id;
  slot.has_data = 1;
  slot.source = source;
  slot.device_time_ms = device_time_ms;
  slot.host_arrival_ms = host_arrival_ms;
  for (uint8_t c = 0; c < num_classes_; ++c) {
    slot.logits[c] = logits[c];
  }
  ring.head = static_cast<uint8_t>((ring.head + 1) % HASH_KWS_AGG_RING_DEPTH);
  if (ring.count < HASH_KWS_AGG_RING_DEPTH) ring.count++;
  return true;
}

void Aggregator::resolve(uint32_t now_ms, Resolved* out) const {
  if (!out) return;
  out->has_decision = 0;
  out->num_voters = 0;
  out->total_in_window = 0;
  out->label = 0;
  out->score = 0;
  out->margin = 0;
  out->mode_used = mode_;
  out->window_anchor_ms = now_ms;

  if (num_nodes_ == 0 || num_classes_ == 0) return;

  // Per-node count of in-window slots. voter_count = nodes with >= 1 fresh slot.
  uint8_t per_node_count[HASH_KWS_AGG_MAX_NODES] = {0};
  uint8_t voter_count = 0;
  uint16_t total_in_window = 0;
  for (uint8_t i = 0; i < num_nodes_; ++i) {
    const NodeRing& ring = rings_[i];
    uint8_t fresh = 0;
    for (uint8_t s = 0; s < ring.count; ++s) {
      if (slot_in_window(ring.slots[s], now_ms, window_ms_)) ++fresh;
    }
    per_node_count[i] = fresh;
    if (fresh > 0) ++voter_count;
    total_in_window = static_cast<uint16_t>(total_in_window + fresh);
  }
  if (voter_count < 2) return;

  Mode effective_mode = mode_;
  if (effective_mode == Mode::kModeTemperatureScaled && !have_temperatures_) {
    effective_mode = Mode::kModeMeanLogits;
  }
  if (effective_mode == Mode::kModeLearnedWeights && !have_learned_weights_) {
    effective_mode = Mode::kModeMeanLogits;
  }
  out->mode_used = effective_mode;

  float aggregated[HASH_KWS_AGG_MAX_CLASSES] = {0};

  if (effective_mode == Mode::kModeTemperatureScaled) {
    // Per-slot softmax(logits / T_node), accumulated over EVERY in-window slot
    // across all nodes, then averaged.
    float vec[HASH_KWS_AGG_MAX_CLASSES];
    uint16_t n_used = 0;
    for (uint8_t i = 0; i < num_nodes_; ++i) {
      const NodeRing& ring = rings_[i];
      const float scale = 1.f / temperatures_[i];
      for (uint8_t s = 0; s < ring.count; ++s) {
        const Vote& v = ring.slots[s];
        if (!slot_in_window(v, now_ms, window_ms_)) continue;
        load_logits(v, num_classes_, scale, vec);
        softmax_in_place(vec, num_classes_);
        for (uint8_t c = 0; c < num_classes_; ++c) aggregated[c] += vec[c];
        ++n_used;
      }
    }
    if (n_used > 0) {
      const float inv = 1.f / static_cast<float>(n_used);
      for (uint8_t c = 0; c < num_classes_; ++c) aggregated[c] *= inv;
    }
  } else if (effective_mode == Mode::kModeLearnedWeights) {
    // First average each node's in-window slots into a per-node mean vector,
    // then take the weighted sum across nodes (using learned_weights_).
    // This keeps the "per-node weight" semantics intact while benefiting
    // from temporal averaging within each node.
    float per_node_mean[HASH_KWS_AGG_MAX_NODES][HASH_KWS_AGG_MAX_CLASSES] = {{0}};
    for (uint8_t i = 0; i < num_nodes_; ++i) {
      if (per_node_count[i] == 0) continue;
      const NodeRing& ring = rings_[i];
      for (uint8_t s = 0; s < ring.count; ++s) {
        const Vote& v = ring.slots[s];
        if (!slot_in_window(v, now_ms, window_ms_)) continue;
        for (uint8_t c = 0; c < num_classes_; ++c) {
          per_node_mean[i][c] += static_cast<float>(v.logits[c]);
        }
      }
      const float inv = 1.f / static_cast<float>(per_node_count[i]);
      for (uint8_t c = 0; c < num_classes_; ++c) per_node_mean[i][c] *= inv;
    }
    float weight_sum = 0.f;
    for (uint8_t i = 0; i < num_nodes_; ++i) {
      if (per_node_count[i] > 0) weight_sum += learned_weights_[i];
    }
    if (weight_sum <= 0.f) {
      effective_mode = Mode::kModeMeanLogits;
      out->mode_used = effective_mode;
    } else {
      const float renorm = 1.f / weight_sum;
      for (uint8_t i = 0; i < num_nodes_; ++i) {
        if (per_node_count[i] == 0) continue;
        const float w = learned_weights_[i] * renorm;
        for (uint8_t c = 0; c < num_classes_; ++c) {
          aggregated[c] += w * per_node_mean[i][c];
        }
      }
    }
  }

  if (effective_mode == Mode::kModeMeanLogits) {
    // Mean of int8 logits over EVERY in-window slot from every node — not
    // just the latest one per node. With ~5 invokes/sec and a 1.2 s window
    // this averages ~15 logit vectors instead of 3, smoothing out one-off
    // noise-driven drifts.
    for (uint8_t c = 0; c < num_classes_; ++c) aggregated[c] = 0.f;
    uint16_t n_used = 0;
    for (uint8_t i = 0; i < num_nodes_; ++i) {
      const NodeRing& ring = rings_[i];
      for (uint8_t s = 0; s < ring.count; ++s) {
        const Vote& v = ring.slots[s];
        if (!slot_in_window(v, now_ms, window_ms_)) continue;
        for (uint8_t c = 0; c < num_classes_; ++c) {
          aggregated[c] += static_cast<float>(v.logits[c]);
        }
        ++n_used;
      }
    }
    if (n_used > 0) {
      const float inv = 1.f / static_cast<float>(n_used);
      for (uint8_t c = 0; c < num_classes_; ++c) aggregated[c] *= inv;
    }
  }

  // Optional: bias toward noise classes (unknown / silence). Only meaningful
  // for logit-domain modes; temperature_scaled lives in [0..1] probabilities.
  if (num_noise_classes_ > 0 && noise_boost_ != 0.0f) {
    const bool logit_domain =
        (effective_mode == Mode::kModeMeanLogits) ||
        (effective_mode == Mode::kModeLearnedWeights);
    if (logit_domain) {
      for (uint8_t i = 0; i < num_noise_classes_; ++i) {
        const uint8_t idx = noise_classes_[i];
        if (idx < num_classes_) {
          aggregated[idx] += noise_boost_;
        }
      }
    }
  }

  uint8_t top1_idx, top2_idx;
  float top1_val, top2_val;
  top2(aggregated, num_classes_, &top1_idx, &top1_val, &top2_idx, &top2_val);

  out->has_decision = 1;
  out->num_voters = voter_count;
  out->total_in_window = static_cast<uint8_t>(total_in_window > 255 ? 255 : total_in_window);
  out->label = top1_idx;
  // Scale by 256 for integer transport while preserving sign.
  const float top1_q = top1_val * static_cast<float>(1 << kFixedShift);
  const float margin_q = (top1_val - top2_val) * static_cast<float>(1 << kFixedShift);
  out->score = static_cast<int16_t>(top1_q < -32767 ? -32767 : (top1_q > 32767 ? 32767 : top1_q));
  out->margin = static_cast<int16_t>(margin_q < 0 ? 0 : (margin_q > 32767 ? 32767 : margin_q));
}

}  // namespace hash_kws_ensemble
