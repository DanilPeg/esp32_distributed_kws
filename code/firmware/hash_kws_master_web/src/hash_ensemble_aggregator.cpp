#include "hash_ensemble_aggregator.h"

#include <math.h>
#include <string.h>

namespace hash_kws_ensemble {

namespace {

constexpr int kFixedShift = 8;  // Q8.8 fixed-point for transport-friendly scores

inline int rank_source(SourceKind s) {
  switch (s) {
    case SourceKind::kEmit:    return 2;
    case SourceKind::kEpisode: return 1;
    case SourceKind::kInfer:   return 0;
  }
  return 0;
}

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
  for (uint8_t i = 0; i < HASH_KWS_AGG_MAX_NODES; ++i) {
    temperatures_[i] = 1.f;
    learned_weights_[i] = 1.f / static_cast<float>(num_nodes_ ? num_nodes_ : 1);
    votes_[i].has_data = 0;
    votes_[i].node_id = 0;
    votes_[i].source = SourceKind::kInfer;
    votes_[i].device_time_ms = 0;
    votes_[i].host_arrival_ms = 0;
  }
}

void Aggregator::setTemperatures(const float* temperatures) {
  if (!temperatures || num_nodes_ == 0) return;
  for (uint8_t i = 0; i < num_nodes_; ++i) {
    temperatures_[i] = (temperatures[i] > 1e-3f) ? temperatures[i] : 1.f;
  }
  have_temperatures_ = true;
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

  Vote& slot = votes_[node_id - 1];
  // If we already have a fresh vote from this node within window, prefer the
  // higher-rank source (emit > episode > infer). Otherwise overwrite.
  const bool slot_fresh = slot.has_data &&
      (host_arrival_ms - slot.host_arrival_ms) <= window_ms_;
  if (slot_fresh && rank_source(source) < rank_source(slot.source)) {
    return false;
  }
  slot.node_id = node_id;
  slot.has_data = 1;
  slot.source = source;
  slot.device_time_ms = device_time_ms;
  slot.host_arrival_ms = host_arrival_ms;
  for (uint8_t c = 0; c < num_classes_; ++c) {
    slot.logits[c] = logits[c];
  }
  return true;
}

void Aggregator::resolve(uint32_t now_ms, Resolved* out) const {
  if (!out) return;
  out->has_decision = 0;
  out->num_voters = 0;
  out->label = 0;
  out->score = 0;
  out->margin = 0;
  out->mode_used = mode_;
  out->window_anchor_ms = now_ms;

  if (num_nodes_ == 0 || num_classes_ == 0) return;

  // Determine which slots fall within the window.
  bool in_window[HASH_KWS_AGG_MAX_NODES] = {false};
  uint8_t voter_count = 0;
  for (uint8_t i = 0; i < num_nodes_; ++i) {
    if (votes_[i].has_data && (now_ms - votes_[i].host_arrival_ms) <= window_ms_) {
      in_window[i] = true;
      ++voter_count;
    }
  }
  if (voter_count < 2) return;  // require at least 2 voters

  float aggregated[HASH_KWS_AGG_MAX_CLASSES] = {0};

  Mode effective_mode = mode_;
  if (effective_mode == Mode::kModeTemperatureScaled && !have_temperatures_) {
    effective_mode = Mode::kModeMeanLogits;
  }
  if (effective_mode == Mode::kModeLearnedWeights && !have_learned_weights_) {
    effective_mode = Mode::kModeMeanLogits;
  }
  out->mode_used = effective_mode;

  if (effective_mode == Mode::kModeTemperatureScaled) {
    // Mean of softmax(logits_k / T_k) — output is probabilities.
    float vec[HASH_KWS_AGG_MAX_CLASSES];
    uint8_t n_used = 0;
    for (uint8_t i = 0; i < num_nodes_; ++i) {
      if (!in_window[i]) continue;
      const float scale = 1.f / temperatures_[i];
      load_logits(votes_[i], num_classes_, scale, vec);
      softmax_in_place(vec, num_classes_);
      for (uint8_t c = 0; c < num_classes_; ++c) aggregated[c] += vec[c];
      ++n_used;
    }
    const float inv = (n_used > 0) ? (1.f / static_cast<float>(n_used)) : 0.f;
    for (uint8_t c = 0; c < num_classes_; ++c) aggregated[c] *= inv;
  } else if (effective_mode == Mode::kModeLearnedWeights) {
    // Σ w_k * logits_k — output is logits.
    float weight_sum = 0.f;
    for (uint8_t i = 0; i < num_nodes_; ++i) {
      if (in_window[i]) weight_sum += learned_weights_[i];
    }
    if (weight_sum <= 0.f) {
      effective_mode = Mode::kModeMeanLogits;
      out->mode_used = effective_mode;
    } else {
      const float renorm = 1.f / weight_sum;
      for (uint8_t i = 0; i < num_nodes_; ++i) {
        if (!in_window[i]) continue;
        const float w = learned_weights_[i] * renorm;
        for (uint8_t c = 0; c < num_classes_; ++c) {
          aggregated[c] += w * static_cast<float>(votes_[i].logits[c]);
        }
      }
    }
  }

  if (effective_mode == Mode::kModeMeanLogits) {
    // Plain mean of int8 logits — output is logits.
    uint8_t n_used = 0;
    for (uint8_t i = 0; i < num_nodes_; ++i) {
      if (!in_window[i]) continue;
      for (uint8_t c = 0; c < num_classes_; ++c) {
        aggregated[c] += static_cast<float>(votes_[i].logits[c]);
      }
      ++n_used;
    }
    const float inv = (n_used > 0) ? (1.f / static_cast<float>(n_used)) : 0.f;
    for (uint8_t c = 0; c < num_classes_; ++c) aggregated[c] *= inv;
  }

  uint8_t top1_idx, top2_idx;
  float top1_val, top2_val;
  top2(aggregated, num_classes_, &top1_idx, &top1_val, &top2_idx, &top2_val);

  out->has_decision = 1;
  out->num_voters = voter_count;
  out->label = top1_idx;
  // Scale by 256 for integer transport while preserving sign.
  const float top1_q = top1_val * static_cast<float>(1 << kFixedShift);
  const float margin_q = (top1_val - top2_val) * static_cast<float>(1 << kFixedShift);
  out->score = static_cast<int16_t>(top1_q < -32767 ? -32767 : (top1_q > 32767 ? 32767 : top1_q));
  out->margin = static_cast<int16_t>(margin_q < 0 ? 0 : (margin_q > 32767 ? 32767 : margin_q));
}

}  // namespace hash_kws_ensemble
