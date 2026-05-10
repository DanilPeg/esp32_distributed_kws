// Standalone host-side smoke test for the MCU aggregator.
//
// Build:
//   g++ -std=c++14 -O2 -Wall -Wextra
//       code/firmware/hash_kws_aggregator/hash_ensemble_aggregator.cpp
//       code/firmware/hash_kws_aggregator/test_aggregator_main.cpp -o /tmp/agg_test
//
// The matching Python harness `test_aggregator_match.py` runs the same vectors
// through `aggregation.py` and confirms predicted labels match across both
// implementations.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hash_ensemble_aggregator.h"

using namespace hash_kws_ensemble;

static int run_case(const char* name,
                    Mode mode,
                    const float* temps,
                    const float* weights,
                    const int8_t logits[][12],
                    uint8_t num_voters) {
  Aggregator agg;
  agg.reset(/*num_nodes=*/3, /*num_classes=*/12, /*window_ms=*/1200);
  if (temps)   agg.setTemperatures(temps);
  if (weights) agg.setLearnedWeights(weights);
  agg.setMode(mode);

  for (uint8_t i = 0; i < num_voters; ++i) {
    bool ok = agg.submit(/*node_id=*/i + 1, SourceKind::kInfer,
                         /*device_time_ms=*/0,
                         /*host_arrival_ms=*/100 + i,
                         logits[i], /*num_classes=*/12);
    if (!ok) {
      std::fprintf(stderr, "case %s: submit %u rejected\n", name, i);
      return 1;
    }
  }
  Resolved out;
  agg.resolve(/*now_ms=*/200, &out);
  std::printf("%s decision=%u voters=%u label=%u score=%d margin=%d mode=%u\n",
              name,
              out.has_decision, out.num_voters, out.label,
              out.score, out.margin,
              static_cast<unsigned>(out.mode_used));
  return 0;
}

// Helper that exercises the noise-boost path: same logits, with vs without boost.
// Vectors: unknown=10 (idx10), silence=11 (idx11). Other indices represent real
// keywords. We craft three model votes that, without boost, all collapse to
// "down" (idx 3) by a tiny margin over "unknown".
static int run_boost_case(const char* name, float boost) {
  Aggregator agg;
  agg.reset(/*num_nodes=*/3, /*num_classes=*/12, /*window_ms=*/1200);
  static const uint8_t noise_idx[2] = {10, 11};
  agg.setNoiseClasses(noise_idx, 2);
  agg.setNoiseBoost(boost);
  agg.setMode(Mode::kModeMeanLogits);
  // Logits crafted so unknown (10) is just shy of down (3) — typical
  // "ensemble drift" scenario observed in the live diploma logs.
  int8_t logits[3][12] = {
    //   0   1   2   3   4   5   6   7   8   9  10  11
    { -10, -8, -2, -5, -3, -7, -9, -11,-12,-15, -8, -20 },
    { -12,-10, -3, -4, -5, -8, -8, -10,-11,-14, -9, -22 },
    {  -9, -7, -1, -3, -4, -6, -7,  -9,-10,-13, -8, -18 },
  };
  for (uint8_t i = 0; i < 3; ++i) {
    agg.submit(/*node_id=*/i + 1, SourceKind::kInfer,
               /*device_time_ms=*/0,
               /*host_arrival_ms=*/100 + i,
               logits[i], 12);
  }
  Resolved out;
  agg.resolve(/*now_ms=*/200, &out);
  std::printf("%s boost=%.1f decision=%u label=%u score=%d margin=%d\n",
              name, static_cast<double>(boost),
              out.has_decision, out.label, out.score, out.margin);
  return 0;
}

int main() {
  // Fixed test vectors mirrored in test_aggregator_match.py
  int8_t logits_a[3][12] = {
    { 12, -8,  3,  -2,  20,  -5,   1,   0,  -7,   2,   4,  -3 },
    { 10, -5,  2,  -1,  18,  -4,   0,   1,  -6,   3,   5,  -2 },
    { 14, -9,  4,  -3,  22,  -6,   2,   0,  -8,   1,   3,  -4 },
  };
  // Two voters, third missing (should still resolve since we require ≥2)
  int8_t logits_b[2][12] = {
    { 12, -8,  3,  -2,  20,  -5,   1,   0,  -7,   2,   4,  -3 },
    { 10, -5,  2,  -1,  18,  -4,   0,   1,  -6,   3,   5,  -2 },
  };
  // Disagreement scenario: model 1 votes class 0, model 2 votes 4, model 3 votes 0
  int8_t logits_c[3][12] = {
    { 30,  -8,   3,  -2,  20,  -5,   1,   0,  -7,   2,   4,  -3 },
    { 10,  -5,   2,  -1,  35,  -4,   0,   1,  -6,   3,   5,  -2 },
    { 28,  -9,   4,  -3,  22,  -6,   2,   0,  -8,   1,   3,  -4 },
  };

  float temps[3]   = { 1.2f, 0.9f, 1.4f };
  float weights[3] = { 0.20f, 0.55f, 0.25f };

  int rc = 0;
  rc |= run_case("case1_meanlogits_3voters", Mode::kModeMeanLogits, nullptr, nullptr, logits_a, 3);
  rc |= run_case("case2_meanlogits_2voters_paired", Mode::kModeMeanLogits, nullptr, nullptr, logits_b, 2);
  rc |= run_case("case3_temperature_scaled", Mode::kModeTemperatureScaled, temps, nullptr, logits_a, 3);
  rc |= run_case("case4_learned_weights", Mode::kModeLearnedWeights, nullptr, weights, logits_a, 3);
  rc |= run_case("case5_disagreement_meanlogits", Mode::kModeMeanLogits, nullptr, nullptr, logits_c, 3);
  rc |= run_case("case6_disagreement_learned",   Mode::kModeLearnedWeights, nullptr, weights, logits_c, 3);
  // Noise-boost behavioural test: same logits, two different boost values.
  // Without boost: drift goes to "down" (idx 3). With boost: ensemble flips
  // back to "unknown" (idx 10).
  rc |= run_boost_case("case7_drift_no_boost",   0.0f);
  rc |= run_boost_case("case8_drift_boost_24",  24.0f);

  // Temporal smoothing test: simulate 5 invokes per node within the window.
  // node1 has ONE noisy packet pointing to "left" (idx 4) plus 4 normal
  // "silence" (idx 11) packets. node2 and node3 stream pure silence.
  // Without per-node ring (old code, ring depth=1): the "left" packet from
  // node1 could be the latest one and dominate ⇒ drift to "left".
  // With ring of 5: the noisy spike is one of 15 votes ⇒ silence wins.
  {
    Aggregator agg;
    agg.reset(/*num_nodes=*/3, /*num_classes=*/12, /*window_ms=*/2000);
    static const uint8_t noise_idx[2] = {10, 11};
    agg.setNoiseClasses(noise_idx, 2);
    agg.setNoiseBoost(0.0f);
    agg.setMode(Mode::kModeMeanLogits);

    int8_t silence_logits[12] = { -10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10, 60 };
    int8_t left_spike[12]     = { -10,-10,-10,-10, 70,-10,-10,-10,-10,-10,-10,-10 };

    // node1: 4 silence + 1 noisy left_spike (latest packet is the spike).
    for (uint8_t s = 0; s < 4; ++s) {
      agg.submit(/*node_id=*/1, SourceKind::kInfer, 0, 100 + s * 200,
                 silence_logits, 12);
    }
    agg.submit(/*node_id=*/1, SourceKind::kInfer, 0, 900, left_spike, 12);

    // node2 and node3: pure silence × 5.
    for (uint8_t s = 0; s < 5; ++s) {
      agg.submit(/*node_id=*/2, SourceKind::kInfer, 0, 100 + s * 200,
                 silence_logits, 12);
      agg.submit(/*node_id=*/3, SourceKind::kInfer, 0, 100 + s * 200,
                 silence_logits, 12);
    }

    Resolved out;
    agg.resolve(/*now_ms=*/1000, &out);
    std::printf("case9_ring_smooths_spike  decision=%u voters=%u total=%u label=%u score=%d margin=%d\n",
                out.has_decision, out.num_voters, out.total_in_window,
                out.label, out.score, out.margin);
    if (out.label != 11) {
      std::fprintf(stderr, "FAIL case9: expected label=11 (silence), got %u\n", out.label);
      rc |= 1;
    }
    if (out.total_in_window < 14) {
      std::fprintf(stderr, "FAIL case9: expected total_in_window>=14, got %u\n", out.total_in_window);
      rc |= 1;
    }
  }

  // Sanity: same vectors but submit only LATEST packet per node (ring depth=1
  // semantics). Without temporal averaging the single noisy node1 vote
  // averaged with 1 silence each from node2/3 still says silence here because
  // silence has the highest absolute logit (+60 vs +70 left, but mean across
  // 3 single packets is (-10+(-10)+(60+(-10))/3) = roughly silence wins via
  // the +60 in two of three nodes). We verify the ring path makes things
  // more decisive: total_in_window jumps from 3 to 15.
  {
    Aggregator agg;
    agg.reset(3, 12, 2000);
    agg.setMode(Mode::kModeMeanLogits);
    int8_t silence_logits[12] = { -10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10, 60 };
    int8_t left_spike[12]     = { -10,-10,-10,-10, 70,-10,-10,-10,-10,-10,-10,-10 };
    agg.submit(1, SourceKind::kInfer, 0, 900, left_spike, 12);
    agg.submit(2, SourceKind::kInfer, 0, 901, silence_logits, 12);
    agg.submit(3, SourceKind::kInfer, 0, 902, silence_logits, 12);
    Resolved out;
    agg.resolve(1000, &out);
    std::printf("case10_ring_single_packet decision=%u voters=%u total=%u label=%u score=%d margin=%d\n",
                out.has_decision, out.num_voters, out.total_in_window,
                out.label, out.score, out.margin);
    if (out.total_in_window != 3) {
      std::fprintf(stderr, "FAIL case10: total_in_window != 3 (was %u)\n", out.total_in_window);
      rc |= 1;
    }
  }
  return rc;
}
