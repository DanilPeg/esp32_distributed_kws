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
  return rc;
}
