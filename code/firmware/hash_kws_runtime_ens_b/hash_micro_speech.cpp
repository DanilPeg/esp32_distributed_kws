#include <TensorFlowLite_ESP32.h>

#include <cstdint>
#include <cstdlib>
#include <esp_heap_caps.h>

#include "../micro_speech_sim/audio_provider.h"
#include "../micro_speech_sim/command_responder.h"
#include "../micro_speech_sim/feature_provider.h"
#include "../micro_speech_sim/micro_model_settings.h"
#include "hash_kws_runner.h"
#include "hash_model_data.h"
#include "hash_recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace {

tflite::ErrorReporter* error_reporter = nullptr;
FeatureProvider* feature_provider = nullptr;
hash_kws::HashKwsRunner* runner = nullptr;
hash_kws::HashRecognizeCommands* recognizer = nullptr;

int32_t previous_time = 0;
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
int8_t* scratch_a = nullptr;
int8_t* scratch_b = nullptr;
int8_t output_scores[hash_kws::kCategoryCount];

}  // namespace

void setup() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  if (!hash_kws::g_hash_model.available) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "hash_kws model is not available. Export hash_model_data.cpp first.");
    return;
  }

  static FeatureProvider static_feature_provider(kFeatureElementCount, feature_buffer);
  feature_provider = &static_feature_provider;

  static hash_kws::HashKwsRunner static_runner(&hash_kws::g_hash_model);
  runner = &static_runner;

  if (!runner->IsReady()) {
    TF_LITE_REPORT_ERROR(error_reporter, "hash_kws runtime model validation failed.");
    return;
  }

  static hash_kws::HashRecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  const size_t input_bytes =
      static_cast<size_t>(hash_kws::g_hash_model.input_rows) *
      static_cast<size_t>(hash_kws::g_hash_model.input_cols) *
      static_cast<size_t>(hash_kws::g_hash_model.input_channels);
  const size_t scratch_bytes = runner->RequiredSingleScratchBytes();

  model_input_buffer = static_cast<int8_t*>(heap_caps_malloc(input_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  scratch_a = static_cast<int8_t*>(heap_caps_malloc(scratch_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  scratch_b = static_cast<int8_t*>(heap_caps_malloc(scratch_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if ((model_input_buffer == nullptr) || (scratch_a == nullptr) || (scratch_b == nullptr)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "hash_kws allocation failed: input=%d scratch=%d",
                         static_cast<int>(input_bytes),
                         static_cast<int>(scratch_bytes));
    return;
  }

  TF_LITE_REPORT_ERROR(
      error_reporter,
      "hash_kws ready: classes=%d input=%dx%d scratch_total=%d bytes",
      runner->num_classes(), hash_kws::g_hash_model.input_rows,
      hash_kws::g_hash_model.input_cols,
      static_cast<int>(runner->RequiredScratchArenaBytes()));

  previous_time = 0;
}

void loop() {
  if ((feature_provider == nullptr) || (runner == nullptr) || (recognizer == nullptr) ||
      (model_input_buffer == nullptr) || (scratch_a == nullptr) || (scratch_b == nullptr)) {
    return;
  }

  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;
  if (how_many_new_slices == 0) {
    return;
  }

  runner->PrepareInputFromMicroFeatures(feature_buffer, model_input_buffer);
  if (!runner->Invoke(model_input_buffer, scratch_a, scratch_b, output_scores)) {
    TF_LITE_REPORT_ERROR(error_reporter, "hash_kws Invoke failed");
    return;
  }

  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output_scores, hash_kws::kCategoryCount, current_time, &found_command,
      &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "HashRecognizeCommands::ProcessLatestResults() failed");
    return;
  }

  RespondToCommand(error_reporter, current_time, found_command, score, is_new_command);
}
