#ifndef DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_RECOGNIZE_COMMANDS_H_
#define DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_RECOGNIZE_COMMANDS_H_

#include <cstdint>

#include "hash_model_settings.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace hash_kws {

class PreviousResultsQueue {
 public:
  explicit PreviousResultsQueue(tflite::ErrorReporter* error_reporter)
      : error_reporter_(error_reporter), front_index_(0), size_(0) {}

  struct Result {
    Result() : time_(0), scores() {}
    Result(int32_t time, const int8_t* input_scores) : time_(time) {
      for (int i = 0; i < kCategoryCount; ++i) {
        scores[i] = input_scores[i];
      }
    }
    int32_t time_;
    int8_t scores[kCategoryCount];
  };

  int size() const { return size_; }
  int capacity() const { return kMaxResults; }
  bool empty() const { return size_ == 0; }
  Result& front() { return results_[front_index_]; }
  Result& back() {
    int back_index = front_index_ + (size_ - 1);
    if (back_index >= kMaxResults) {
      back_index -= kMaxResults;
    }
    return results_[back_index];
  }

  void push_back(const Result& entry);
  Result pop_front();
  Result& from_front(int offset);

 private:
  tflite::ErrorReporter* error_reporter_;
  static constexpr int kMaxResults = 50;
  Result results_[kMaxResults];
  int front_index_;
  int size_;
};

class HashRecognizeCommands {
 public:
  explicit HashRecognizeCommands(tflite::ErrorReporter* error_reporter,
                                 int32_t average_window_duration_ms = 1000,
                                 uint8_t detection_threshold = 180,
                                 int32_t suppression_ms = 1500,
                                 int32_t minimum_count = 3);

  int category_count() const { return kCategoryCount; }

  TfLiteStatus ProcessLatestResults(const int8_t* latest_results,
                                    int latest_results_size,
                                    int32_t current_time_ms,
                                    const char** found_command,
                                    uint8_t* score,
                                    bool* is_new_command);

 private:
  tflite::ErrorReporter* error_reporter_;
  int32_t average_window_duration_ms_;
  uint8_t detection_threshold_;
  int32_t suppression_ms_;
  int32_t minimum_count_;

  PreviousResultsQueue previous_results_;
  const char* previous_top_label_;
  int32_t previous_top_label_time_;
};

}  // namespace hash_kws

#endif  // DIPLOMA_ESP32_HASH_KWS_RUNTIME_HASH_RECOGNIZE_COMMANDS_H_
