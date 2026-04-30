#include "hash_recognize_commands.h"

#include <limits>

namespace hash_kws {

void PreviousResultsQueue::push_back(const Result& entry) {
  if (size() >= kMaxResults) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Couldn't push_back latest result, too many already!");
    return;
  }
  size_ += 1;
  back() = entry;
}

PreviousResultsQueue::Result PreviousResultsQueue::pop_front() {
  if (size() <= 0) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Couldn't pop_front result, none present!");
    return Result();
  }
  Result result = front();
  front_index_ += 1;
  if (front_index_ >= kMaxResults) {
    front_index_ = 0;
  }
  size_ -= 1;
  return result;
}

PreviousResultsQueue::Result& PreviousResultsQueue::from_front(int offset) {
  if ((offset < 0) || (offset >= size_)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Attempt to read beyond the end of the queue!");
    offset = size_ - 1;
  }
  int index = front_index_ + offset;
  if (index >= kMaxResults) {
    index -= kMaxResults;
  }
  return results_[index];
}

HashRecognizeCommands::HashRecognizeCommands(tflite::ErrorReporter* error_reporter,
                                             int32_t average_window_duration_ms,
                                             uint8_t detection_threshold,
                                             int32_t suppression_ms,
                                             int32_t minimum_count)
    : error_reporter_(error_reporter),
      average_window_duration_ms_(average_window_duration_ms),
      detection_threshold_(detection_threshold),
      suppression_ms_(suppression_ms),
      minimum_count_(minimum_count),
      previous_results_(error_reporter),
      previous_top_label_(kCategoryLabels[kSilenceIndex]),
      previous_top_label_time_(std::numeric_limits<int32_t>::min()) {}

TfLiteStatus HashRecognizeCommands::ProcessLatestResults(const int8_t* latest_results,
                                                         int latest_results_size,
                                                         int32_t current_time_ms,
                                                         const char** found_command,
                                                         uint8_t* score,
                                                         bool* is_new_command) {
  if (latest_results_size != kCategoryCount) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Expected %d hash-KWS scores, got %d",
                         kCategoryCount, latest_results_size);
    return kTfLiteError;
  }

  if ((!previous_results_.empty()) && (current_time_ms < previous_results_.front().time_)) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Results must be fed in increasing time order.");
    return kTfLiteError;
  }

  const int64_t time_limit = current_time_ms - average_window_duration_ms_;
  while ((!previous_results_.empty()) && (previous_results_.front().time_ <= time_limit)) {
    previous_results_.pop_front();
  }

  // In fake-mic mode timestamps can hit the window boundary exactly, so prune
  // first and keep the queue bounded before appending the newest result.
  while (previous_results_.size() >= previous_results_.capacity()) {
    previous_results_.pop_front();
  }
  previous_results_.push_back({current_time_ms, latest_results});

  const int64_t how_many_results = previous_results_.size();
  const int64_t earliest_time = previous_results_.front().time_;
  const int64_t samples_duration = current_time_ms - earliest_time;
  if ((how_many_results < minimum_count_) || (samples_duration < (average_window_duration_ms_ / 4))) {
    *found_command = previous_top_label_;
    *score = 0;
    *is_new_command = false;
    return kTfLiteOk;
  }

  int32_t average_scores[kCategoryCount];
  for (int offset = 0; offset < previous_results_.size(); ++offset) {
    PreviousResultsQueue::Result previous_result = previous_results_.from_front(offset);
    const int8_t* scores = previous_result.scores;
    for (int i = 0; i < kCategoryCount; ++i) {
      if (offset == 0) {
        average_scores[i] = scores[i] + 128;
      } else {
        average_scores[i] += scores[i] + 128;
      }
    }
  }
  for (int i = 0; i < kCategoryCount; ++i) {
    average_scores[i] /= how_many_results;
  }

  int current_top_index = 0;
  int32_t current_top_score = 0;
  for (int i = 0; i < kCategoryCount; ++i) {
    if (average_scores[i] > current_top_score) {
      current_top_score = average_scores[i];
      current_top_index = i;
    }
  }
  const char* current_top_label = kCategoryLabels[current_top_index];

  int64_t time_since_last_top;
  if ((previous_top_label_ == kCategoryLabels[kSilenceIndex]) ||
      (previous_top_label_time_ == std::numeric_limits<int32_t>::min())) {
    time_since_last_top = std::numeric_limits<int32_t>::max();
  } else {
    time_since_last_top = current_time_ms - previous_top_label_time_;
  }

  if ((current_top_score > detection_threshold_) &&
      ((current_top_label != previous_top_label_) || (time_since_last_top > suppression_ms_))) {
    previous_top_label_ = current_top_label;
    previous_top_label_time_ = current_time_ms;
    *is_new_command = true;
  } else {
    *is_new_command = false;
  }

  *found_command = current_top_label;
  *score = static_cast<uint8_t>(current_top_score);
  return kTfLiteOk;
}

}  // namespace hash_kws
