#pragma once
// ---------------------------------------------------------------------------
// Compatibility shim: tflite::ErrorReporter / tflite::MicroErrorReporter
//
// The new tflite-micro-arduino-examples library (post Nov-2022) removed these
// classes in favour of MicroPrintf().  This header re-introduces them so that
// existing code that was written against the old API compiles unchanged.
// ---------------------------------------------------------------------------

#include <cstdarg>
#include <cstdio>
#include "tensorflow/lite/c/common.h"

// Provide TF_LITE_REPORT_ERROR if the new library dropped it.
#ifndef TF_LITE_REPORT_ERROR
#define TF_LITE_REPORT_ERROR(reporter, ...) \
  do { if (reporter) (reporter)->Report(__VA_ARGS__); } while (0)
#endif

namespace tflite {

class ErrorReporter {
 public:
  virtual ~ErrorReporter() = default;
  virtual int Report(const char* format, ...) {
    va_list args;
    va_start(args, format);
    int n = vprintf(format, args);
    va_end(args);
    putchar('\n');
    return n;
  }
};

class MicroErrorReporter : public ErrorReporter {
 public:
  int Report(const char* format, ...) override {
    va_list args;
    va_start(args, format);
    int n = vprintf(format, args);
    va_end(args);
    putchar('\n');
    return n;
  }
};

}  // namespace tflite
