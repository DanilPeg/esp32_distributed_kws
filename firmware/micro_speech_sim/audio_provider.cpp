/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "audio_provider.h"

#include <cstdlib>
#include <cstring>

// FreeRTOS.h must be included before some of the following dependencies.
// Solves b/150260343.
// clang-format off
#include "freertos/FreeRTOS.h"
// clang-format on

#include "driver/i2s.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "ringbuf.h"
#include "micro_model_settings.h"

using namespace std;

static const char* TAG = "TF_LITE_AUDIO_PROVIDER";
/* ringbuffer to hold the incoming audio data */
ringbuf_t* g_audio_capture_buffer;
volatile int32_t g_latest_audio_timestamp = 0;
/* model requires 20ms new data from g_audio_capture_buffer and 10ms old data
 * each time , storing old data in the histrory buffer , {
 * history_samples_to_keep = 10 * 16 } */
constexpr int32_t history_samples_to_keep =
    ((kFeatureSliceDurationMs - kFeatureSliceStrideMs) *
     (kAudioSampleFrequency / 1000));
/* new samples to get each time from ringbuffer, { new_samples_to_get =  20 * 16
 * } */
constexpr int32_t new_samples_to_get =
    (kFeatureSliceStrideMs * (kAudioSampleFrequency / 1000));

namespace {
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
bool g_is_audio_initialized = false;
int16_t g_history_buffer[history_samples_to_keep];
}  // namespace

#ifndef USE_FAKE_MIC
#define USE_FAKE_MIC 0
#endif

#ifndef HASH_KWS_AUDIO_CAPTURE_BUFFER_BYTES
#define HASH_KWS_AUDIO_CAPTURE_BUFFER_BYTES 131072
#endif

#ifndef HASH_KWS_AUDIO_CAPTURE_BUFFER_MIN_BYTES
#define HASH_KWS_AUDIO_CAPTURE_BUFFER_MIN_BYTES 32768
#endif

#ifndef HASH_KWS_CAPTURE_TASK_STACK_BYTES
#define HASH_KWS_CAPTURE_TASK_STACK_BYTES (8 * 1024)
#endif

const int32_t kAudioCaptureBufferSize = HASH_KWS_AUDIO_CAPTURE_BUFFER_BYTES;
// 20 ms @ 16 kHz = 320 samples
const int32_t kI2SReadSamples = 320;

static void i2s_init(void) {
  // Start listening for audio: MONO @ 16KHz
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = 16000,
      // NMP/INMP441 outputs 24-bit samples in 32-bit frames.
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = 0,
      .dma_buf_count = 3,
      .dma_buf_len = 300,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = -1,
  };
  i2s_pin_config_t pin_config = {
      .bck_io_num = 15,    // BCLK
      .ws_io_num = 16,     // WS/LRCLK
      .data_out_num = -1,  // not used
      .data_in_num = 17,   // SD/DOUT
  };
  esp_err_t ret = 0;
  ret = i2s_driver_install((i2s_port_t)1, &i2s_config, 0, NULL);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Error in i2s_driver_install");
  }
  ret = i2s_set_pin((i2s_port_t)1, &pin_config);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Error in i2s_set_pin");
  }

  ret = i2s_zero_dma_buffer((i2s_port_t)1);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Error in initializing dma buffer with 0");
  }
}

static void CaptureSamples(void* arg) {
  const size_t i2s_bytes_to_read = kI2SReadSamples * sizeof(int32_t);
  size_t bytes_read = i2s_bytes_to_read;
  int32_t i2s_read_buffer[kI2SReadSamples] = {};
  int16_t pcm16_buffer[kI2SReadSamples] = {};
  i2s_init();
  while (1) {
    /* read 100ms data at once from i2s */
    i2s_read((i2s_port_t)1, (void*)i2s_read_buffer, i2s_bytes_to_read,
             &bytes_read, pdMS_TO_TICKS(50));
    if (bytes_read <= 0) {
      ESP_LOGE(TAG, "Error in I2S read : %d", bytes_read);
    } else {
      if (bytes_read < i2s_bytes_to_read) {
        ESP_LOGW(TAG, "Partial I2S read");
      }
      const int samples_read = bytes_read / sizeof(int32_t);
      for (int i = 0; i < samples_read; ++i) {
        // Convert 24-bit left-aligned in 32-bit frame to signed 16-bit.
        pcm16_buffer[i] = (int16_t)(i2s_read_buffer[i] >> 16);
      }
      /* write 16-bit PCM into ring buffer */
      int bytes_written = rb_write(g_audio_capture_buffer,
                                   (uint8_t*)pcm16_buffer,
                                   samples_read * sizeof(int16_t),
                                   pdMS_TO_TICKS(50));
      /* update the timestamp (in ms) to let the model know that new data has
       * arrived */
      if (bytes_written > 0) {
        g_latest_audio_timestamp +=
            ((1000 * (bytes_written / 2)) / kAudioSampleFrequency);
      }
      if (bytes_written <= 0) {
        ESP_LOGE(TAG, "Could Not Write in Ring Buffer: %d ", bytes_written);
        rb_reset(g_audio_capture_buffer);
      } else if (bytes_written < (samples_read * (int)sizeof(int16_t))) {
        ESP_LOGW(TAG, "Partial Write");
      }
    }
  }
  vTaskDelete(NULL);
}

static ringbuf_t* InitAudioRingBuffer() {
  const int32_t candidates[] = {
      HASH_KWS_AUDIO_CAPTURE_BUFFER_BYTES,
      98304,
      65536,
      HASH_KWS_AUDIO_CAPTURE_BUFFER_MIN_BYTES,
  };
  int32_t previous_size = -1;
  for (int i = 0; i < 4; ++i) {
    const int32_t size = candidates[i];
    if ((size < HASH_KWS_AUDIO_CAPTURE_BUFFER_MIN_BYTES) ||
        (size == previous_size)) {
      continue;
    }
    previous_size = size;
    ESP_LOGI(TAG,
             "Trying audio ring buffer: %d bytes, free_8bit=%d, largest_8bit=%d, free_psram=%d",
             size,
             (int)heap_caps_get_free_size(MALLOC_CAP_8BIT),
             (int)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
             (int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    ringbuf_t* buffer = rb_init("tf_ringbuffer", size);
    if (buffer) {
      ESP_LOGI(TAG, "Audio ring buffer allocated: %d bytes", size);
      return buffer;
    }
  }
  return nullptr;
}

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
#if USE_FAKE_MIC
  ESP_LOGW(TAG, "USE_FAKE_MIC=1, live microphone capture is disabled");
  g_is_audio_initialized = true;
  g_latest_audio_timestamp = 0;
  return kTfLiteOk;
#else
  ESP_LOGI(TAG, "Starting live microphone capture on I2S pins BCK=15 WS=16 SD=17");
  g_audio_capture_buffer = InitAudioRingBuffer();
  if (!g_audio_capture_buffer) {
    ESP_LOGE(TAG, "Error creating ring buffer");
    return kTfLiteError;
  }
  /* create CaptureSamples Task which will get the i2s_data from mic and fill it
   * in the ring buffer */
  xTaskCreate(CaptureSamples, "CaptureSamples", HASH_KWS_CAPTURE_TASK_STACK_BYTES,
              NULL, 10, NULL);
  while (!g_latest_audio_timestamp) {
  }
  ESP_LOGI(TAG, "Audio Recording started");
  return kTfLiteOk;
#endif
}

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
#if USE_FAKE_MIC
  static int16_t g_fake_audio[kMaxAudioSampleSize] = {0};
  (void)error_reporter;
  (void)start_ms;
  (void)duration_ms;
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_fake_audio;
  return kTfLiteOk;
#else
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }
  /* copy 160 samples (320 bytes) into output_buff from history */
  memcpy((void*)(g_audio_output_buffer), (void*)(g_history_buffer),
         history_samples_to_keep * sizeof(int16_t));

  /* copy 320 samples (640 bytes) from rb at ( int16_t*(g_audio_output_buffer) +
   * 160 ), first 160 samples (320 bytes) will be from history */
  int32_t bytes_read =
      rb_read(g_audio_capture_buffer,
              ((uint8_t*)(g_audio_output_buffer + history_samples_to_keep)),
              new_samples_to_get * sizeof(int16_t), 10);
  if (bytes_read < 0) {
    ESP_LOGE(TAG, " Model Could not read data from Ring Buffer");
  } else if (bytes_read < new_samples_to_get * sizeof(int16_t)) {
    ESP_LOGD(TAG, "RB FILLED RIGHT NOW IS %d",
             rb_filled(g_audio_capture_buffer));
    ESP_LOGD(TAG, " Partial Read of Data by Model ");
    ESP_LOGV(TAG, " Could only read %d bytes when required %d bytes ",
             bytes_read, new_samples_to_get * sizeof(int16_t));
  }

  /* copy 320 bytes from output_buff into history */
  memcpy((void*)(g_history_buffer),
         (void*)(g_audio_output_buffer + new_samples_to_get),
         history_samples_to_keep * sizeof(int16_t));

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
#endif
}

int32_t LatestAudioTimestamp() {
#if USE_FAKE_MIC
  // Simulate time progression (20 ms per call).
  g_latest_audio_timestamp += kFeatureSliceStrideMs;
  return g_latest_audio_timestamp;
#else
  return g_latest_audio_timestamp;
#endif
}
