// FRESH_MARKER: i2s mic probe sketch, updated 2026-03-30
#include <Arduino.h>
#include <math.h>
#include "driver/i2s.h"

namespace {

constexpr i2s_port_t kI2SPort = static_cast<i2s_port_t>(1);
constexpr int kSampleRate = 16000;
constexpr int kReadSamples = 320;
constexpr int kBclkPin = 15;
constexpr int kWsPin = 16;
constexpr int kDataInPin = 17;

int32_t g_raw_i2s_buffer[kReadSamples] = {};
int16_t g_pcm16_buffer[kReadSamples] = {};
uint32_t g_read_count = 0;

void InitI2S() {
  const i2s_config_t i2s_config = {
      .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = kSampleRate,
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
  const i2s_pin_config_t pin_config = {
      .bck_io_num = kBclkPin,
      .ws_io_num = kWsPin,
      .data_out_num = -1,
      .data_in_num = kDataInPin,
  };

  esp_err_t status = i2s_driver_install(kI2SPort, &i2s_config, 0, nullptr);
  if (status != ESP_OK) {
    Serial.printf("i2s_driver_install failed: %d\n", static_cast<int>(status));
    return;
  }
  status = i2s_set_pin(kI2SPort, &pin_config);
  if (status != ESP_OK) {
    Serial.printf("i2s_set_pin failed: %d\n", static_cast<int>(status));
    return;
  }
  status = i2s_zero_dma_buffer(kI2SPort);
  if (status != ESP_OK) {
    Serial.printf("i2s_zero_dma_buffer failed: %d\n", static_cast<int>(status));
    return;
  }
  Serial.printf("I2S ready: BCK=%d WS=%d SD=%d SR=%d\n",
                kBclkPin, kWsPin, kDataInPin, kSampleRate);
}

void PrintStats(const int16_t* samples, int count) {
  int32_t min_value = 32767;
  int32_t max_value = -32768;
  int64_t sum_abs = 0;
  int64_t sum_sq = 0;

  for (int i = 0; i < count; ++i) {
    const int32_t value = samples[i];
    if (value < min_value) {
      min_value = value;
    }
    if (value > max_value) {
      max_value = value;
    }
    const int32_t abs_value = value < 0 ? -value : value;
    sum_abs += abs_value;
    sum_sq += static_cast<int64_t>(value) * static_cast<int64_t>(value);
  }

  const float mean_abs = static_cast<float>(sum_abs) / static_cast<float>(count);
  const float rms = sqrtf(static_cast<float>(sum_sq) / static_cast<float>(count));

  Serial.printf("read=%lu min=%ld max=%ld mean_abs=%.1f rms=%.1f first8=",
                static_cast<unsigned long>(g_read_count),
                static_cast<long>(min_value),
                static_cast<long>(max_value),
                mean_abs,
                rms);
  for (int i = 0; i < 8 && i < count; ++i) {
    Serial.printf("%d%s", static_cast<int>(samples[i]), (i == 7 || i == count - 1) ? "" : ",");
  }
  Serial.println();
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("i2s mic probe boot");
  InitI2S();
}

void loop() {
  size_t bytes_read = 0;
  const size_t bytes_to_read = sizeof(g_raw_i2s_buffer);
  const esp_err_t status = i2s_read(
      kI2SPort,
      static_cast<void*>(g_raw_i2s_buffer),
      bytes_to_read,
      &bytes_read,
      pdMS_TO_TICKS(250));

  if (status != ESP_OK) {
    Serial.printf("i2s_read failed: %d\n", static_cast<int>(status));
    delay(250);
    return;
  }
  if (bytes_read == 0) {
    Serial.println("i2s_read returned 0 bytes");
    delay(250);
    return;
  }

  const int samples_read = static_cast<int>(bytes_read / sizeof(int32_t));
  for (int i = 0; i < samples_read; ++i) {
    g_pcm16_buffer[i] = static_cast<int16_t>(g_raw_i2s_buffer[i] >> 16);
  }

  g_read_count += 1;
  PrintStats(g_pcm16_buffer, samples_read);
  delay(200);
}
