#include <Arduino.h>
#include "driver/i2s.h"
#include <math.h>

// Adjust these to match your wiring
#define PIN_BCLK 15
#define PIN_WS   16
#define PIN_DIN  17

#define I2S_PORT I2S_NUM_0
#define I2S_STEREO_TEST 1  // 1 = read both channels to see which one has data

static void i2s_init() {
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = 16000,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_STEREO_TEST ? I2S_CHANNEL_FMT_RIGHT_LEFT
                                        : I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = 0,
      .dma_buf_count = 4,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = -1,
  };

  i2s_pin_config_t pin_config = {
      .bck_io_num = PIN_BCLK,
      .ws_io_num = PIN_WS,
      .data_out_num = -1,
      .data_in_num = PIN_DIN,
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_zero_dma_buffer(I2S_PORT);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("I2S mic test starting...");
  i2s_init();
}

void loop() {
  static int32_t samples[256];
  size_t bytes_read = 0;
  esp_err_t ret = i2s_read(I2S_PORT, samples, sizeof(samples), &bytes_read, portMAX_DELAY);
  if (ret != ESP_OK) {
    Serial.printf("i2s_read error: %d\n", (int)ret);
    delay(500);
    return;
  }

  int count = bytes_read / sizeof(int32_t);
  if (count <= 0) {
    Serial.println("no samples");
    delay(200);
    return;
  }

  // Most I2S mics output 24-bit data left-aligned in 32-bit frames.
#if I2S_STEREO_TEST
  int32_t max0 = 0, max1 = 0;
  int64_t sum0 = 0, sum1 = 0;
  int pairs = count / 2;
  for (int i = 0; i + 1 < count; i += 2) {
    int32_t v0 = (samples[i] >> 8);
    int32_t v1 = (samples[i + 1] >> 8);
    int32_t a0 = v0 >= 0 ? v0 : -v0;
    int32_t a1 = v1 >= 0 ? v1 : -v1;
    if (a0 > max0) max0 = a0;
    if (a1 > max1) max1 = a1;
    sum0 += (int64_t)v0 * (int64_t)v0;
    sum1 += (int64_t)v1 * (int64_t)v1;
  }
  float rms0 = pairs ? sqrtf((float)sum0 / (float)pairs) : 0.0f;
  float rms1 = pairs ? sqrtf((float)sum1 / (float)pairs) : 0.0f;
  Serial.printf("pairs=%d ch0 max=%ld rms=%.1f | ch1 max=%ld rms=%.1f\n",
                pairs, (long)max0, rms0, (long)max1, rms1);
#else
  int32_t max_abs = 0;
  int64_t sum_sq = 0;
  for (int i = 0; i < count; i++) {
    int32_t v = (samples[i] >> 8);
    int32_t a = v >= 0 ? v : -v;
    if (a > max_abs) max_abs = a;
    sum_sq += (int64_t)v * (int64_t)v;
  }
  float rms = sqrtf((float)sum_sq / (float)count);
  Serial.printf("samples=%d max=%ld rms=%.1f\n", count, (long)max_abs, rms);
#endif
  delay(200);
}
