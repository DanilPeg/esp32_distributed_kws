// camera_classifier.ino
//
// ESP32-S3 camera node: per-frame VWW classification (person / no-person)
// using a pretrained MCUNet INT8 model from HAN Lab (mit-han-lab/mcunet),
// embedded by code/training/image_classifier/import_mcunet_vww.py.
//
// USB framing protocol -- IDENTICAL to camera_stream_serial.ino:
//
//   MAGIC : 4 bytes = A5 5A A5 5A
//   TYPE  : 1 byte  (0x01 JPEG, 0x02 UTF-8 text)
//   LEN   : 4 bytes uint32 LE
//   SEQ   : 4 bytes uint32 LE
//   BODY  : LEN bytes
//   CRC32 : 4 bytes uint32 LE  (IEEE 802.3, over TYPE..BODY)
//
// Each loop iteration:
//   1. Capture a QVGA RGB565 frame from the OV5640 (no on-board JPEG path).
//   2. Center-crop the 240x240 square and bilinear-resize straight from
//      RGB565 to VWW_INPUT_WIDTHxVWW_INPUT_HEIGHT INT8 (no intermediate
//      RGB888 buffer; saves ~250 ms vs the previous JPEG-decode pipeline).
//   3. Apply symmetric [-1, 1] normalisation (MobileNetV2-style, what MCUNet
//      was trained with), folded into the INT8 quant of the model input.
//   4. Run TFLM invoke -> 2-class logits.
//   5. Emit one text event:  kind=infer label=... score=... latency_ms=... seq=...
//
// The previous variant also forwarded the captured JPEG over USB for a live
// preview in the host dashboard. That path is removed -- preview is no longer
// needed; the kind=infer text events are enough to verify the model is firing
// (dashboard banner reads label / score from last_text_kv).
//
// Build (Arduino IDE):
//   Board:            ESP32S3 Dev Module
//   USB CDC On Boot:  Enabled
//   USB Mode:         Hardware CDC and JTAG
//   Flash:            QIO 80MHz / 16MB
//   PSRAM:            OPI PSRAM
//   Partition:        Huge APP (3MB No OTA / 1MB SPIFFS)
//   CPU Freq:         240 MHz
//
// Run host:
//   python code/scripts/camera_classifier_server.py --port COM4
//   open  http://127.0.0.1:8766/

#include <Arduino.h>
#include <math.h>
#include "esp_camera.h"
#include "esp_rom_crc.h"
#include "esp_heap_caps.h"

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_mac.h>     // esp_read_mac, ESP_MAC_WIFI_STA

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "camera_pins.h"
#include "vww_model_data.h"
#include "vww_model_meta.h"
#include "camera_link_protocol.h"

// ---------------------------------------------------------------------------
// Cluster integration: ESP-NOW trigger from master
// ---------------------------------------------------------------------------
// Default behaviour is triggered-mode: the camera sits idle until the master
// broadcasts a CameraCmdPacket (e.g. when KWS keyword "yes" fires), then
// this node captures one frame, runs inference, and unicasts the result
// back to the master's MAC. Set CAMERA_CLASSIFIER_STANDALONE_INFER=1 to fall
// back to continuous-infer-every-frame (useful for solo debug or when no
// master is on the air).

#ifndef HASH_KWS_ESPNOW_CHANNEL
#define HASH_KWS_ESPNOW_CHANNEL 1
#endif
#ifndef CAMERA_CLASSIFIER_STANDALONE_INFER
#define CAMERA_CLASSIFIER_STANDALONE_INFER 0
#endif
// In triggered mode: drain a fresh frame every this-many ms so a long idle
// followed by a trigger doesn't capture a stale buffer.
#ifndef CAMERA_CLASSIFIER_IDLE_DRAIN_MS
#define CAMERA_CLASSIFIER_IDLE_DRAIN_MS 200
#endif

// Camera joins the master's SoftAP so the radio channel is forced by the
// association — eliminates the "compile-time channel vs. real radio channel"
// drift we hit on S3 when STA reconnects to a stored AP from NVS. Must match
// the master's HASH_KWS_AP_SSID / HASH_KWS_AP_PASS in hash_kws_master_web.ino.
#ifndef HASH_KWS_AP_SSID
#define HASH_KWS_AP_SSID "esp32-hash-master"
#endif
#ifndef HASH_KWS_AP_PASS
#define HASH_KWS_AP_PASS "12345678"
#endif
#ifndef HASH_KWS_AP_JOIN_TIMEOUT_MS
#define HASH_KWS_AP_JOIN_TIMEOUT_MS 6000
#endif

// ---------------------------------------------------------------------------
// USB framing -- copied verbatim from camera_stream_serial for protocol
// compatibility with the existing host parser.
// ---------------------------------------------------------------------------

static const uint8_t MAGIC[4] = {0xA5, 0x5A, 0xA5, 0x5A};
static const uint8_t TYPE_JPEG = 0x01;
static const uint8_t TYPE_TEXT = 0x02;

static uint32_t s_seq = 0;

static void send_framed(uint8_t type, const uint8_t* body, size_t body_len,
                        uint32_t seq) {
  uint8_t hdr[13];
  memcpy(hdr, MAGIC, 4);
  hdr[4] = type;
  uint32_t len = (uint32_t)body_len;
  memcpy(hdr + 5, &len, 4);
  memcpy(hdr + 9, &seq, 4);
  Serial.write(hdr, sizeof(hdr));
  if (body_len > 0) Serial.write(body, body_len);
  uint32_t crc = esp_rom_crc32_le(0, hdr + 4, 9);
  if (body_len > 0) crc = esp_rom_crc32_le(crc, body, body_len);
  Serial.write((const uint8_t*)&crc, 4);
  // Trailing CRLF makes the raw stream human-readable in plain Serial Monitor.
  // host parser keys off the leading MAGIC bytes, so inter-frame bytes (incl.
  // \r\n) are skipped without affecting framing or CRC validation.
  Serial.write((const uint8_t*)"\r\n", 2);
}

static void send_text(const char* msg) {
  send_framed(TYPE_TEXT, (const uint8_t*)msg, strlen(msg), 0);
}

static void send_textf(const char* fmt, ...) {
  char buf[224];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0) return;
  if (n >= (int)sizeof(buf)) n = sizeof(buf) - 1;
  send_framed(TYPE_TEXT, (const uint8_t*)buf, (size_t)n, 0);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

static bool init_camera() {
  bool psram_ok = psramFound() && ESP.getPsramSize() > 0;
  send_textf("kind=psram_runtime found=%d total_kb=%u",
             (int)psram_ok, (unsigned)(ESP.getPsramSize() / 1024));
  if (!psram_ok) {
    send_text("kind=fatal reason=no_psram next=halt "
              "msg=QVGA RGB565 fb_count=2 prefers PSRAM");
    return false;
  }

  camera_config_t cfg = {};
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0       = Y2_GPIO_NUM;
  cfg.pin_d1       = Y3_GPIO_NUM;
  cfg.pin_d2       = Y4_GPIO_NUM;
  cfg.pin_d3       = Y5_GPIO_NUM;
  cfg.pin_d4       = Y6_GPIO_NUM;
  cfg.pin_d5       = Y7_GPIO_NUM;
  cfg.pin_d6       = Y8_GPIO_NUM;
  cfg.pin_d7       = Y9_GPIO_NUM;
  cfg.pin_xclk     = XCLK_GPIO_NUM;
  cfg.pin_pclk     = PCLK_GPIO_NUM;
  cfg.pin_vsync    = VSYNC_GPIO_NUM;
  cfg.pin_href     = HREF_GPIO_NUM;
  cfg.pin_sccb_sda = SIOD_GPIO_NUM;
  cfg.pin_sccb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn     = PWDN_GPIO_NUM;
  cfg.pin_reset    = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 20000000;
  // RGB565 native, no on-board JPEG encode / decode in the inference path.
  // 320*240*2 = 153600 bytes per FB; fb_count=2 → ~307 KB total, in PSRAM.
  cfg.pixel_format = PIXFORMAT_RGB565;
  cfg.grab_mode    = CAMERA_GRAB_LATEST;
  cfg.frame_size   = FRAMESIZE_QVGA;     // 320x240
  cfg.jpeg_quality = 0;                  // unused in RGB565 path
  cfg.fb_count     = 2;
  cfg.fb_location  = CAMERA_FB_IN_PSRAM;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    send_textf("kind=camera phase=init status=fail err=0x%x", err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    send_textf("kind=camera phase=init status=ok pid=0x%04x ver=0x%02x",
               s->id.PID, s->id.VER);
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_gain_ctrl(s, 1);
  }
  return true;
}

// ---------------------------------------------------------------------------
// TFLM globals
// ---------------------------------------------------------------------------

namespace {
constexpr int kArenaSize = 1024 * 1024;        // 1 MB in PSRAM, generous
uint8_t* g_tensor_arena = nullptr;

const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

// Camera FB layout (RGB565, QVGA). No separate RGB888 scratch -- preprocess
// reads RGB565 pixels directly from fb->buf during the bilinear walk.
constexpr int kSrcW = 320;
constexpr int kSrcH = 240;

// OV5640 + esp32-camera DMA writes RGB565 in big-endian within each pixel:
//   byte[0] = high byte (R[7:3] | G[7:5]), byte[1] = low byte (G[4:2] | B[7:3]).
// If a future sensor reconfig flips this, set to 0 and recompile.
#define CAMERA_RGB565_BIG_ENDIAN 1
}  // namespace

static bool init_tflm() {
  if (vww_model_data_len < 1024) {
    send_textf("kind=fatal reason=no_model model_len=%u next=halt "
               "msg=run code/training/image_classifier/export_vww_tflite.py",
               (unsigned)vww_model_data_len);
    return false;
  }

  g_model = tflite::GetModel(vww_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    send_textf("kind=fatal reason=schema_mismatch model_ver=%lu expected=%lu",
               (unsigned long)g_model->version(),
               (unsigned long)TFLITE_SCHEMA_VERSION);
    return false;
  }

  g_tensor_arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM);
  if (!g_tensor_arena) {
    send_text("kind=fatal reason=arena_alloc next=halt");
    return false;
  }

  // MCUNet VWW op set (verified by introspecting the tflite from HAN Lab):
  //   CONV_2D, DEPTHWISE_CONV_2D, ADD, PAD, AVERAGE_POOL_2D, RESHAPE.
  // No HardSwish / Mean / Softmax / Resize / FullyConnected -- model emits
  // raw 2-class int8 logits, host code does the softmax.
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddPad();
  resolver.AddAveragePool2D();
  resolver.AddReshape();

  static tflite::MicroInterpreter static_interpreter(
      g_model, resolver, g_tensor_arena, kArenaSize);
  g_interpreter = &static_interpreter;

  TfLiteStatus alloc = g_interpreter->AllocateTensors();
  if (alloc != kTfLiteOk) {
    send_textf("kind=fatal reason=allocate_tensors status=%d", (int)alloc);
    return false;
  }
  g_input  = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  size_t used = g_interpreter->arena_used_bytes();
  send_textf("kind=tflm phase=init status=ok arena_used_kb=%u model_kb=%u "
             "in_shape=%dx%dx%d in_type=%d out_shape=%dx%d out_type=%d",
             (unsigned)(used / 1024),
             (unsigned)(vww_model_data_len / 1024),
             g_input->dims->data[1], g_input->dims->data[2],
             g_input->dims->data[3], (int)g_input->type,
             g_output->dims->data[0], g_output->dims->data[1],
             (int)g_output->type);
  return true;
}

// ---------------------------------------------------------------------------
// Preprocess: 320x240 RGB565 -> VWW_INPUT_HEIGHT x VWW_INPUT_WIDTH INT8
// MCUNet uses MobileNetV2-style symmetric normalisation -- no ImageNet
// mean/std. Folded into the model's INT8 input quantisation:
//     x_norm = px / 127.5 - 1                     (target tensor value)
//     q      = round(x_norm / in_scale + in_zp)   (int8 storage)
//          = round(px * (1 / (127.5 * in_scale))
//                  + (-1 / in_scale + in_zp))
// With HAN Lab's published quant (in_scale = 1/127.5, in_zp = -1) the slope
// collapses to ~1.0 and the bias to ~-128.5, which is just "byte minus 128".
//
// Bilinear walk: center-crop 240x240 out of 320x240, then resize to model
// input. We unpack each touched RGB565 sample to RGB888 on the fly -- there
// is no precomputed RGB888 buffer in this pipeline (saves ~250 ms vs the
// previous JPEG-decode path). Only the (up to) 4 source pixels per dst pixel
// are unpacked, vs decoding the full 153 KB frame upfront.
// ---------------------------------------------------------------------------

static float g_inv[3];
static float g_bias[3];

static void init_preproc(float in_scale, int in_zp) {
  const float inv  = 1.0f / (127.5f * in_scale);
  const float bias = -1.0f / in_scale + (float)in_zp;
  for (int c = 0; c < 3; ++c) {
    g_inv[c]  = inv;
    g_bias[c] = bias;
  }
}

// Read one RGB565 pixel out of the camera fb and expand to 8-bit RGB888.
// Returns into r/g/b passed by pointer. Low bits dropped (no replication --
// the model doesn't care about <2 LSBs once symmetric-quantised to int8).
static inline void rgb565_at(const uint8_t* src, int sy, int sx,
                             uint8_t* r, uint8_t* g, uint8_t* b) {
  const uint8_t* p = src + (sy * kSrcW + sx) * 2;
#if CAMERA_RGB565_BIG_ENDIAN
  const uint16_t px = ((uint16_t)p[0] << 8) | (uint16_t)p[1];
#else
  const uint16_t px = ((uint16_t)p[1] << 8) | (uint16_t)p[0];
#endif
  *r = (uint8_t)((px >> 11) & 0x1F) << 3;
  *g = (uint8_t)((px >>  5) & 0x3F) << 2;
  *b = (uint8_t)( px        & 0x1F) << 3;
}

static void preprocess_into_input(const uint8_t* src_rgb565, int8_t* dst_q) {
  const int crop_w = 240;                          // square center crop
  const int x_off = (kSrcW - crop_w) / 2;          // = 40 (320 → 240)
  const int y_off = 0;
  const float scale = (float)crop_w / (float)VWW_INPUT_WIDTH;

  for (int y = 0; y < VWW_INPUT_HEIGHT; ++y) {
    float sy_f = (y + 0.5f) * scale - 0.5f;
    int   sy0  = (int)floorf(sy_f);
    float wy   = sy_f - sy0;
    if (sy0 < 0)            { sy0 = 0;            wy = 0.0f; }
    if (sy0 >= crop_w - 1)  { sy0 = crop_w - 2;   wy = 1.0f; }
    int sy1 = sy0 + 1;

    for (int x = 0; x < VWW_INPUT_WIDTH; ++x) {
      float sx_f = (x + 0.5f) * scale - 0.5f;
      int   sx0  = (int)floorf(sx_f);
      float wx   = sx_f - sx0;
      if (sx0 < 0)           { sx0 = 0;          wx = 0.0f; }
      if (sx0 >= crop_w - 1) { sx0 = crop_w - 2; wx = 1.0f; }
      int sx1 = sx0 + 1;

      uint8_t r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11;
      rgb565_at(src_rgb565, y_off + sy0, x_off + sx0, &r00, &g00, &b00);
      rgb565_at(src_rgb565, y_off + sy0, x_off + sx1, &r01, &g01, &b01);
      rgb565_at(src_rgb565, y_off + sy1, x_off + sx0, &r10, &g10, &b10);
      rgb565_at(src_rgb565, y_off + sy1, x_off + sx1, &r11, &g11, &b11);

      const float w00 = (1.0f - wy) * (1.0f - wx);
      const float w01 = (1.0f - wy) * wx;
      const float w10 = wy          * (1.0f - wx);
      const float w11 = wy          * wx;

      const float rgb[3] = {
          r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11,
          g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11,
          b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11,
      };
      for (int c = 0; c < 3; ++c) {
        const float q = rgb[c] * g_inv[c] + g_bias[c];
        int qi = (int)lrintf(q);
        if (qi < -128) qi = -128; else if (qi > 127) qi = 127;
        dst_q[(y * VWW_INPUT_WIDTH + x) * 3 + c] = (int8_t)qi;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// One inference cycle on the current camera frame.
// Returns true on success and fills label_out / score_out / latency_ms_out.
// ---------------------------------------------------------------------------

static bool run_inference(camera_fb_t* fb, const char** label_out,
                          float* score_out, uint32_t* latency_ms_out) {
  uint32_t t0 = millis();

  // RGB565 native -- no decode stage. Bilinear walks fb->buf directly.
  preprocess_into_input(fb->buf, g_input->data.int8);
  uint32_t t_pre = millis();

  TfLiteStatus invoke = g_interpreter->Invoke();
  if (invoke != kTfLiteOk) {
    send_textf("kind=infer phase=invoke status=%d", (int)invoke);
    return false;
  }
  uint32_t t_invoke = millis();

  // 2-class output -> softmax -> person probability.
  const float oscale = (float)VWW_OUTPUT_SCALE;
  const int   ozp    = VWW_OUTPUT_ZERO_POINT;
  float l0 = ((int)g_output->data.int8[0] - ozp) * oscale;
  float l1 = ((int)g_output->data.int8[1] - ozp) * oscale;
  float m  = l0 > l1 ? l0 : l1;
  float e0 = expf(l0 - m), e1 = expf(l1 - m);
  float p_person = e1 / (e0 + e1);

  int top = (l1 > l0) ? 1 : 0;
  *label_out = (top == 1) ? VWW_LABEL_1 : VWW_LABEL_0;
  *score_out = (top == 1) ? p_person : (1.0f - p_person);
  *latency_ms_out = t_invoke - t0;

  // One verbose timing line so we can see where time goes; cheap to leave on.
  // decode_ms is intentionally absent now -- the RGB565 native path has no
  // standalone decode stage; preprocess walks fb->buf directly.
  send_textf("kind=infer_timing pre_ms=%lu invoke_ms=%lu "
             "logit0=%.3f logit1=%.3f",
             (unsigned long)(t_pre - t0),
             (unsigned long)(t_invoke - t_pre),
             l0, l1);
  return true;
}

// ---------------------------------------------------------------------------
// ESP-NOW link: receive trigger from master, send reply back
// ---------------------------------------------------------------------------

namespace {

struct PendingTrigger {
  volatile bool      have;
  uint16_t           trigger_id;
  uint8_t            trigger_label;
  uint32_t           t_ms_master;
  uint8_t            master_mac[6];
  uint32_t           received_at_ms;
};
PendingTrigger g_pending = {};

uint32_t g_packets_seen        = 0;  // any ESP-NOW callback fire, any length
uint32_t g_triggers_received   = 0;  // CameraCmdPacket validated
uint32_t g_triggers_rejected   = 0;  // cmd-sized but failed magic/crc/kind
uint32_t g_inferences_done     = 0;
uint32_t g_replies_sent        = 0;
uint32_t g_replies_failed      = 0;
uint8_t  g_last_master_mac[6]  = {0};
bool     g_broadcast_peer_added = false;
const uint8_t kBroadcastMac[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

bool MacEq(const uint8_t* a, const uint8_t* b) {
  for (int i = 0; i < 6; ++i) if (a[i] != b[i]) return false;
  return true;
}

bool MacZero(const uint8_t* a) {
  for (int i = 0; i < 6; ++i) if (a[i] != 0) return false;
  return true;
}

void EnsurePeer(const uint8_t* mac) {
  if (MacZero(mac)) return;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, mac, 6);
  peer.channel = HASH_KWS_ESPNOW_CHANNEL;
  peer.ifidx   = WIFI_IF_STA;
  peer.encrypt = false;
  if (!esp_now_is_peer_exist(mac)) {
    esp_err_t e = esp_now_add_peer(&peer);
    if (e != ESP_OK) {
      send_textf("kind=espnow phase=add_peer status=fail err=0x%x", (unsigned)e);
    }
  }
}

void EnsureBroadcastPeer() {
  if (g_broadcast_peer_added) return;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, kBroadcastMac, 6);
  peer.channel = HASH_KWS_ESPNOW_CHANNEL;
  peer.ifidx   = WIFI_IF_STA;
  peer.encrypt = false;
  esp_err_t e = esp_now_add_peer(&peer);
  if (e == ESP_OK || e == ESP_ERR_ESPNOW_EXIST) {
    g_broadcast_peer_added = true;
  } else {
    send_textf("kind=espnow phase=add_bcast status=fail err=0x%x", (unsigned)e);
  }
}

void SendStatusBroadcast() {
  EnsureBroadcastPeer();
  CameraStatusPacket p = {};
  p.channel             = HASH_KWS_ESPNOW_CHANNEL;
  p.uptime_ms           = millis();
  p.packets_seen        = g_packets_seen;
  p.triggers_received   = (uint16_t)g_triggers_received;
  p.triggers_rejected   = (uint16_t)g_triggers_rejected;
  p.inferences_done     = (uint16_t)g_inferences_done;
  p.replies_sent        = (uint16_t)g_replies_sent;
  p.replies_failed      = (uint16_t)g_replies_failed;
  p.free_heap_kb        = (uint16_t)(ESP.getFreeHeap() / 1024);
  p.psram_free_kb       = (uint16_t)(ESP.getFreePsram() / 1024);
  CameraLinkSignStatus(&p);
  esp_now_send(kBroadcastMac,
               reinterpret_cast<const uint8_t*>(&p), sizeof(p));
}

#if ESP_ARDUINO_VERSION_MAJOR >= 3
void OnEspNowRecv(const esp_now_recv_info_t* info,
                  const uint8_t* data, int len) {
  const uint8_t* src_mac = info ? info->src_addr : nullptr;
#else
void OnEspNowRecv(const uint8_t* src_mac, const uint8_t* data, int len) {
#endif
  g_packets_seen++;
  // Diagnostic: own status echoes are visible to us too (broadcast); ignore
  // any packet that's not the cmd size.
  if (len != static_cast<int>(sizeof(CameraCmdPacket))) return;
  CameraCmdPacket p;
  memcpy(&p, data, sizeof(p));
  if (!CameraLinkValidateCmd(p)) {
    g_triggers_rejected++;
    return;
  }

  // Latch trigger for the loop to pick up. If a previous trigger is still
  // unhandled we overwrite -- newest wins, that's the right semantic for
  // a "fire on keyword" pipeline (don't queue stale snapshots).
  g_pending.trigger_id    = p.trigger_id;
  g_pending.trigger_label = p.trigger_label;
  g_pending.t_ms_master   = p.t_ms_master;
  if (src_mac) memcpy(g_pending.master_mac, src_mac, 6);
  g_pending.received_at_ms = millis();
  g_pending.have = true;
  g_triggers_received++;

  // Immediate status echo so the master sees "yes, I got it" within ~50ms.
  SendStatusBroadcast();
}

bool SendReply(const PendingTrigger& trig, bool ok,
               uint8_t label, uint8_t score_q8,
               uint16_t fb_ms, uint16_t invoke_ms, uint16_t latency_ms) {
  EnsurePeer(trig.master_mac);
  CameraReplyPacket r = {};
  r.kind         = ok ? kCamRepKindInferDone : kCamRepKindInferFail;
  r.trigger_id   = trig.trigger_id;
  r.t_ms_camera  = millis();
  r.label        = label;
  r.score_q8     = score_q8;
  r.latency_ms   = latency_ms;
  r.fb_ms        = fb_ms;
  r.invoke_ms    = invoke_ms;
  CameraLinkSignReply(&r);
  esp_err_t e = esp_now_send(trig.master_mac,
                             reinterpret_cast<const uint8_t*>(&r), sizeof(r));
  if (e != ESP_OK) {
    g_replies_failed++;
    send_textf("kind=espnow phase=tx status=fail err=0x%x trigger=%u",
               (unsigned)e, (unsigned)trig.trigger_id);
    return false;
  }
  g_replies_sent++;
  return true;
}

bool BringUpEspNowLink() {
  // Don't pull stored WiFi creds from NVS on next mode change — that's how
  // a previously-flashed test sketch's WiFi association sneaks back and
  // drags the radio off our intended channel.
  WiFi.persistent(false);

  WiFi.mode(WIFI_STA);
  // DO NOT call WiFi.disconnect(false, true) here — the second arg=true
  // erases stored WiFi creds and (observed on ESP32-S3) leaves the radio
  // in a state where WiFi.macAddress() returns 00:00:00:00:00:00 even
  // though esp_now_init() reports OK. ESP-NOW then silently fails to
  // deliver because peers are matched against zero MAC.
  WiFi.disconnect(false);   // drop any prior association, KEEP stored creds
  WiFi.setSleep(false);
  delay(100);               // STA radio actually live before begin()

  // Associate with the master's SoftAP. The association forces the radio
  // onto the AP's channel for the lifetime of the link, so ESP-NOW peers
  // are guaranteed to be on the same channel without any manual
  // esp_wifi_set_channel that NVS auto-reconnect could later undo.
  WiFi.begin(HASH_KWS_AP_SSID, HASH_KWS_AP_PASS);
  send_textf("kind=wifi phase=join ssid=%s timeout_ms=%d",
             HASH_KWS_AP_SSID, (int)HASH_KWS_AP_JOIN_TIMEOUT_MS);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED &&
         (millis() - t0) < HASH_KWS_AP_JOIN_TIMEOUT_MS) {
    delay(100);
  }
  const bool joined = (WiFi.status() == WL_CONNECTED);
  if (joined) {
    send_textf("kind=wifi phase=joined ssid=%s rssi=%d channel=%u ip=%s",
               HASH_KWS_AP_SSID, WiFi.RSSI(), WiFi.channel(),
               WiFi.localIP().toString().c_str());
  } else {
    // Master AP unreachable — keep going, but pin the channel manually so
    // ESP-NOW at least has a chance on the compile-time channel.
    send_textf("kind=wifi phase=join_timeout ssid=%s "
               "fallback=manual_set_channel channel=%d",
               HASH_KWS_AP_SSID, HASH_KWS_ESPNOW_CHANNEL);
    esp_wifi_set_channel(HASH_KWS_ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  }

  if (esp_now_init() != ESP_OK) {
    send_text("kind=espnow phase=init status=fail node=camera");
    return false;
  }
  esp_now_register_recv_cb(OnEspNowRecv);
  EnsureBroadcastPeer();

  // Read the *actual* radio channel — when joined it's whatever the AP
  // told us, when fallback it's HASH_KWS_ESPNOW_CHANNEL. Logging the real
  // value (not the compile-time constant) was the missing diagnostic that
  // hid this whole class of channel-drift bugs.
  uint8_t pri_ch = 0; wifi_second_chan_t sec_ch = WIFI_SECOND_CHAN_NONE;
  esp_wifi_get_channel(&pri_ch, &sec_ch);
  // Read MAC directly from esp-idf — WiFi.macAddress() can lag behind
  // mode-change on ESP32-S3 and return all-zeros even when STA is up.
  uint8_t mac[6] = {0};
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  send_textf("kind=espnow phase=init status=ok node=camera "
             "mac=%02X:%02X:%02X:%02X:%02X:%02X "
             "channel_actual=%u channel_want=%d joined=%d bcast=%d",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
             (unsigned)pri_ch, HASH_KWS_ESPNOW_CHANNEL,
             (int)joined,
             (int)g_broadcast_peer_added);
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Setup / loop
// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  Serial.setTxBufferSize(8192);
  delay(400);
  send_text("kind=boot status=start app=camera_classifier");
  send_textf("kind=psram total_kb=%u free_kb=%u",
             (unsigned)(ESP.getPsramSize() / 1024),
             (unsigned)(ESP.getFreePsram() / 1024));

  if (!init_camera()) {
    while (true) delay(1000);
  }
  if (!init_tflm()) {
    while (true) delay(1000);
  }
  init_preproc((float)VWW_INPUT_SCALE, VWW_INPUT_ZERO_POINT);

  send_textf("kind=preproc in_scale=%.6f in_zp=%d "
             "labels=%s,%s norm=symmetric_pm1 input=%dx%dx%d",
             (float)VWW_INPUT_SCALE, VWW_INPUT_ZERO_POINT,
             VWW_LABEL_0, VWW_LABEL_1,
             VWW_INPUT_HEIGHT, VWW_INPUT_WIDTH, VWW_INPUT_CHANNELS);

  if (!BringUpEspNowLink()) {
    send_text("kind=espnow note=continuing_without_link "
              "msg=triggered_mode_will_never_fire");
  } else {
    // One status broadcast on boot — confirms ESP-NOW radio actually emits.
    SendStatusBroadcast();
  }
  send_textf("kind=mode standalone=%d trigger_drain_ms=%d",
             (int)CAMERA_CLASSIFIER_STANDALONE_INFER,
             (int)CAMERA_CLASSIFIER_IDLE_DRAIN_MS);
  send_text("kind=stream phase=start status=ok");
}

// One full inference cycle on the freshest available frame.
// Returns false if the camera failed; on success fills *label_idx_out (0/1),
// *score_q8_out, *fb_ms_out, *invoke_ms_out, *latency_ms_out.
static bool capture_and_infer(uint8_t* label_idx_out,
                              uint8_t* score_q8_out,
                              uint16_t* fb_ms_out,
                              uint16_t* invoke_ms_out,
                              uint16_t* latency_ms_out) {
  uint32_t t_outer = millis();
  uint32_t t_fb0 = millis();
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    send_text("kind=camera phase=capture status=fb_get_fail");
    return false;
  }
  if (fb->format != PIXFORMAT_RGB565) {
    esp_camera_fb_return(fb);
    send_text("kind=camera phase=capture status=non_rgb565");
    return false;
  }
  uint32_t fb_ms = millis() - t_fb0;

  const char* label_name = "?";
  float       score      = 0.0f;
  uint32_t    inner_lat  = 0;
  bool ok = run_inference(fb, &label_name, &score, &inner_lat);
  esp_camera_fb_return(fb);
  if (!ok) return false;

  const uint8_t label_idx =
      (strcmp(label_name, VWW_LABEL_1) == 0) ? 1 : 0;
  int s_int = (int)lrintf(score * 255.0f);
  if (s_int < 0) s_int = 0; else if (s_int > 255) s_int = 255;

  *label_idx_out  = label_idx;
  *score_q8_out   = (uint8_t)s_int;
  *fb_ms_out      = (uint16_t)fb_ms;
  // run_inference reported pre_ms + invoke_ms as its inner_lat. Approximate
  // invoke_ms ≈ inner_lat - small_pre (which is ~10-20 ms on QVGA RGB565).
  *invoke_ms_out  = (uint16_t)inner_lat;
  *latency_ms_out = (uint16_t)(millis() - t_outer);
  return true;
}

#if CAMERA_CLASSIFIER_STANDALONE_INFER
// Continuous-infer-every-frame loop (the old behaviour, kept for solo debug).
static void loop_standalone() {
  static uint32_t s_last_hb = 0;
  static uint32_t s_frames_window = 0;
  static uint32_t s_window_start = 0;

  uint8_t  lbl = 0, sc = 0;
  uint16_t fb_ms = 0, inv_ms = 0, lat_ms = 0;
  if (!capture_and_infer(&lbl, &sc, &fb_ms, &inv_ms, &lat_ms)) {
    delay(50);
    return;
  }
  uint32_t this_seq = ++s_seq;
  s_frames_window++;
  send_textf("kind=infer label=%s score=%.3f fb_ms=%u latency_ms=%u seq=%lu",
             (lbl == 1) ? VWW_LABEL_1 : VWW_LABEL_0,
             sc / 255.0f, (unsigned)fb_ms, (unsigned)lat_ms,
             (unsigned long)this_seq);

  uint32_t now = millis();
  if (s_window_start == 0) s_window_start = now;
  if (now - s_last_hb >= 2000) {
    uint32_t dt = now - s_window_start;
    float fps = dt > 0 ? s_frames_window * 1000.0f / dt : 0.0f;
    send_textf("kind=heartbeat uptime_s=%lu seq=%lu fps=%.1f "
               "free_heap_kb=%u psram_free_kb=%u",
               (unsigned long)(now / 1000),
               (unsigned long)s_seq, fps,
               (unsigned)(ESP.getFreeHeap() / 1024),
               (unsigned)(ESP.getFreePsram() / 1024));
    s_frames_window = 0;
    s_window_start = now;
    s_last_hb = now;
  }
}
#endif

// Triggered-mode loop. Idle until a CameraCmdPacket arrives; drain the
// camera FB periodically so the trigger doesn't see a stale buffer.
static void loop_triggered() {
  static uint32_t s_last_hb     = 0;
  static uint32_t s_last_drain  = 0;

  // Periodic drain: pull and immediately return a frame so the sensor pipeline
  // keeps producing fresh content. Without this, after ~1 s of idle the next
  // get() can deliver an aged frame.
  uint32_t now = millis();
  if (!g_pending.have &&
      (now - s_last_drain) >= CAMERA_CLASSIFIER_IDLE_DRAIN_MS) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    s_last_drain = now;
  }

  if (g_pending.have) {
    PendingTrigger trig = g_pending;     // copy out of volatile struct
    g_pending.have = false;              // clear before work; another can latch

    uint8_t  lbl = 0, sc = 0;
    uint16_t fb_ms = 0, inv_ms = 0, lat_ms = 0;
    bool ok = capture_and_infer(&lbl, &sc, &fb_ms, &inv_ms, &lat_ms);
    if (ok) g_inferences_done++;
    SendReply(trig, ok, lbl, sc, fb_ms, inv_ms, lat_ms);
    SendStatusBroadcast();   // post-inference snapshot so master sees counters bump

    uint32_t age_ms = millis() - trig.received_at_ms;
    send_textf("kind=infer trigger_id=%u trigger_label=%u "
               "label=%s score=%.3f fb_ms=%u invoke_ms=%u "
               "latency_ms=%u age_ms=%lu",
               (unsigned)trig.trigger_id, (unsigned)trig.trigger_label,
               (lbl == 1) ? VWW_LABEL_1 : VWW_LABEL_0,
               sc / 255.0f,
               (unsigned)fb_ms, (unsigned)inv_ms,
               (unsigned)lat_ms, (unsigned long)age_ms);
    s_last_drain = millis();
  } else {
    delay(5);   // give the receive task room to run
  }

  if (now - s_last_hb >= 2000) {
    send_textf("kind=heartbeat uptime_s=%lu mode=triggered "
               "triggers=%lu replies_ok=%lu replies_fail=%lu "
               "free_heap_kb=%u psram_free_kb=%u",
               (unsigned long)(now / 1000),
               (unsigned long)g_triggers_received,
               (unsigned long)g_replies_sent,
               (unsigned long)g_replies_failed,
               (unsigned)(ESP.getFreeHeap() / 1024),
               (unsigned)(ESP.getFreePsram() / 1024));
    // Also broadcast the same counters over ESP-NOW so the master can print
    // them. Critical for debugging "trigger doesn't fire" without two USB ports.
    SendStatusBroadcast();
    s_last_hb = now;
  }
}

void loop() {
#if CAMERA_CLASSIFIER_STANDALONE_INFER
  loop_standalone();
#else
  loop_triggered();
#endif
}
