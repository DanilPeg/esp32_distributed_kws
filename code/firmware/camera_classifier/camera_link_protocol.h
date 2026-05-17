// camera_link_protocol.h
//
// Shared ESP-NOW packet definitions for the master → camera trigger and
// camera → master inference reply. Mirrored byte-for-byte in:
//   code/firmware/hash_kws_master_web/camera_link_protocol.h
//   code/firmware/camera_classifier/camera_link_protocol.h
// Keep these two files in sync. Either side rejects packets whose magic /
// version / CRC don't validate, so a stale copy is loud, not silent.
//
// Wire layout is little-endian (ESP32 native, no endian swaps anywhere).
// CRC16 is Modbus/IBM polynomial 0xA001, init 0xFFFF, over all bytes of the
// struct except the trailing crc16 field itself. Same primitive the audio
// KWS path uses (HashKwsEspNowPacket::crc16 in micro_speech.ino), so we can
// reuse the helper.

#ifndef DIPLOMA_CAMERA_LINK_PROTOCOL_H_
#define DIPLOMA_CAMERA_LINK_PROTOCOL_H_

#include <stdint.h>
#include <string.h>

// ---- Master → camera trigger ---------------------------------------------

constexpr uint32_t kCamCmdMagic   = 0x434D4431UL;  // "CMD1"
constexpr uint8_t  kCamCmdVersion = 1;

constexpr uint8_t kCamCmdKindInferRequest = 1;

struct __attribute__((packed)) CameraCmdPacket {
  uint32_t magic;
  uint8_t  version;
  uint8_t  kind;            // kCamCmdKind*
  uint16_t trigger_id;      // master-assigned, camera echoes verbatim in reply
  uint32_t t_ms_master;     // master uptime when fired
  uint8_t  trigger_label;   // KWS label index that fired this trigger
  uint8_t  reserved[3];     // pad to 4-byte boundary
  uint16_t crc16;
};
static_assert(sizeof(CameraCmdPacket) == 18,
              "CameraCmdPacket size drift -- breaks ESP-NOW length check");

// ---- Camera → master inference reply -------------------------------------

constexpr uint32_t kCamRepMagic   = 0x52455031UL;  // "REP1"
constexpr uint8_t  kCamRepVersion = 1;

constexpr uint8_t kCamRepKindInferDone = 1;
constexpr uint8_t kCamRepKindInferFail = 2;

struct __attribute__((packed)) CameraReplyPacket {
  uint32_t magic;
  uint8_t  version;
  uint8_t  kind;            // kCamRepKind*
  uint16_t trigger_id;      // echoed from the cmd packet
  uint32_t t_ms_camera;     // camera uptime when inference finished
  uint8_t  label;           // VWW index: 0 = no_person, 1 = person
  uint8_t  score_q8;        // round(p * 255) for the predicted label
  uint16_t latency_ms;      // wall-time from cmd-received to reply-sent
  uint16_t fb_ms;           // camera_fb_get() time on this trigger
  uint16_t invoke_ms;       // tflm Invoke() time on this trigger
  uint16_t crc16;
};
static_assert(sizeof(CameraReplyPacket) == 22,
              "CameraReplyPacket size drift -- breaks ESP-NOW length check");

// ---- Camera → master diagnostic heartbeat --------------------------------
// Broadcast every ~2 s and ALSO once immediately when the camera receives a
// valid trigger. Lets the operator diagnose "trigger doesn't fire" issues
// (channel mismatch, CRC drift, peer not added, etc.) without having to plug
// the camera node into USB. Master prints "hash_evt kind=cam_status ..." which
// the demux bridge picks up like any other event.

constexpr uint32_t kCamStatusMagic   = 0x53544131UL;  // "STA1"
constexpr uint8_t  kCamStatusVersion = 1;

struct __attribute__((packed)) CameraStatusPacket {
  uint32_t magic;
  uint8_t  version;
  uint8_t  channel;              // current ESP-NOW channel as seen by camera
  uint32_t uptime_ms;
  uint32_t packets_seen;         // total ESP-NOW callbacks (any length)
  uint16_t triggers_received;    // valid CameraCmdPacket count (post-CRC)
  uint16_t triggers_rejected;    // packets that looked cmd-sized but failed CRC/magic
  uint16_t inferences_done;
  uint16_t replies_sent;
  uint16_t replies_failed;
  uint16_t free_heap_kb;
  uint16_t psram_free_kb;
  uint16_t crc16;
};
static_assert(sizeof(CameraStatusPacket) == 30,
              "CameraStatusPacket size drift -- breaks ESP-NOW length check");

// ---- Shared CRC16 (Modbus, poly 0xA001) ----------------------------------
// Inlined here so each sketch builds without depending on the other.
static inline uint16_t CameraLinkCrc16(const uint8_t* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint16_t>(data[i]);
    for (int bit = 0; bit < 8; ++bit) {
      if ((crc & 1) != 0) {
        crc = static_cast<uint16_t>((crc >> 1) ^ 0xA001);
      } else {
        crc = static_cast<uint16_t>(crc >> 1);
      }
    }
  }
  return crc;
}

static inline bool CameraLinkValidateCmd(const CameraCmdPacket& p) {
  if (p.magic != kCamCmdMagic) return false;
  if (p.version != kCamCmdVersion) return false;
  if (p.kind != kCamCmdKindInferRequest) return false;
  const uint16_t want = CameraLinkCrc16(
      reinterpret_cast<const uint8_t*>(&p),
      sizeof(p) - sizeof(p.crc16));
  return p.crc16 == want;
}

static inline bool CameraLinkValidateReply(const CameraReplyPacket& p) {
  if (p.magic != kCamRepMagic) return false;
  if (p.version != kCamRepVersion) return false;
  if (p.kind != kCamRepKindInferDone && p.kind != kCamRepKindInferFail) return false;
  const uint16_t want = CameraLinkCrc16(
      reinterpret_cast<const uint8_t*>(&p),
      sizeof(p) - sizeof(p.crc16));
  return p.crc16 == want;
}

static inline bool CameraLinkValidateStatus(const CameraStatusPacket& p) {
  if (p.magic != kCamStatusMagic) return false;
  if (p.version != kCamStatusVersion) return false;
  const uint16_t want = CameraLinkCrc16(
      reinterpret_cast<const uint8_t*>(&p),
      sizeof(p) - sizeof(p.crc16));
  return p.crc16 == want;
}

static inline void CameraLinkSignCmd(CameraCmdPacket* p) {
  p->magic   = kCamCmdMagic;
  p->version = kCamCmdVersion;
  p->crc16   = CameraLinkCrc16(reinterpret_cast<const uint8_t*>(p),
                               sizeof(*p) - sizeof(p->crc16));
}

static inline void CameraLinkSignReply(CameraReplyPacket* p) {
  p->magic   = kCamRepMagic;
  p->version = kCamRepVersion;
  p->crc16   = CameraLinkCrc16(reinterpret_cast<const uint8_t*>(p),
                               sizeof(*p) - sizeof(p->crc16));
}

static inline void CameraLinkSignStatus(CameraStatusPacket* p) {
  p->magic   = kCamStatusMagic;
  p->version = kCamStatusVersion;
  p->crc16   = CameraLinkCrc16(reinterpret_cast<const uint8_t*>(p),
                               sizeof(*p) - sizeof(p->crc16));
}

#endif  // DIPLOMA_CAMERA_LINK_PROTOCOL_H_
