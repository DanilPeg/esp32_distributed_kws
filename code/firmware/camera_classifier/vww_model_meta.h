// vww_model_meta.h  GENERATED -- do not hand-edit.
//
// Model: MCUNet vww1 (mcunet-5fps_vww.tflite)
// Source: HAN Lab pretrained release
//   https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-5fps_vww.tflite
// Reported VWW INT8 top-1 (HAN Lab README): 0.889
// Reported on STM32F746 / TinyEngine: SRAM~162KB flash~689KB
// Embedded blob size: 635968 bytes
//
// Preprocessing the model expects (NO ImageNet mean/std):
//   x_norm = (pixel / 127.5) - 1.0      (symmetric [-1, 1])
// Folded into INT8 input quantisation below.
//
// Regenerate: python code/training/image_classifier/import_mcunet_vww.py --variant vww1

#pragma once

#define VWW_INPUT_HEIGHT      80
#define VWW_INPUT_WIDTH       80
#define VWW_INPUT_CHANNELS    3
#define VWW_INPUT_SCALE       0.0078431377f
#define VWW_INPUT_ZERO_POINT  (-1)

#define VWW_NUM_CLASSES       2
#define VWW_OUTPUT_SCALE      0.1138677523f
#define VWW_OUTPUT_ZERO_POINT (-1)

#define VWW_LABEL_0  "no_person"
#define VWW_LABEL_1  "person"
