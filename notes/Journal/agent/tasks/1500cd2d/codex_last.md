Report and journal update are done.

**Artifacts**
- Report: `notes/Journal/agent/tasks/1500cd2d/report.md`
- Journal entry: `notes/Journal/2026-02-24_agent_codex_1500cd2d.yaml`

Key findings (see report for details and file refs):
- Busy-wait in `InitAudioRecording` can spin if the mic path is dead and risk WDT resets.
- Partial ringbuffer reads still feed stale audio into inference.
- I2S port mismatch between `i2s_mic_test` (`I2S_NUM_0`) and inference (`I2S_NUM_1`).
- README says fake-mic default is on, but code default is off.
- Duplicate micro_speech copies risk drift.

Tests were not run (review only).

If you want me to implement the fixes, confirm the intended I2S port/pin mapping and I’ll patch `audio_provider.cpp` + README, then ask for ESP32 serial logs to validate.