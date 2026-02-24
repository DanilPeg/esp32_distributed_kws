**Summary**
- Reviewed `code/firmware/` focusing on I2S mic test and micro_speech audio/inference pipeline.
- Found several stability and configuration mismatches; no code changes made.

**Findings**
- `InitAudioRecording` busy-waits on `g_latest_audio_timestamp` without a delay, so a dead mic/I2S path can spin and trigger watchdog resets. `code/firmware/micro_speech_sim/micro_speech/audio_provider.cpp:162`.
- Ringbuffer partial reads log warnings but still feed the model with whatever data is left in the output buffer, which can reuse stale samples and cause spurious detections during underruns. `code/firmware/micro_speech_sim/micro_speech/audio_provider.cpp:195-205`.
- I2S port mismatch between the test sketch and inference path: `i2s_mic_test` uses `I2S_NUM_0` while `audio_provider` uses port 1, which can make "works in test, fails in inference" on some boards. `code/firmware/i2s_mic_test/i2s_mic_test.ino:10` vs `code/firmware/micro_speech_sim/micro_speech/audio_provider.cpp:90`.
- README claims fake-mic default is `USE_FAKE_MIC=1`, but the code default is 0, so runs without a physical mic may silently block. `code/firmware/micro_speech_sim/README.md:4-9` vs `code/firmware/micro_speech_sim/micro_speech/audio_provider.cpp:59-60`.
- Duplicate micro_speech sources exist in `code/firmware/micro_speech_sim/` and `code/firmware/micro_speech_sim/micro_speech/`, which risks drift when applying fixes.

**Actions**
- No code changes in this review.
- Proposed fixes: add a timeout + `vTaskDelay` in the audio-init wait loop; zero-fill or return error on partial `rb_read`; align I2S port configuration between test and inference; update README or `USE_FAKE_MIC` default; pick one canonical micro_speech source tree and remove/sync the duplicate.

**Tests**
- Not run (review only).

**Next**
- If you want changes implemented, I can patch `audio_provider.cpp`/README and request ESP32 serial logs to validate.
- Confirm the intended I2S port/pin mapping for the target board so the test sketch and inference path stay aligned.
