# Global Diploma Plan

## Current State
- PP draft exists: `~$oject_Proposal_IEEE_v3.docx` (mtime: 2026-02-23 22:54).
- Baseline micro_speech inference confirmed on ESP32S3.
- Web portal running with auth + ngrok.

## Milestones
1. Project Proposal (PP)
   - Finalize IEEE PP document
   - Align with TA and supervisor feedback

2. Hardware Baselines
   - Voice command baseline (done)
   - Image baseline (ESP32 camera) - pending

3. Model Training + Compression
   - Train baseline models (voice + image)
   - Quantize and convert to TFLite Micro

4. Distributed Protocol
   - Decide protocol (ESP-NOW/UDP)
   - Define message schema and tests

5. Integration
   - Deploy models to nodes
   - Sync inference + comms
   - Web UI visualization

6. Evaluation
   - Latency, accuracy, energy
   - Logs + reproducible artifacts

7. Report + Defense Prep
   - GOST formatting
   - Slides + demo scripts

## Risks
- Hardware noise and microphone variability
- Memory limits for TFLite Micro
- Schedule drift between firmware + ML pipelines

## Next Actions
- Confirm latest PP version and finalize structure.
- Add image baseline plan and hardware logs.
- Start protocol decision log.
