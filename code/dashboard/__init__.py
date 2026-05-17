"""New master dashboard for the distributed ESP32 hash-KWS cluster.

Replaces the legacy ``code/web_portal/`` entrypoint. Reads the same JSONL
event streams emitted by ``hash_kws_serial_bridge.py``,
``hash_kws_cluster_sim.py`` and ``hash_kws_dual_audio_fusion.py``.
"""
