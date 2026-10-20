# Security & privacy

AeroFlow is designed for sensitive public spaces. Principles and where they live in the code:

| Principle | Implementation |
|---|---|
| No biometric identification | No face recognition anywhere. Tracks are anonymous, short-lived integers. Appearance embeddings are colour descriptors used only for hand-off between cameras and are never stored. |
| Data minimisation | Only per-zone counts, wait estimates and alert records are persisted (`store.py`). No frames, no trajectories. |
| No raw video at rest | `RtspSource` processes frames in memory. Optional debug previews (`privacy.save_preview_dir`) pass through `privacy.blur_heads` first. |
| Retention | `aeroflow purge` / `Store.purge()` delete data older than `storage.retention_days`. Schedule it (cron/Kubernetes CronJob). |
| Least privilege | Docker image runs as a non-root user. |
| Secrets | No keys are required. The optional `GEMINI_API_KEY` is read from the environment, never from files in the repo (`.env` is git-ignored). |
| Minimal egress | The optional LLM receives only the question text and zone names. |
| Input validation | Search queries are length-limited; SQL uses parameter binding only. |

## Threats not covered (do these before a real deployment)

* **Authentication/TLS**: the API is unauthenticated and serves plain HTTP. Put it behind a reverse
  proxy with TLS and SSO, or add an auth dependency in `api.py`.
* **RTSP credentials** in `configs/*.yaml` are plain text; use a secrets manager/env substitution.
* **Camera network isolation**: keep cameras on a dedicated VLAN.
* **Legal basis**: GDPR/local privacy law and a DPIA are required before filming passengers, even
  with anonymisation. Anonymous tracking can still be personal data in some jurisdictions.
* **Model/supply-chain**: pin and verify model weights; scan images in CI.
