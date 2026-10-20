# ADR 0001: Fuse cameras on the floor plane

**Status:** accepted

**Context.** We need persistent identities across overlapping cameras. Options: (a) multi-view 3D
tracking (e.g. NVIDIA MV3DT), (b) image-space Re-ID only, (c) project 2D detections onto a shared
floor plane and associate there.

**Decision.** (c). Homographies are cheap to calibrate (≥4 floor points per camera), work with any 2D
detector, and put all cameras in one metric space where gating is simple and explainable.

**Consequences.** Needs a flat floor and reasonably accurate foot points; occluded feet or ramps add
error. Appearance Re-ID is a secondary cue. If calibration quality or crowd density demands it, swap
in MV3DT behind the same `Engine.process` interface.
