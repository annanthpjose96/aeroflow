# ADR 0002: Simulator-first development

**Status:** accepted

**Context.** Airport footage can't be shared or collected freely, GPUs aren't always available, and
tuning needs ground truth.

**Decision.** Ship a deterministic multi-camera simulator that feeds the same `Detection` interface as
real cameras. CI, demos and regression tests run on it with no hardware.

**Consequences.** Fast, reproducible tests and measurable identity metrics. Risk: tuning to the
simulator. Mitigation: documented limits, seeds sweep in evaluation, and a roadmap item to validate on
public multi-camera datasets.
