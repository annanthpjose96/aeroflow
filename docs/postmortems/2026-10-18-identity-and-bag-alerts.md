# Post-mortem: identity switches and unattended-bag alert quality

*Blameless. Found while building the first end-to-end version, using the simulator's ground truth.*

## Summary

The first working pipeline looked fine on a dashboard but, measured against ground truth, kept
identities poorly (**ID consistency 0.52, 271 global IDs for 147 real objects**), never raised the
scripted abandoned-bag alert, and raised a false "long wait" within a minute of start-up. After fixes:
consistency 0.88-0.95, every scripted incident detected exactly once, zero false incident alerts
across four seeds (see `docs/evaluation.md`).

## Timeline of findings and root causes

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | ~6.6 / 13.2 local track IDs per person (cam1 / cam2) | With a 3 % miss rate, one missed frame doubled displacement and exceeded the match gate, so tracks restarted. Tiny bags (12 px) could never match a walking owner. | Constant-velocity prediction + gate that widens after misses → 4.1 / 9.8. |
| 2 | Still many switches, concentrated in the static queue | Residual velocity "predicted" a person onto the neighbour 0.8 m away once the queue stopped. | Added an appearance term to the association cost → 1.8 / 3.2. |
| 3 | Bags still fragmented | Gate scale came from bbox size, too small for tiny objects at 2 Hz. | Configurable `min_scale` floor. Person ID consistency 0.98, bag 0.995 locally. |
| 4 | Abandoned bag **never** alerted | Rule was "no person within 3 m". The check-in hall is always busy, so someone was always near. | Bind each bag to its owner (nearest person while carried); alert when the *owner* leaves. |
| 5 | 3 false bag alerts (owner still holding the bag) | Fusion ID switches made the bound owner ID vanish. | Re-bind to anyone sustainedly within 0.8 m of the bag. |
| 6 | Alert delayed ~2 min and fired repeatedly | Passers-by briefly "attended" the bag and cleared the alert. | Require ≥ 6 s proximity; latch the alert until the bag moves. |
| 7 | Same bag alerted again 350 s later (seed 99) | Anchor frozen on a noisy first sample; normal jitter exceeded the 0.6 m "moved" threshold. | Smoothed anchor, 0.8 m threshold, state kept through short dropouts. |
| 8 | "Long wait" at t = 69 s | Throughput estimated from 1-2 exits looked tiny, so wait = count ÷ tiny. | Trust throughput only after ≥ 4 exits; otherwise use current dwell. |
| 9 | `open_lane` raised 13 times in one run | Forecast hovered around its threshold. | 15 % hysteresis and a 120 s cooldown for threshold-type alerts. |
| 10 | "how **many** people…" answered as an alert query | Keyword `"any "` matched inside "m**any** ". | Word-boundary regex; regression test added. |

## What went well

* Building a simulator with ground truth turned vague "looks off" into numbers within hours.
* Each fix got a regression test (`tests/test_pipeline.py`, `tests/test_nlsearch.py`).

## What we would do differently / actions

* [x] Evaluate identity metrics from day one, not after the UI worked.
* [x] Treat security-critical alerts as stateful (latch, hysteresis) by design.
* [ ] Re-validate all thresholds on real footage (WILDTRACK/MMPTRACK) - simulator tuning may not transfer.
* [ ] Replace the colour-histogram descriptor with a learned Re-ID model (OSNet → TensorRT).
* [ ] Add a replay tool to re-run recorded detections after threshold changes.
