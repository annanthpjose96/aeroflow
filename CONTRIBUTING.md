# Contributing

```bash
pip install -e ".[dev]"
make lint test
```

* Keep `Engine.process` as the single integration point; new detectors/sources only need to emit `Detection`s.
* Every bug fix needs a regression test (see `tests/test_pipeline.py`).
* Write a short post-mortem (`docs/postmortem_template.md`) for any incident-level bug.
* Never commit camera credentials, video, or model weights (see `.gitignore`).
