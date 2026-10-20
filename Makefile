.PHONY: install test lint fmt run eval bench docker clean
install:  ; pip install -e ".[dev]"
test:     ; pytest -q
lint:     ; ruff check src tests && ruff format --check src tests
fmt:      ; ruff format src tests && ruff check --fix src tests
run:      ; aeroflow run
eval:     ; aeroflow eval --seconds 600
bench:    ; python benchmarks/benchmark_inference.py
docker:   ; docker compose up --build
clean:    ; rm -rf .pytest_cache .ruff_cache build *.egg-info aeroflow.db events.jsonl
