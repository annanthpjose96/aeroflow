"""Holt's linear exponential smoothing for short-horizon queue forecasting."""

from __future__ import annotations


class Holt:
    def __init__(self, alpha: float = 0.3, beta: float = 0.1, min_samples: int = 12):
        self.alpha, self.beta, self.min_samples = alpha, beta, min_samples
        self.level: float | None = None
        self.trend = 0.0
        self.n = 0

    def update(self, x: float) -> None:
        if self.level is None:
            self.level = float(x)
        else:
            prev = self.level
            self.level = self.alpha * x + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev) + (1 - self.beta) * self.trend
        self.n += 1

    def predict(self, steps: int) -> float | None:
        if self.level is None or self.n < self.min_samples:
            return None
        return max(0.0, self.level + steps * self.trend)
