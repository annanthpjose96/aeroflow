from aeroflow.forecast import Holt


def test_needs_min_samples():
    h = Holt(min_samples=5)
    for x in range(3):
        h.update(x)
    assert h.predict(10) is None


def test_follows_linear_ramp():
    h = Holt()
    for i in range(80):
        h.update(2 * i)  # +2 per step
    pred = h.predict(10)
    assert 2 * 79 + 10 < pred < 2 * 79 + 30  # roughly continues the trend


def test_never_negative():
    h = Holt()
    for i in range(40):
        h.update(max(0, 40 - 3 * i))
    assert h.predict(100) == 0.0
