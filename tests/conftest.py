import pytest

from aeroflow.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/default.yaml")
