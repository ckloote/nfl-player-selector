"""Settings that can be moved for one block and are always put back."""

import pytest

from pool import config


# --- config overrides -------------------------------------------------------
def test_override_restores_values_even_when_the_body_raises():
    before = config.FUTURE_DISCOUNT
    with pytest.raises(RuntimeError):
        with config.override(FUTURE_DISCOUNT=0.5):
            assert config.FUTURE_DISCOUNT == 0.5
            raise RuntimeError("boom")
    assert config.FUTURE_DISCOUNT == before


def test_override_rejects_unknown_names():
    """A typo'd sweep parameter that silently changed nothing would report a
    flat surface and read as a finding."""
    with pytest.raises(KeyError, match="FUTURE_DISCOUNTT"):
        with config.override(FUTURE_DISCOUNTT=0.5):
            pass
