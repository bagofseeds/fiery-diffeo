"""Import smoke tests."""

import importlib


def test_diffeo_importable() -> None:
    """The diffeo package should be importable after installation."""
    module = importlib.import_module("fiery.diffeo")
    assert module is not None


def test_layer_classes_importable() -> None:
    """The public layer classes should be importable from the package root."""
    from fiery.diffeo import BCH, Exp, Pull, Push, Shoot  # noqa: F401
