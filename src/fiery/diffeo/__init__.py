"""Scaling-and-squaring and geodesic shooting layers in PyTorch."""

from .layers import *  # noqa: F401, F403

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0+unknown"
