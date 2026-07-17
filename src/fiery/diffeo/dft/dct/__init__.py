"""Discrete cosine and sine transforms, backed by ``fiery.bounds``.

These thin wrappers preserve diffeo's call convention
``dct(x, dim=-1, norm=None, type=2)`` while delegating the computation to
[`fiery.bounds.realtransforms`][], which implements the transforms with
``torch.fft`` (no scipy/cupy dependency). ``norm=None`` maps to the
``'backward'`` normalisation, which is numerically identical (to machine
precision) to the previous scipy-based implementation.

``fiery.bounds`` implements DCT/DST types I-III; diffeo historically fell
back to type III for the (rarely used) type IV, and that behaviour is
preserved here.
"""
from fiery.bounds.realtransforms import dct as _dct
from fiery.bounds.realtransforms import dst as _dst
from fiery.bounds.realtransforms import idct as _idct
from fiery.bounds.realtransforms import idst as _idst

__all__ = [
    'dct', 'dct1', 'dct2', 'dct3', 'dct4',
    'dst', 'dst1', 'dst2', 'dst3', 'dst4',
    'idct', 'idct1', 'idct2', 'idct3', 'idct4',
    'idst', 'idst1', 'idst2', 'idst3', 'idst4',
]


def _norm(norm):
    return 'backward' if norm is None else norm


def _type(type):
    # fiery.bounds implements types I-III; preserve diffeo's type-IV -> III.
    return min(type, 3)


def dct(x, dim=-1, norm=None, type=2):
    return _dct(x, dim=dim, norm=_norm(norm), type=_type(type))


def idct(x, dim=-1, norm=None, type=2):
    return _idct(x, dim=dim, norm=_norm(norm), type=_type(type))


def dst(x, dim=-1, norm=None, type=2):
    return _dst(x, dim=dim, norm=_norm(norm), type=_type(type))


def idst(x, dim=-1, norm=None, type=2):
    return _idst(x, dim=dim, norm=_norm(norm), type=_type(type))


def dct1(x, dim=-1, norm=None):
    return dct(x, dim, norm, 1)


def dct2(x, dim=-1, norm=None):
    return dct(x, dim, norm, 2)


def dct3(x, dim=-1, norm=None):
    return dct(x, dim, norm, 3)


def dct4(x, dim=-1, norm=None):
    return dct(x, dim, norm, 4)


def idct1(x, dim=-1, norm=None):
    return idct(x, dim, norm, 1)


def idct2(x, dim=-1, norm=None):
    return idct(x, dim, norm, 2)


def idct3(x, dim=-1, norm=None):
    return idct(x, dim, norm, 3)


def idct4(x, dim=-1, norm=None):
    return idct(x, dim, norm, 4)


def dst1(x, dim=-1, norm=None):
    return dst(x, dim, norm, 1)


def dst2(x, dim=-1, norm=None):
    return dst(x, dim, norm, 2)


def dst3(x, dim=-1, norm=None):
    return dst(x, dim, norm, 3)


def dst4(x, dim=-1, norm=None):
    return dst(x, dim, norm, 4)


def idst1(x, dim=-1, norm=None):
    return idst(x, dim, norm, 1)


def idst2(x, dim=-1, norm=None):
    return idst(x, dim, norm, 2)


def idst3(x, dim=-1, norm=None):
    return idst(x, dim, norm, 3)


def idst4(x, dim=-1, norm=None):
    return idst(x, dim, norm, 4)
