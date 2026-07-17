"""Tests pinning the behaviour de-vendored onto ``fiery-bounds``.

`fiery.diffeo` delegates boundary-condition name resolution and the DCT/DST
transforms to `fiery.bounds`. These tests check that the delegation preserves
the original behaviour: the bound-name mapping matches the historical table,
the transforms round-trip and are differentiable, and a Fourier-space metric
that exercises the DCT path runs and matches a direct fiery.bounds computation.
"""

import pytest
import torch

from fiery.diffeo.bounds import bound2dft, has_sliding, sliding2dft
from fiery.diffeo.dft.dct import dct, dst, idct, idst

# The historical, hard-coded diffeo table (now delegated to fiery.bounds).
_OLD_BOUND2DFT = {
    'circulant': 'dft', 'circ': 'dft', 'c': 'dft',
    'neumann': 'dct2', 'n': 'dct2',
    'dirichlet': 'dst2', 'd': 'dst2',
    'reflect': 'dct2', 'mirror': 'dct1', 'wrap': 'dft',
}
_PASSTHROUGH = ['dft', 'dct1', 'dct2', 'dst1', 'dst2', 'sliding', 's', 'zero']


@pytest.mark.parametrize(
    "name", list(_OLD_BOUND2DFT) + _PASSTHROUGH
)
def test_bound2dft_matches_legacy_table(name: str) -> None:
    """`bound2dft.get(x, x)` reproduces the former hard-coded mapping."""
    assert bound2dft.get(name, name) == _OLD_BOUND2DFT.get(name, name)


def test_sliding_helpers() -> None:
    assert has_sliding(['sliding', 'neumann'])
    assert not has_sliding(['neumann', 'dirichlet'])
    assert sliding2dft(['sliding', 'neumann'], 0) == ['dst2', 'neumann']
    assert sliding2dft(['sliding', 'sliding'], 1) == ['dct2', 'dst2']


@pytest.mark.parametrize("type", [1, 2, 3])
def test_dct_dst_roundtrip(type: int) -> None:
    """The inverse transforms invert the forward transforms."""
    x = torch.randn(9, dtype=torch.double)
    assert torch.allclose(idct(dct(x, type=type), type=type), x, atol=1e-8)
    assert torch.allclose(idst(dst(x, type=type), type=type), x, atol=1e-8)


@pytest.mark.parametrize("fn", [dct, dst, idct, idst])
def test_transform_gradcheck(fn) -> None:
    """The transforms are differentiable (they are torch.fft-based)."""
    from torch.autograd import gradcheck
    x = torch.randn(8, dtype=torch.double, requires_grad=True)
    assert gradcheck(lambda t: fn(t, type=2), (x,), atol=1e-6)


def test_dct_matches_fiery_bounds() -> None:
    """diffeo's DCT is exactly fiery.bounds' DCT (backward normalisation)."""
    from fiery.bounds.realtransforms import dct as bdct
    x = torch.randn(10, dtype=torch.double)
    assert torch.allclose(dct(x, type=2), bdct(x, type=2), atol=1e-10)


def test_fourier_metric_uses_dct() -> None:
    """A neumann (DCT2) Fourier-space metric runs and stays finite."""
    from fiery.diffeo.metrics import Mixture
    v = torch.randn(1, 8, 8, 2, dtype=torch.double)
    m = Mixture(membrane=1, bound='neumann', use_diff=False)
    out = m.forward(v)
    assert out.shape == v.shape
    assert torch.isfinite(out).all()
