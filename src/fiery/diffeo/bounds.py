"""
Boundary conditions
===================

There is no common convention to name boundary conditions.
The mapping between the various naming conventions and the discrete
transforms that implement them is provided by [`fiery.bounds`][]; this
module is a thin adapter that adds diffeo's short aliases and the
flow-specific "sliding" boundary condition.

=========   ===========   =======================   =======================
Fourier     SciPy         Metric                    Description
=========   ===========   =======================   =======================
dft         wrap          circular                  c  d | a b c d |  a  b
dct2        reflect       neumann                   b  a | a b c d |  d  c
dct1        mirror                                  c  b | a b c d |  c  b
dst2                      dirichlet                -b -a | a b c d | -d -c
dst1                                               -a  0 | a b c d |  0 -d

We further define a flow-specific "sliding" boundary condition, which
uses a combination of dct2 and dst2 (dst2 along the component's own axis,
dct2 along the others).
"""
from fiery.bounds import to_fourier as _to_fourier

# diffeo-specific short aliases that fiery.bounds does not recognise
_short_aliases = {
    'circ': 'circulant',
    'c': 'circulant',
    'n': 'neumann',
    'd': 'dirichlet',
}


class _Bound2DFT:
    """Dict-like mapping from any bound name to its Fourier transform.

    The conversion itself is delegated to
    [`fiery.bounds.to_fourier`][fiery.bounds.types.to_fourier]; only the
    short aliases are handled here. Names that fiery.bounds does not know
    (e.g. `'sliding'`, `'zero'`, or a name already in Fourier convention
    that raises) fall back to the provided default -- reproducing the
    behaviour of the former hard-coded ``dict.get`` lookup.
    """

    def get(self, name, default=None):
        name = _short_aliases.get(name, name)
        try:
            return _to_fourier(name)
        except (ValueError, KeyError):
            return default


bound2dft = _Bound2DFT()


def has_sliding(bound):
    return any(map(lambda x: x.lower()[0] == 's', bound))


def sliding2dft(bound, d):
    new_bound = []
    for i, b in enumerate(bound):
        if b[0].lower() == 's':
            # sliding
            new_bound.append('dst2' if i == d else 'dct2')
        else:
            new_bound.append(b)
    return new_bound
