"""Riemannian metrics on velocity fields.

A metric is a positive semi-definite linear operator `L` that maps a
velocity field to its momentum field. It plays two roles in this
package: it regularizes velocity fields (the penalty is the inner
product `(v, Lv)`), and it defines the geodesics that
[`fiery.diffeo.layers.Shoot`][] integrates.

Three families are available:

- [`Mixture`][fiery.diffeo.metrics.Mixture] builds `L` from a weighted
  sum of finite-difference energies (absolute, membrane, bending and
  linear-elastic);
- [`Laplace`][fiery.diffeo.metrics.Laplace] and
  [`Helmoltz`][fiery.diffeo.metrics.Helmoltz] use the analytical Green's
  function of the corresponding differential operator;
- [`Gaussian`][fiery.diffeo.metrics.Gaussian] uses a Gaussian filter as
  its Green's function.

All of them derive from the `Metric` base class and therefore share the
same `forward` / `inverse` / `whiten` / `color` / `logdet` interface.
"""

from .base import Metric
from .differential.layers import *
from .discrete.layers import *
from .gaussian.layers import *
