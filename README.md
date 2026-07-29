# fiery-diffeo
Scaling-and-squaring and geodesic shooting layers in PyTorch

`fiery-diffeo` is a [`fiery`](https://bagofseeds.github.io/fiery/) bag: it
installs on its own and imports as `fiery.diffeo`.

## Getting started

This package requires `pytorch >= 1.8`,
[`fiery-interpol`](https://bagofseeds.github.io/fiery-interpol/) and
[`fiery-bounds`](https://bagofseeds.github.io/fiery-bounds/).
We require this pytorch version so that complex values and the modern
`torch.fft` module are supported. To install with pip, simply do:
```shell
pip install fiery-diffeo
```

The DCT/DST boundary modes (which allow using Neumann or Dirichlet boundary
conditions) work out of the box: they are provided by
[`fiery-bounds`](https://bagofseeds.github.io/fiery-bounds/) using
`torch.fft`, on both CPU and GPU, with no extra dependency.

Only the 2D Helmholtz metric additionally requires `scipy` (for the
analytical Green's function); it is bundled under the `[helmholtz]` tag:
```shell
pip install "fiery-diffeo[helmholtz]"
```

## Layers

```python
Exp(bound='circulant', steps=8, anagrad=False): ...
"""Exponentiate a stationary velocity field by scaling and squaring

Parameters
----------
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions.
steps : int
    Number of scaling and squaring steps
    (equivalent to 2**steps Euler integration steps).
anagrad : bool
    Use analytical gradients instead of autograd.
    They use less memory, as intermediate time steps need not be
    stored, but may be slightly less accurate.
"""

BCH(bound='circulant', order=2): ...
"""Compose two stationary velocity fields using the BCH formula

The Baker-Campbell-Hausdorff (BCH) formula gives the velocity field z
such that exp(z) = exp(x) o exp(y), as a series of nested Lie brackets
of x and y. See https://en.wikipedia.org/wiki/BCH_formula

Parameters
----------
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions.
order : {1, 2, 3, 4}
    Maximum order used in the BCH series.
"""

Shoot(metric=Mixture(absolute=0.001, membrane=0.1), steps=None,
      fast=True): ...
ShootInv(...): ...
ShootBoth(...): ...
"""Exponentiate an initial velocity field by geodesic shooting

`Shoot` returns the forward transform, `ShootInv` the inverse
transform, and `ShootBoth` returns both. All three take the same
arguments.

Parameters
----------
metric : Metric
    A Riemannian metric.
steps : int, optional
    Number of Euler integration steps.
    If None, use an educated guess based on the magnitude of the
    initial velocity.
fast : bool
    Use a faster but slightly less accurate integration scheme.
"""

Compose(bound='circulant'): ...
"""Compose two displacement fields

Parameters
----------
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions.
"""

Pull(bound='wrap'): ...
"""Warp an image using a displacement field

Parameters
----------
bound : [list of] {'wrap', 'reflect', 'mirror'}
    Boundary conditions.
    If warping a displacement field, can also be one of the
    metric bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
"""

Push(bound='wrap', normalize=False): ...
"""Splat an image using a displacement field

Parameters
----------
bound : [list of] {'wrap', 'reflect', 'mirror'}
    Boundary conditions.
    If splatting a displacement field, can also be one of the
    metric bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
normalize : bool
    Divide the pushed values by the result of `Count`.
"""

Count(bound='wrap'): ...
"""Splat an image of ones using a displacement field

Parameters
----------
bound : [list of] {'wrap', 'reflect', 'mirror'}
    Boundary conditions.
    If splatting a displacement field, can also be one of the
    metric bounds: {'circulant', 'neumann', 'dirichlet', 'sliding'}
"""
```

All of these layers follow PyTorch's channel-first convention: images
are `(batch, C, *spatial)` tensors, and velocity, displacement and
momentum fields are `(batch, D, *spatial)` tensors, where `D` is the
number of spatial dimensions.

## Metrics

We define a range of Riemannian metrics that can be used to regularize
velocity fields, and that must be used for geodesic shooting.

A metric is a positive semi-definite linear operator `L` that maps a
velocity field `v` to its momentum field `m = Lv`. The inner product
`(v, Lv)` is the squared norm of `v` under the metric, and is the
quantity that gets penalized when the metric is used as a regularizer.
Its inverse `K = inv(L)` is a convolution with the Green's function of
`L`, and maps momenta back to velocities.

All metrics implement the following methods. Each takes an extra
`factor: bool = True` argument, which controls whether the global
regularization factor is applied:
```python
metric.forward(x: Tensor) -> Tensor: ...
"""
Apply the forward linear operator `L`

Parameters
----------
x : (..., *spatial, D) tensor
    A velocity field.

Returns
-------
m : (..., *spatial, D) tensor
    A momentum field.
"""

metric.inverse(x: Tensor) -> Tensor: ...
"""
Apply the inverse linear operator `K = inv(L)`

Parameters
----------
x : (..., *spatial, D) tensor
    A momentum field.

Returns
-------
v : (..., *spatial, D) tensor
    A velocity field.
"""

metric.whiten(x: Tensor) -> Tensor: ...
"""
Apply the square root of the forward linear operator `sqrt(L)`

Parameters
----------
x : (..., *spatial, D) tensor
    A velocity field.

Returns
-------
w : (..., *spatial, D) tensor
    A white field.
"""

metric.color(x: Tensor) -> Tensor: ...
"""
Apply the square root of the inverse linear operator `sqrt(K)`

Parameters
----------
x : (..., *spatial, D) tensor
    A white field.

Returns
-------
v : (..., *spatial, D) tensor
    A velocity field.
"""

metric.logdet(x: Tensor) -> Tensor: ...
"""
Compute the log-determinant of the linear operator `logdet(L)`

Parameters
----------
x : (..., *spatial, D) tensor
    A velocity field.
    Its values are not used. Only its shape, dtype and device are used.

Returns
-------
ld : scalar tensor
    Log-determinant (scaled by batch size).
"""
```

`whiten` and `color` are inverses of each other: `whiten` maps a
velocity field to a white field (one with identity covariance under the
Gaussian prior whose precision matrix is `L`), and `color` maps a white
field back to a velocity field.

Here is the list of metrics currently available:
```python
Mixture(absolute=0, membrane=0, bending=0, lame_shears=0, lame_div=0,
        factor=1, voxel_size=1, bound='circulant', use_diff=True,
        learnable=False, cache=False): ...
"""
Positive semi-definite metric based on finite-difference regularisers.

Mixture of "absolute", "membrane", "bending" and "linear-elastic" energies.
Note that these quantities refer to what's penalised when computing the
inner product (v, Lv). The "membrane" energy is therefore closely related
to the "Laplacian" metric.

Parameters
----------
absolute : float
    Penalty on (squared) absolute values
membrane: float
    Penalty on (squared) first derivatives
bending : float
    Penalty on (squared) second derivatives
lame_shears : float
    Penalty on the (squared) symmetric component of the Jacobian
lame_div : float
    Penalty on the trace of the Jacobian
factor : float
    Global regularization factor (optionally: learnable)
voxel_size : list[float]
    Voxel size
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions
use_diff : bool
    Use finite differences to perform the forward pass.
    Otherwise, perform the convolution in Fourier space.
learnable : bool or {'factor', 'components', 'factor+components'}
    Make `factor` a learnable parameter.
    If 'components', the individual factors (absolute, membrane, etc.)
    are learned instead of the global factor, which is then fixed.
    Individual components can also be named one by one, e.g.
    'membrane+bending'.
    `True` is equivalent to 'factor'.
cache : bool or int
    Cache up to `n` kernels.
    This cannot be used together with learnable components.
"""

Laplace(factor=1, voxel_size=1, bound='circulant',
        learnable=False, cache=False): ...
"""
Positive semi-definite metric based on the Laplace operator.
This is relatively similar to SPM's "membrane" energy, but relies on
the (ill-posed) analytical form of the Green's function.

https://en.wikipedia.org/wiki/Laplace%27s_equation
https://en.wikipedia.org/wiki/Green%27s_function

Parameters
----------
factor : float
    Regularization factor (optionally: learnable)
voxel_size : list[float]
    Voxel size
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions
learnable : bool
    Make `factor` a learnable parameter
cache : bool or int
    Cache up to `n` kernels
"""

Helmholtz(factor=1, alpha=1e-3, voxel_size=1, bound='circulant',
         learnable=False, cache=False): ...
"""
Positive semi-definite metric based on the Helmholtz operator.
This is relatively similar to SPM's mixture of "absolute" and
"membrane" energies, but relies on the (ill-posed) analytical form
of the Green's function.

https://en.wikipedia.org/wiki/Helmholtz_equation
https://en.wikipedia.org/wiki/Green%27s_function

Parameters
----------
factor : float
    Regularization factor (optionally: learnable)
alpha : float
    Diagonal regularizer (cannot be learned).
    It is the square of the eigenvalue in the Helmholtz equation.
voxel_size : list[float]
    Voxel size
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions
learnable : bool
    Make `factor` a learnable parameter
cache : bool or int
    Cache up to `n` kernels
"""

Gaussian(fwhm=16, factor=1, voxel_size=1, bound='circulant',
         learnable=False, cache=False): ...
"""
Positive semi-definite metric whose Green's function is a Gaussian
filter.

Parameters
----------
fwhm : float
    Full-width at half-maximum of the Gaussian filter, in mm
    (optionally: learnable)
factor : float
    Global regularization factor (optionally: learnable)
voxel_size : list[float]
    Voxel size
bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
    Boundary conditions
learnable : bool or {'factor', 'fwhm', 'fwhm+factor'}
    Make `factor` and/or `fwhm` a learnable parameter.
    `True` is equivalent to 'factor'.
cache : bool or int
    Cache up to `n` kernels.
    This cannot be used together with a learnable `fwhm`.
"""
```

## Backends

We support three different backends for the underlying sampling
operations:

- `torch`: This backend uses `torch.nn.functional.grid_sample`. It does not
  implement all the boundary conditions that our metrics handle, and it
  approximates splatting with a small-deformation model. It should be
  fast, but also quite inaccurate.
- `interpol`: This backend uses the package
  [`fiery-interpol`](https://bagofseeds.github.io/fiery-interpol/), which
  implements all the necessary operators using TorchScript. It is not the
  fastest, but all operators and boundary conditions should be consistent.
  **This is the default backend.**
- `jitfields`: This backend uses the package
  [`jitfields`](https://github.com/balbasty/jitfields), which
  implements the same operators as `fiery-interpol`, but in pure C++/CUDA.
  It does require additional dependencies (`cupy` and `cppyy`), though.
  Therefore, `jitfields` is not a mandatory dependency of `fiery-diffeo`
  and must be installed manually by the user.

All our layers and functions take a `backend` argument:
```python
from fiery.diffeo.layers import Exp
from fiery.diffeo.backends import jitfields

layer = Exp(backend=jitfields)
```
Alternatively, we provide a context manager that sets the backend for
an entire block:
```python
from fiery.diffeo.layers import Exp, BCH
from fiery.diffeo.backends import backend, jitfields

with backend(jitfields):
    layer1 = Exp()
    layer2 = BCH()
```

Note that we currently have issues when using the `torch` backend
together with the geodesic shooting layers. Classic interpolation and
stationary velocity fields should work fine, however.

All backends implement the following functions:
```python
def pull(image, flow, bound='dct2', has_identity=False): ...
"""Warp an image according to a (voxel) displacement field.

Parameters
----------
image : (..., *shape_in, C) tensor
    Input image.
flow : (..., *shape_out, D) tensor
    Displacement field, in voxels.
bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}, default='dct2'
    Boundary conditions.
    Can also be one of the metric bounds
    {'circulant', 'neumann', 'dirichlet', 'sliding'}, in which case
    the image is assumed to be a flow field.
has_identity : bool, default=False
    - If False, `flow` contains a relative displacement.
    - If True, `flow` contains absolute coordinates.

Returns
-------
warped : (..., *shape_out, C) tensor
    Warped image
"""

def push(image, flow, shape=None, bound='dct2', has_identity=False): ...
"""Splat an image according to a (voxel) displacement field.

Parameters
----------
image : (..., *shape_in, C) tensor
    Input image.
flow : (..., *shape_out, D) tensor
    Displacement field, in voxels.
shape : list[int], optional
    Output shape
bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}, default='dct2'
    Boundary conditions.
    Can also be one of the metric bounds
    {'circulant', 'neumann', 'dirichlet', 'sliding'}, in which case
    the image is assumed to be a flow field.
has_identity : bool, default=False
    - If False, `flow` contains a relative displacement.
    - If True, `flow` contains absolute coordinates.

Returns
-------
pushed : (..., *shape_out, C) tensor
    Pushed image
"""

def count(flow, shape=None, bound='dct2', has_identity=False): ...
"""Splat an image of ones according to a (voxel) displacement field.

Parameters
----------
flow : (..., *shape_out, D) tensor
    Displacement field, in voxels.
shape : list[int], optional
    Output shape
bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}, default='dct2'
    Boundary conditions.
    Can also be one of the metric bounds
    {'circulant', 'neumann', 'dirichlet', 'sliding'}, in which case
    the count image may have D channels.
has_identity : bool, default=False
    - If False, `flow` contains a relative displacement.
    - If True, `flow` contains absolute coordinates.

Returns
-------
count : (..., *shape_out, 1|D) tensor
    Count image
"""

def grad(image, flow, bound='dct2', has_identity=False): ...
"""Compute image spatial gradients along a displacement field.

Parameters
----------
image : (..., *shape_in, C) tensor
    Input image.
flow : (..., *shape_out, D) tensor
    Displacement field, in voxels.
bound : {'dft', 'dct[1|2|3|4]', 'dst[1|2|3|4]'}, default='dct2'
    Boundary conditions.
    Can also be one of the metric bounds
    {'circulant', 'neumann', 'dirichlet', 'sliding'}, in which case
    the image is assumed to be a flow field.
has_identity : bool, default=False
    - If False, `flow` contains a relative displacement.
    - If True, `flow` contains absolute coordinates.

Returns
-------
grad : (..., *shape_out, C, D) tensor
    Sampled gradients
"""
```
