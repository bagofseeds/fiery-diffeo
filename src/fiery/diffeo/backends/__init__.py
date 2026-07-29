"""Interchangeable backends for the sampling operations.

Every layer and function in this package delegates warping (`pull`),
splatting (`push`, `count`) and sampled gradients (`grad`) to a backend
module. Three are available:

- `torch` uses `torch.nn.functional.grid_sample`. It is fast, but it
  does not implement all the boundary conditions used by the metrics and
  it approximates splatting with a small-deformation model.
- `interpol` uses
  [`fiery-interpol`](https://bagofseeds.github.io/fiery-interpol/), which
  implements every operator and boundary condition consistently in
  TorchScript. **This is the default backend.**
- `jitfields` uses [`jitfields`](https://github.com/balbasty/jitfields),
  which implements the same operators in C++/CUDA. It needs `cupy` and
  `cppyy`, so it is an optional dependency.

Pick one per layer with the `backend=` argument, or for a whole block
with the [`backend`][fiery.diffeo.backends.backend] context manager.
"""

from contextlib import contextmanager

from . import interpol

default_backend = interpol
"""Backend used by every layer and function that is not given an
explicit `backend=` argument. Defaults to `interpol`."""


@contextmanager
def backend(bck):
    """Temporarily set the default backend.

    Parameters
    ----------
    bck : module
        Backend to use inside the block. Must be one of the modules
        under `fiery.diffeo.backends`.

    Yields
    ------
    bck : module
        The backend that was set.

    Examples
    --------
    ```python
    from fiery.diffeo.layers import BCH, Exp
    from fiery.diffeo.backends import backend, jitfields

    with backend(jitfields):
        layer1 = Exp()
        layer2 = BCH()
    ```
    """
    saved_backend = default_backend
    try:
        globals()['default_backend'] = bck
        yield bck
    finally:
        globals()['default_backend'] = saved_backend
