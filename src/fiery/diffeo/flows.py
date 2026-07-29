"""Utilities for displacement and transformation fields.

Two closely related objects appear throughout this package:

- a **displacement field** stores, at each voxel, the offset (in voxels)
  from that voxel to its target, so the identity displacement is zero;
- a **transformation field** stores the target coordinate itself, so the
  identity transformation is the sampling grid.

They differ by the identity grid, and `add_identity` / `sub_identity`
convert between them. Most functions here take displacement fields by
default and accept transformation fields through `has_identity=True`.

A **velocity field** has the same layout but a different meaning: it is
a tangent vector, integrated over unit time by `fiery.diffeo.svf.exp`
(stationary) or `fiery.diffeo.shoot.shoot` (geodesic) to yield a
displacement field.

All fields are `(..., *spatial, D) tensor`, i.e. channel-last, where `D`
is the number of spatial dimensions.
"""

import torch

from fiery.diffeo.backends import interpol as interpol_backend
from fiery.diffeo.bounds import bound2dft, has_sliding, sliding2dft
from fiery.diffeo.diffdiv import diff
from fiery.diffeo.linalg import batchmatvec
from fiery.diffeo.utils import cartesian_grid, ensure_list


def add_identity_(flow):
    """Add the identity grid to a displacement field, in place.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Displacement field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Transformation field

    """
    dim = flow.shape[-1]
    spatial = flow.shape[-dim - 1 : -1]
    grid = cartesian_grid(spatial, dtype=flow.dtype, device=flow.device)
    flow = flow.movedim(-1, 0)
    for i, grid1 in enumerate(grid):
        flow[i].add_(grid1)
    flow = flow.movedim(0, -1)
    return flow


def sub_identity_(flow):
    """Subtract the identity grid from a transformation field, in place.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Transformation field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Displacement field

    """
    dim = flow.shape[-1]
    spatial = flow.shape[-dim - 1 : -1]
    grid = cartesian_grid(spatial, dtype=flow.dtype, device=flow.device)
    flow = flow.movedim(-1, 0)
    for i, grid1 in enumerate(grid):
        flow[i].sub_(grid1)
    flow = flow.movedim(0, -1)
    return flow


def add_identity(flow):
    """Add the identity grid to a displacement field.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Displacement field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Transformation field

    """
    return add_identity_(flow.clone())


def sub_identity(flow):
    """Subtract the identity grid from a transformation field.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Transformation field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Displacement field

    """
    return sub_identity_(flow.clone())


def identity(shape, **backend):
    """Return an identity transformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial shape of the field.
    **backend : dict
        Keyword arguments passed to `torch.arange`, typically `dtype`
        and `device`.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    backend.setdefault('dtype', torch.get_default_dtype())
    return torch.stack(cartesian_grid(shape, **backend), dim=-1)


def affine_field(affine, shape, add_identity=False):
    """Generate an affine flow field

    Parameters
    ----------
    affine : (..., D+1, D+1) tensor
        Affine matrix
    shape : (D,) list[int]
        Lattice size
    add_identity : bool, default=False
        If True, return a transformation field (absolute coordinates).
        Otherwise, return a displacement field.

    Returns
    -------
    flow : (..., *shape, D) tensor
        Affine flow

    """
    ndim = len(shape)
    backend = dict(dtype=affine.dtype, device=affine.device)

    # add spatial dimensions so that we can use batch matmul
    for _ in range(ndim):
        affine = affine.unsqueeze(-3)
    lin, trl = affine[..., :ndim, :ndim], affine[..., :ndim, -1]

    # create affine transform
    flow = identity(shape, **backend)
    flow = lin.matmul(flow.unsqueeze(-1)).squeeze(-1)
    flow = flow.add_(trl)

    # subtract identity to get a flow
    if not add_identity:
        flow = sub_identity_(flow)

    return flow


def jacobian(
    flow,
    bound='circulant',
    voxel_size=1,
    has_identity=False,
    add_identity=None,
):
    """Compute the Jacobian of a transformation field

    Parameters
    ----------
    flow : (..., *spatial, dim) tensor
        Transformation or displacement field
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
        Boundary condition
    voxel_size : [sequence of] float, default=1
        Voxel size
    has_identity : bool, default=False
        Whether the input is a transformation (True) or displacement
        (False) field.
    add_identity : bool, default=`has_identity`
        Add the identity to the Jacobian of the displacement, making it
        the Jacobian of the transformation.

    Returns
    -------
    jac : (..., *spatial, dim, dim) tensor
        Jacobian. In each matrix: `jac[i, j] = d psi[i] / d x[j]`

    """
    ndim = flow.shape[-1]
    if has_identity:
        flow = sub_identity(flow)
    bound = ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))
    if has_sliding(bound):
        dims = list(range(-ndim, 0))
        jac = flow.new_zeros([*flow.shape, ndim])
        for d in range(ndim):
            bound1 = sliding2dft(bound, d)
            jac[..., d] = diff(
                flow[..., d],
                dim=dims,
                bound=bound1,
                voxel_size=voxel_size,
                side='c',
            )
    else:
        dims = list(range(-ndim - 1, -1))
        jac = diff(
            flow, dim=dims, bound=bound, voxel_size=voxel_size, side='c'
        )
    if add_identity is None:
        add_identity = has_identity
    if add_identity:
        torch.diagonal(jac, 0, -1, -2).add_(1)
    return jac


def jacdet(
    flow,
    bound='circulant',
    voxel_size=1,
    has_identity=False,
    add_identity=True,
):
    """Compute the determinant of the Jacobian of a transformation field

    Parameters
    ----------
    flow : (..., *spatial, dim) tensor
        Transformation or displacement field
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
        Boundary condition
    voxel_size : [sequence of] float, default=1
        Voxel size
    has_identity : bool, default=False
        Whether the input is a transformation (True) or displacement
        (False) field.
    add_identity : bool, default=True
        Add the identity to the Jacobian of the displacement, making it
        the Jacobian of the transformation.

    Returns
    -------
    det : (..., *spatial) tensor
        Jacobian determinant.

    """
    jac = jacobian(
        flow,
        bound=bound,
        voxel_size=voxel_size,
        has_identity=has_identity,
        add_identity=add_identity,
    )
    return jac.det()


def compose(
    flow_left,
    flow_right,
    bound='circulant',
    has_identity=False,
    backend=interpol_backend,
):
    """Compute the composition `flow_left o flow_right`

    Parameters
    ----------
    flow_left : (..., *shape, D) tensor
        Left-hand side field.
    flow_right : (..., *shape, D) tensor
        Right-hand side field.
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
        Boundary conditions.
    has_identity : bool, default=False
        Whether the inputs are transformation (True) or displacement
        (False) fields.
    backend : module
        Backend used to resample `flow_left`.

    Returns
    -------
    flow : (..., *shape, D) tensor
        Composed field, in the same convention as the inputs.

    """
    if has_identity:
        flow_left = sub_identity(flow_left)
    flow = backend.pull(
        flow_left, flow_right, bound=bound, has_identity=has_identity
    )
    if flow.requires_grad:
        flow = flow + flow_right
    else:
        flow += flow_right
    return flow


def compose_jacobian(
    jac,
    rhs,
    lhs=None,
    bound='circulant',
    has_identity=False,
    backend=interpol_backend,
):
    """Jacobian of the composition `(lhs)o(rhs)`

    Parameters
    ----------
    jac : (..., *spatial, ndim, ndim) tensor
        Jacobian of input RHS transformation
    rhs : (..., *spatial, ndim) tensor
        Right-hand side transformation
    lhs : (..., *spatial, ndim) tensor, default=`rhs`
        Left-hand side small displacement
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
        Boundary condition
    has_identity : bool, default=False
        Whether the left-hand side is a transformation (True) or
        displacement (False) field.
    backend : module
        Backend used to resample the left-hand side.

    Returns
    -------
    composed_jac : (..., *spatial, ndim, ndim) tensor
        Jacobian of composition

    """
    if lhs is None:
        lhs = rhs
    if has_identity:
        lhs = sub_identity(lhs)
    ndim = rhs.shape[-1]
    jac = jac.transpose(-1, -2)
    new_jac = torch.empty_like(jac)
    ensure_list(bound, ndim)
    bound = list(map(lambda x: bound2dft.get(x, x), bound))
    # NOTE: loop across dimensions to save memory
    for d in range(ndim):
        bound1 = sliding2dft(bound, d)
        jac_left = diff(lhs[..., d], bound=bound1, dim=range(-ndim, 0))
        jac_left = backend.pull(
            jac_left, rhs, bound=bound1, has_identity=has_identity
        )
        jac_left[..., d] += 1
        new_jac[..., d] = batchmatvec(jac, jac_left)
    return new_jac.transpose(-1, -2)


def bracket(
    vel_left,
    vel_right,
    bound='circulant',
    has_identity=False,
    backend=interpol_backend,
):
    """Compute the Lie bracket of two stationary velocity fields

    The Lie bracket `[u, w]` is the leading correction term of the BCH
    formula, which is why `exp(u) o exp(w)` is not `exp(u + w)` in
    general. It is approximated here by `u o w - w o u`, using the
    composition of displacement fields.

    Parameters
    ----------
    vel_left : (..., *shape, D) tensor
        Left-hand side velocity field.
    vel_right : (..., *shape, D) tensor
        Right-hand side velocity field.
    bound : [list of] {'circulant', 'neumann', 'dirichlet', 'sliding'}
        Boundary conditions.
    has_identity : bool, default=False
        Whether the inputs include the identity grid.
    backend : module
        Backend used to resample the velocity fields.

    Returns
    -------
    bkt : (..., *shape, D) tensor
        Lie bracket of the two velocity fields.

    """
    return compose(
        vel_left, vel_right, bound, has_identity, backend
    ) - compose(vel_right, vel_left, bound, has_identity, backend)
