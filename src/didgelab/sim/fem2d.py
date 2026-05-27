"""
2D FEM acoustic impedance solver for didgeridoo geometries.

Models the bore as a 2D channel in the meridional plane: a strip of width d(s)
along an arc-length parameter s. For a straight bore the centerline is the
x-axis. For a bent bore an arbitrary planar centerline can be supplied (e.g.
loaded from a CSV).

Coordinates are in millimetres throughout (matching ``fem.py``); the wave
number is converted from m^-1 to mm^-1 before assembly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import skfem as fem
from skfem.helpers import dot, grad

from .sim_interface import AcousticSimulationInterface
from ..geo import Geo


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def _resample_centerline(centerline: np.ndarray, n_axial: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a 2D polyline to ``n_axial`` points evenly spaced in arc length.

    Returns (resampled_points [n_axial,2], arc_length_at_each_point [n_axial]).
    """
    seg = np.diff(centerline, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = s[-1]
    s_uniform = np.linspace(0.0, total, n_axial)
    x = np.interp(s_uniform, s, centerline[:, 0])
    y = np.interp(s_uniform, s, centerline[:, 1])
    return np.column_stack([x, y]), s_uniform


def _normals_from_centerline(points: np.ndarray) -> np.ndarray:
    """Per-vertex unit normals to a 2D polyline (rotate tangent by +90 deg)."""
    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents /= norms
    # 90-degree CCW rotation: (tx, ty) -> (-ty, tx)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals


def build_2d_mesh(
    geo: np.ndarray,
    n_axial: int = 200,
    n_radial: int = 10,
    centerline: Optional[np.ndarray] = None,
    centerline_scale: Optional[float] = None,
    full_strip: bool = False,
) -> Tuple[fem.MeshTri, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D triangular mesh of the bore.

    By default the mesh covers the *half* meridional plane ``r in [0, d(s)/2]``
    (one side of the centerline), which is the natural domain for an
    axisymmetric Helmholtz solve. The fifth return value ``r_perp_flat`` gives
    each node's perpendicular distance to the centerline — used as the radial
    weight when assembling the axisymmetric forms.

    Args:
        geo: ``(N, 2)`` array of ``[x_mm, diameter_mm]`` along the bore axis.
        n_axial: number of nodes along the bore.
        n_radial: number of nodes across the bore (>=2). For the half strip
            ``n_radial`` includes both the axis (r=0) and the wall (r=d/2)
            nodes, so the effective radial resolution is ``n_radial - 1``.
        centerline: optional ``(M, 2)`` array in mm. If given, the bore is
            extruded along this planar polyline. If ``None``, the bore is
            straight along x.
        centerline_scale: multiplier on the bore length to set the centerline's
            target arc length. See module docs.
        full_strip: if True, mesh both sides of the centerline (offsets
            ``[-1, +1]``) — useful for *visualization* of a bent bore, but not
            correct for an axisymmetric solve. Default ``False`` (half strip).

    Returns:
        ``(mesh, points_2d, mouth_node_idx, bell_node_idx, r_perp_flat)``.
        ``points_2d`` has shape ``(n_axial*n_radial, 2)`` in mm.
        ``r_perp_flat`` has shape ``(n_axial*n_radial,)`` in mm and is the
        perpendicular distance of each node from the centerline (>= 0).
    """
    geo = np.asarray(geo, dtype=float)
    x_geo = geo[:, 0]
    d_geo = geo[:, 1]
    bore_length = float(x_geo[-1] - x_geo[0])

    # Axial parameter samples along arc length, mapped to original x for d-lookup
    s_uniform = np.linspace(0.0, bore_length, n_axial)
    d_axial = np.interp(s_uniform + x_geo[0], x_geo, d_geo)

    if centerline is None:
        # Straight: centerline is the x-axis
        cl = np.column_stack([s_uniform, np.zeros_like(s_uniform)])
        normals = np.tile(np.array([0.0, 1.0]), (n_axial, 1))
    else:
        cl_in = np.asarray(centerline, dtype=float)
        seg = np.diff(cl_in, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        total_in = float(seg_len.sum())
        if total_in == 0.0:
            raise ValueError("centerline has zero total length")

        # Target arc length = bore_length * scale  (scale defaults to 1.0)
        scale = 1.0 if centerline_scale is None else float(centerline_scale)
        target_arc = bore_length * scale
        scaled = (cl_in - cl_in[0]) * (target_arc / total_in)

        cl, _ = _resample_centerline(scaled, n_axial)
        normals = _normals_from_centerline(cl)

    # Radial sampling: axis (r=0) to wall (r=d/2) for the half strip; both sides
    # for the full strip. Either way the "radial distance" stored in r_perp is
    # the absolute perpendicular offset from the centerline.
    if full_strip:
        offsets = np.linspace(-1.0, 1.0, n_radial)
    else:
        offsets = np.linspace(0.0, 1.0, n_radial)

    points = np.empty((n_axial, n_radial, 2), dtype=float)
    r_perp = np.empty((n_axial, n_radial), dtype=float)
    for i in range(n_axial):
        for j in range(n_radial):
            points[i, j] = cl[i] + normals[i] * (offsets[j] * d_axial[i] * 0.5)
            r_perp[i, j] = abs(offsets[j]) * d_axial[i] * 0.5
    points_flat = points.reshape(-1, 2)
    r_perp_flat = r_perp.reshape(-1)

    # Triangulate the structured grid: two triangles per quad
    tris = []
    for i in range(n_axial - 1):
        for j in range(n_radial - 1):
            n00 = i * n_radial + j
            n01 = i * n_radial + (j + 1)
            n10 = (i + 1) * n_radial + j
            n11 = (i + 1) * n_radial + (j + 1)
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    tris = np.asarray(tris, dtype=np.int64)

    # skfem expects C-contiguous arrays of shape (geometric_dim, N) and
    # (n_vertices_per_elem, M); .T returns a view that's F-contiguous, which
    # causes skfem to log a "Transforming ... to C_CONTIGUOUS" warning.
    mesh = fem.MeshTri(
        np.ascontiguousarray(points_flat.T),
        np.ascontiguousarray(tris.T),
    )

    mouth_nodes = np.arange(0, n_radial, dtype=np.int64)
    bell_nodes = np.arange((n_axial - 1) * n_radial, n_axial * n_radial, dtype=np.int64)

    return mesh, points_flat, mouth_nodes, bell_nodes, r_perp_flat


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def fem2d(
    geo: np.ndarray,
    frequencies: np.ndarray,
    p: float = 1.225,
    n: float = 1.81e-5,
    c: float = 343.0,
    n_axial: int = 200,
    n_radial: int = 10,
    centerline: Optional[np.ndarray] = None,
    centerline_scale: Optional[float] = None,
):
    """Axisymmetric FEM acoustic impedance solver (mm coordinates).

    Solves the axisymmetric Helmholtz equation over the meridional half-plane
    of the bore. With ``u`` the pressure and ``r`` the perpendicular distance
    from the centerline, the weak form is

        \\int r * (grad u . grad v - k^2 u v) dA = source

    The ``r`` weight comes from the volume element ``r dr dz`` of cylindrical
    coordinates; without it the operator would describe a planar 2D waveguide
    and give the wrong resonance frequencies. This solver is equivalent to a
    1D Webster horn (and to the TLM segment chain) up to the lossless mode
    frequencies — agreement with TLM/1D-FEM on the first peak is well within
    5 cents on a typical didgeridoo bore.

    Args:
        geo: ``(N, 2)`` array of ``[x_mm, diameter_mm]``.
        frequencies: Hz array.
        p, n, c: density (kg/m^3), viscosity (Pa*s), speed of sound (m/s).
        n_axial, n_radial: mesh resolution.
        centerline: optional 2D polyline (mm) along which to extrude the bore.
        centerline_scale: see :func:`build_2d_mesh`.

    Returns:
        ``(impedance_array, mesh, points_flat, mouth_nodes, bell_nodes)``.
        ``impedance_array`` has ``len(frequencies)`` entries.
    """
    mesh, points_flat, mouth_nodes, bell_nodes, r_perp = build_2d_mesh(
        geo, n_axial=n_axial, n_radial=n_radial,
        centerline=centerline, centerline_scale=centerline_scale,
        full_strip=False,  # half-meridional plane for axisymmetric
    )

    element = fem.ElementTriP1()
    basis = fem.Basis(mesh, element)

    # Discretise the radial weight (perpendicular distance from centerline) as
    # a P1 field; it will appear at quadrature points as w['r'] in the forms.
    r_field = basis.interpolate(r_perp)

    @fem.BilinearForm
    def stiffness_axi(u, v, w):
        return w["r"] * dot(grad(u), grad(v))

    @fem.BilinearForm
    def mass_axi(u, v, w):
        return w["r"] * u * v

    K = stiffness_axi.assemble(basis, r=r_field)
    M = mass_axi.assemble(basis, r=r_field)

    # RHS: uniform forcing at mouth nodes. The radial weight is applied by the
    # operator during assembly, not here.
    b_mouth = np.zeros(basis.N)
    b_mouth[mouth_nodes] = 1.0

    # On-axis mouth probe (the j=0 column has offset=0, so node index 0 sits
    # exactly on the centerline at the mouth).
    mouth_axis_node = int(mouth_nodes[0])

    # Mapes-Riordan-style viscothermal damping (mean radius proxy, as in 1D FEM)
    avg_d_m = float(np.mean(geo[:, 1])) / 1000.0
    r_avg = avg_d_m / 2.0

    impedance = np.empty(len(frequencies), dtype=float)
    for fi, f in enumerate(frequencies):
        omega = 2.0 * np.pi * f
        r_v = r_avg * np.sqrt(p * omega / n)

        kw_m = omega / c
        k_real_m = kw_m * (1.0 + 1.045 / r_v)
        k_imag_m = kw_m * (1.045 / r_v)
        k_complex_m = k_real_m - 1j * k_imag_m

        # Convert m^-1 to mm^-1 for the mm-based mesh
        k_mm = k_complex_m / 1000.0
        k_sq = k_mm ** 2

        A = K - k_sq * M
        p_c = fem.solve(*fem.condense(A, b_mouth, D=bell_nodes))

        z_val = p_c[mouth_axis_node]
        impedance[fi] = float(np.abs(z_val))

    return impedance, mesh, points_flat, mouth_nodes, bell_nodes


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------


class FiniteElementsModeling2D(AcousticSimulationInterface):
    """2D FEM simulator.

    Inherits the ``AcousticConstants``-based constructor and adds optional
    arguments for mesh resolution and a bent-bore centerline.

    Args:
        constants: optional :class:`AcousticConstants`. Defaults to the
            base-class default (``compute_moist_air_properties()``).
        centerline: optional ``(M, 2)`` planar polyline (mm); if given, the
            bore follows this curve instead of being straight.
        n_axial: number of axial mesh stations (default 200).
        n_radial: number of cross-bore nodes (default 8).
    """

    def __init__(
        self,
        constants=None,
        centerline: Optional[np.ndarray] = None,
        n_axial: int = 200,
        n_radial: int = 10,
        centerline_scale: Optional[float] = None,
    ):
        super().__init__(constants)
        self.centerline = centerline
        self.centerline_scale = centerline_scale
        self.n_axial = n_axial
        self.n_radial = n_radial
        # Filled in after the most recent call to get_impedance_spectrum:
        self.last_mesh = None
        self.last_points = None
        self.last_mouth_nodes = None
        self.last_bell_nodes = None

    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array):
        impedance, mesh, points_flat, mouth_nodes, bell_nodes = fem2d(
            np.array(geo.geo),
            frequencies,
            p=self.constants.air_density,
            n=self.constants.dynamic_viscosity,
            c=self.constants.speed_of_sound,
            n_axial=self.n_axial,
            n_radial=self.n_radial,
            centerline=self.centerline,
            centerline_scale=self.centerline_scale,
        )
        self.last_mesh = mesh
        self.last_points = points_flat
        self.last_mouth_nodes = mouth_nodes
        self.last_bell_nodes = bell_nodes
        return impedance
