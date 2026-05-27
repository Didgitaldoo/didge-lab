"""
3D FEM acoustic impedance solver for didgeridoo geometries.

Builds a tetrahedral mesh by sweeping a circular cross-section along a 3D
centerline curve. A planar ``(M, 2)`` centerline is interpreted as a curve in
the Y-Z plane (X=0), matching the convention of ``fem2d`` and ``curve.csv``;
a ``(M, 3)`` centerline is taken as-is. For straight bores the centerline is
the x-axis.

Coordinates are in millimetres throughout (matching ``fem.py`` and
``fem2d.py``); the wavenumber is converted from m^-1 to mm^-1 before
assembly.

Unlike :mod:`fem2d`, the 3D solver needs no curvature/shortcut correction:
the bend lives in the actual geometry, so the wavefront-shortcut effect
emerges naturally from the 3D Helmholtz operator.
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


def _disk_template(n_radial: int, n_circ: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-disk 2D template: ``n_radial`` rings (including the center r=0
    and the boundary r=1), ``n_circ`` points around each non-center ring.

    Returns ``(nodes_2d, tris)`` with ``nodes_2d`` of shape ``(N, 2)`` and
    ``tris`` of shape ``(T, 3)`` (counter-clockwise triangles in the +z
    convention).
    """
    if n_radial < 2 or n_circ < 3:
        raise ValueError("need n_radial >= 2 and n_circ >= 3")
    nodes = [[0.0, 0.0]]
    for ring in range(1, n_radial):
        r = ring / (n_radial - 1)
        for k in range(n_circ):
            th = 2.0 * np.pi * k / n_circ
            nodes.append([r * np.cos(th), r * np.sin(th)])
    nodes = np.asarray(nodes, dtype=float)

    tris = []
    # Inner center fan (center node 0 to ring 1)
    for k in range(n_circ):
        kn = (k + 1) % n_circ
        tris.append([0, 1 + k, 1 + kn])
    # Outer ring quads -> two triangles each
    for ring in range(1, n_radial - 1):
        base_in = 1 + (ring - 1) * n_circ
        base_out = 1 + ring * n_circ
        for k in range(n_circ):
            kn = (k + 1) % n_circ
            n00 = base_in + k
            n01 = base_in + kn
            n10 = base_out + k
            n11 = base_out + kn
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    return nodes, np.asarray(tris, dtype=np.int64)


def _embed_centerline(
    centerline: Optional[np.ndarray],
    bore_length: float,
    n_axial: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample the centerline to ``n_axial`` arc-length-uniform points (after
    rescaling its arc to ``bore_length``), and produce a stable orthonormal
    cross-section frame ``(tangent, e1, e2)`` per station via parallel
    transport (Bishop frame). Planar ``(M, 2)`` input is embedded in the YZ
    plane (X=0)."""
    if centerline is None:
        s = np.linspace(0.0, bore_length, n_axial)
        cl3 = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
        tangent = np.tile(np.array([1.0, 0.0, 0.0]), (n_axial, 1))
        e1 = np.tile(np.array([0.0, 1.0, 0.0]), (n_axial, 1))
        e2 = np.tile(np.array([0.0, 0.0, 1.0]), (n_axial, 1))
        return cl3, tangent, e1, e2

    cl_in = np.asarray(centerline, dtype=float)
    if cl_in.ndim != 2 or cl_in.shape[1] not in (2, 3):
        raise ValueError("centerline must be (M, 2) or (M, 3)")
    if cl_in.shape[1] == 2:
        # Planar curve -> embed in YZ plane (matches curve.csv convention)
        cl_in = np.column_stack([np.zeros(len(cl_in)), cl_in[:, 0], cl_in[:, 1]])

    seg = np.diff(cl_in, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s_in = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(s_in[-1])
    if total == 0.0:
        raise ValueError("centerline has zero total length")
    # Rescale arc to bore_length so axial parameter == bore coordinate
    cl_in = (cl_in - cl_in[0]) * (bore_length / total)
    s_in = s_in * (bore_length / total)

    s_u = np.linspace(0.0, bore_length, n_axial)
    cl3 = np.column_stack([np.interp(s_u, s_in, cl_in[:, d]) for d in range(3)])

    # Tangent via central differences
    tangent = np.gradient(cl3, axis=0)
    tn = np.linalg.norm(tangent, axis=1, keepdims=True)
    tn[tn == 0.0] = 1.0
    tangent /= tn

    # Seed e1: pick the world axis least aligned with tangent[0], then project
    abs_t0 = np.abs(tangent[0])
    least = int(np.argmin(abs_t0))
    seed = np.zeros(3)
    seed[least] = 1.0
    e1_0 = seed - tangent[0] * float(np.dot(seed, tangent[0]))
    e1_0 /= np.linalg.norm(e1_0)

    # Parallel transport (Rodrigues rotation between consecutive tangents)
    e1 = np.empty((n_axial, 3))
    e1[0] = e1_0
    for i in range(1, n_axial):
        t0 = tangent[i - 1]
        t1 = tangent[i]
        v = np.cross(t0, t1)
        sin_a = float(np.linalg.norm(v))
        cos_a = float(np.dot(t0, t1))
        if sin_a < 1e-12:
            e1[i] = e1[i - 1]
        else:
            axis = v / sin_a
            e_prev = e1[i - 1]
            e1[i] = (
                e_prev * cos_a
                + np.cross(axis, e_prev) * sin_a
                + axis * float(np.dot(axis, e_prev)) * (1.0 - cos_a)
            )
        # Re-orthogonalise against drift
        e1[i] -= tangent[i] * float(np.dot(tangent[i], e1[i]))
        norm = np.linalg.norm(e1[i])
        if norm > 0:
            e1[i] /= norm

    e2 = np.cross(tangent, e1)
    return cl3, tangent, e1, e2


def _fix_tet_orientation(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Flip any tet whose signed volume is negative so all tets are positively
    oriented (skfem assembles correctly either way, but downstream consumers
    appreciate the convention)."""
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    vol = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0)
    flip = vol < 0.0
    if np.any(flip):
        a = tets[flip, 2].copy()
        tets[flip, 2] = tets[flip, 3]
        tets[flip, 3] = a
    return tets


def build_3d_mesh(
    geo: np.ndarray,
    n_axial: int = 80,
    n_radial: int = 3,
    n_circ: int = 10,
    centerline: Optional[np.ndarray] = None,
) -> Tuple[fem.MeshTet, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 3D tetrahedral mesh of the bore.

    Args:
        geo: ``(N, 2)`` array of ``[x_mm, diameter_mm]`` along the bore axis.
        n_axial: number of axial stations.
        n_radial: number of radial layers in each cross-section (>= 2). Each
            cross-section has ``1 + (n_radial - 1) * n_circ`` nodes.
        n_circ: number of circumferential nodes per non-center radial ring.
        centerline: optional ``(M, 2)`` planar polyline (mm) in the YZ plane,
            or ``(M, 3)`` polyline. If ``None``, the bore is straight along x.

    Returns:
        ``(mesh, points_flat, mouth_nodes, bell_nodes)``.
        ``points_flat`` has shape ``(n_axial * n_disk, 3)`` in mm; ``mouth_nodes``
        and ``bell_nodes`` are the indices of the disk nodes at axial=0 and
        axial=L respectively.
    """
    geo = np.asarray(geo, dtype=float)
    x_geo = geo[:, 0]
    d_geo = geo[:, 1]
    bore_length = float(x_geo[-1] - x_geo[0])

    s_axial = np.linspace(0.0, bore_length, n_axial)
    d_axial = np.interp(s_axial + x_geo[0], x_geo, d_geo)
    r_axial = d_axial / 2.0

    cl3, _tangent, e1, e2 = _embed_centerline(centerline, bore_length, n_axial)
    nodes_2d, disk_tris = _disk_template(n_radial, n_circ)
    n_disk = len(nodes_2d)

    # Sweep disk into 3D
    points = (
        cl3[:, None, :]
        + r_axial[:, None, None] * nodes_2d[None, :, 0:1] * e1[:, None, :]
        + r_axial[:, None, None] * nodes_2d[None, :, 1:2] * e2[:, None, :]
    )
    points_flat = points.reshape(-1, 3)

    # Build tets: each prism (disk triangle x axial layer) -> 3 tets.
    # Diagonal convention: on every rectangular face (vi, vj, vi', vj'),
    # the diagonal goes from min(vi, vj) on the lo level to max(vi, vj) on the
    # hi level. Equivalently: sort each disk triangle's vertex indices
    # ascending and apply the fixed split (a,b,c,c'),(a,b,c',b'),(a,b',c',a').
    # This is mesh-conforming across all prism boundaries (see docstring).
    disk_tris_sorted = np.sort(disk_tris, axis=1)
    n_tris = len(disk_tris_sorted)
    tets = np.empty((n_axial - 1, n_tris, 3, 4), dtype=np.int64)
    for i in range(n_axial - 1):
        base_lo = i * n_disk
        base_hi = (i + 1) * n_disk
        a = base_lo + disk_tris_sorted[:, 0]
        b = base_lo + disk_tris_sorted[:, 1]
        c = base_lo + disk_tris_sorted[:, 2]
        ap = base_hi + disk_tris_sorted[:, 0]
        bp = base_hi + disk_tris_sorted[:, 1]
        cp = base_hi + disk_tris_sorted[:, 2]
        tets[i, :, 0, :] = np.stack([a, b, c,  cp], axis=1)
        tets[i, :, 1, :] = np.stack([a, b, cp, bp], axis=1)
        tets[i, :, 2, :] = np.stack([a, bp, cp, ap], axis=1)
    tets = tets.reshape(-1, 4)
    tets = _fix_tet_orientation(points_flat, tets)

    mesh = fem.MeshTet(
        np.ascontiguousarray(points_flat.T),
        np.ascontiguousarray(tets.T),
    )

    mouth_nodes = np.arange(0, n_disk, dtype=np.int64)
    bell_nodes = np.arange((n_axial - 1) * n_disk, n_axial * n_disk, dtype=np.int64)
    return mesh, points_flat, mouth_nodes, bell_nodes


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def fem3d(
    geo: np.ndarray,
    frequencies: np.ndarray,
    p: float = 1.225,
    n: float = 1.81e-5,
    c: float = 343.0,
    n_axial: int = 80,
    n_radial: int = 3,
    n_circ: int = 10,
    centerline: Optional[np.ndarray] = None,
):
    """3D FEM acoustic impedance solver (mm coordinates).

    Solves the 3D Helmholtz equation over the swept bore volume with rigid
    walls (natural Neumann), uniform pressure source at the mouth, and open
    pressure-release at the bell (Dirichlet=0). The bend, if any, is encoded
    in the geometry; no curvature/shortcut correction is applied.
    """
    mesh, points_flat, mouth_nodes, bell_nodes = build_3d_mesh(
        geo, n_axial=n_axial, n_radial=n_radial, n_circ=n_circ, centerline=centerline,
    )

    element = fem.ElementTetP1()
    basis = fem.Basis(mesh, element)

    @fem.BilinearForm
    def stiffness(u, v, w):
        return dot(grad(u), grad(v))

    @fem.BilinearForm
    def mass(u, v, w):
        return u * v

    K = stiffness.assemble(basis)
    M = mass.assemble(basis)

    b_mouth = np.zeros(basis.N)
    b_mouth[mouth_nodes] = 1.0
    # Centre node of the mouth disk (index 0 in the disk template) is the
    # on-axis probe.
    mouth_axis_node = int(mouth_nodes[0])

    # Mapes-Riordan viscothermal damping with mean-radius proxy (same as 1D/2D)
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

        k_mm = k_complex_m / 1000.0
        k_sq = k_mm ** 2

        A = K - k_sq * M
        p_c = fem.solve(*fem.condense(A, b_mouth, D=bell_nodes))
        impedance[fi] = float(np.abs(p_c[mouth_axis_node]))

    return impedance, mesh, points_flat, mouth_nodes, bell_nodes


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------


class FiniteElementsModeling3D(AcousticSimulationInterface):
    """3D FEM simulator.

    Args:
        constants: optional :class:`AcousticConstants`. Defaults to the base
            class default.
        centerline: optional ``(M, 2)`` planar polyline (mm) in the YZ plane,
            or ``(M, 3)`` polyline; ``None`` for a straight bore.
        n_axial: number of axial stations (default 80).
        n_radial: radial layers per cross-section, >=2 (default 3).
        n_circ: circumferential nodes per radial ring, >=3 (default 10).

    Mesh size grows as ``n_axial * (1 + (n_radial-1) * n_circ)`` nodes.
    Defaults give ~1700 nodes / ~7k tets — well under a second per frequency
    solve on a modern laptop.
    """

    def __init__(
        self,
        constants=None,
        centerline: Optional[np.ndarray] = None,
        n_axial: int = 80,
        n_radial: int = 3,
        n_circ: int = 10,
    ):
        super().__init__(constants)
        self.centerline = centerline
        self.n_axial = n_axial
        self.n_radial = n_radial
        self.n_circ = n_circ
        self.last_mesh = None
        self.last_points = None
        self.last_mouth_nodes = None
        self.last_bell_nodes = None

    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array):
        impedance, mesh, points_flat, mouth_nodes, bell_nodes = fem3d(
            np.array(geo.geo),
            frequencies,
            p=self.constants.air_density,
            n=self.constants.dynamic_viscosity,
            c=self.constants.speed_of_sound,
            n_axial=self.n_axial,
            n_radial=self.n_radial,
            n_circ=self.n_circ,
            centerline=self.centerline,
        )
        self.last_mesh = mesh
        self.last_points = points_flat
        self.last_mouth_nodes = mouth_nodes
        self.last_bell_nodes = bell_nodes
        return impedance
