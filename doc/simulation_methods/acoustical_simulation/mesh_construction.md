# Mesh Construction

This note describes how the 1D, 2D, and 3D FEM solvers in DidgeLab build
their meshes from a bore profile (and, for the bent-bore variants, a
centerline curve). The TLM doesn't have a mesh — it operates segment by
segment on the `Geo` list directly — so this document only covers the
FEM backends.

All coordinates are in **millimetres** throughout. Wave numbers are
converted from m$^{-1}$ to mm$^{-1}$ inside each solver before assembly.

## 1. Input: the Bore Profile

Every backend takes a `Geo` object whose `.geo` attribute is a list of
`[x_mm, d_mm]` pairs along the bore axis. The mesh builders consume this
as a plain numpy array; the bore length is `geo[-1, 0] - geo[0, 0]`, and
the diameter at axial position `x` is `np.interp(x, geo[:,0], geo[:,1])`.

## 2. 1D FEM (`fem1d.py`)

The simplest mesh. The bore is laid out along the $x$-axis as a single line:

```
mesh = fem.MeshLine(np.linspace(0, x_coords[-1], 600))
```

600 equally-spaced nodes; one P1 element per interval. The cross-sectional
area enters the weak form through the `get_area(x)` callback:

$$A(x) = \pi\,\bigl(d(x)/2\bigr)^2,$$

where $d(x)$ is interpolated from the `geo` array at each quadrature point.
The mesh itself doesn't change shape — only the integrand's area weight does.
A bent centerline has no effect on the 1D FEM.

## 3. 2D FEM (`fem2d.py`)

The bore is a planar strip in the *meridional half-plane* $(r, z)$, where
$r$ is the perpendicular distance from the centerline and $z$ the axial
coordinate. By default the centerline is the $x$-axis; with a bent
centerline it's an arbitrary planar polyline.

### 3.1 Straight Strip

For a straight bore, `build_2d_mesh(geo, ...)` builds a structured
`n_axial × n_radial` grid:

* **Axial sampling.** `s_uniform = np.linspace(0, bore_length, n_axial)`,
  with diameters interpolated at each station: `d_axial = np.interp(...)`.
* **Radial sampling.** A half strip uses offsets `np.linspace(0, 1, n_radial)`,
  so node $(i, j)$ sits at perpendicular distance
  $\,r_\text{perp}[i,j] = j/(n_\text{radial}-1) \cdot d_\text{axial}[i]/2$.
  A full strip uses offsets `np.linspace(-1, 1, n_radial)`, mirroring across
  the centerline — used only for visualisation, since the axisymmetric solve
  needs the half strip.
* **Triangulation.** Each axial quad $\{(i,j),(i+1,j),(i,j+1),(i+1,j+1)\}$ is
  split into two `ElementTriP1` triangles.
* **Boundary node sets.** `mouth_nodes = arange(0, n_radial)` (axial station 0),
  `bell_nodes = arange((n_axial-1)*n_radial, n_axial*n_radial)` (axial station
  $L$). The axis ($j=0$) and wall ($j=n_\text{radial}-1$) carry natural
  Neumann BCs.
* **Radial weight field.** `r_perp_flat` is returned alongside the mesh and
  used as the $r$ weight in the axisymmetric forms — this is the single most
  important detail of the 2D FEM (see `2d_fem.md`).

### 3.2 Bent Strip

A bent centerline is an `(M, 2)` polyline in millimetres (typically loaded
from a `curve.csv` in the YZ plane). The construction is the same swept-disk
idea applied to a curve instead of a line:

1. **Rescale arc length.** Compute the polyline's arc length, then scale the
   curve so its arc matches a target (`bore_length * centerline_scale`, with
   `centerline_scale=1.0` by default).
2. **Resample by arc.** `_resample_centerline(...)` puts `n_axial` points
   evenly along the *arc* of the (possibly irregular) input polyline. From
   this point on, axial parameter $s$ runs from 0 to bore length uniformly.
3. **Per-station normals.** `_normals_from_centerline(...)` computes the
   tangent via `np.gradient`, then rotates it by 90° (CCW) to get an
   in-plane unit normal.
4. **Sweep the radial offsets.** Node $(i, j)$ sits at
   `cl[i] + normals[i] * offset[j] * d_axial[i] / 2`. The `r_perp` field is
   still the absolute perpendicular offset (independent of the bend); the
   axisymmetric weight is applied *around the curved centerline*, not around
   a fixed axis.

`length_basis='mean_wall'` iteratively rescales the centerline so the mean
of the two wall polylines' arc lengths matches `bore_length` (a small
correction relative to `length_basis='centerline'`, only relevant when
varying $d$ makes the mean-wall identity discrete-imperfect).

For tight bends, `build_2d_mesh` itself does not check for self-intersection;
the caller is expected to trim such regions (see the `CURVE_END_FRAC` knob
in the example notebooks).

## 4. 3D FEM (`fem3d.py`)

The 3D mesh is a tetrahedral volume built by sweeping a 2D disk template
along the (possibly bent) centerline.

### 4.1 Disk Template

A 2D unit disk is built once per simulator instance by `_disk_template`:

* one centre node at $(0,0)$,
* $n_\text{circ}$ nodes on each non-centre radial ring at radii
  $r = k/(n_\text{radial}-1)$ for $k = 1, \dots, n_\text{radial}-1$,
* triangulation:
  * inner fan: $n_\text{circ}$ triangles connecting the centre to ring 1,
  * outer ring quads (between consecutive radial rings), each split into two
    triangles.

This gives $1 + (n_\text{radial}-1)\,n_\text{circ}$ nodes and
$n_\text{circ} + 2(n_\text{radial}-2)\,n_\text{circ}$ triangles per
cross-section. Triangles are oriented counter-clockwise in the
template's $+z$ convention.

### 4.2 Centerline Embedding and Bishop Frame

`_embed_centerline` produces the per-station frame used to place the disk
in 3D.

* **Straight bore.** Centerline along $x$; cross-section spanned by $y$ and
  $z$. No bending — same frame at every station.
* **Bent bore.** A `(M, 2)` planar polyline is embedded in the YZ plane
  (matching `curve.csv`); a `(M, 3)` polyline is used as-is. The polyline
  is rescaled to bore length and resampled to `n_axial` arc-uniform points.
* **Tangents.** `tangent[i] = np.gradient(cl3, axis=0)[i]`, normalised.
* **Cross-section basis (e1, e2).** A **Bishop frame** (parallel-transported
  frame) is used instead of the Frenet frame, because the Frenet frame is
  ill-defined when curvature vanishes and twists discontinuously around
  inflection points. The seed `e1[0]` is the world axis least aligned with
  `tangent[0]`, projected to be perpendicular to `tangent[0]`. For each
  subsequent station, `e1[i]` is obtained by rotating `e1[i-1]` from
  `tangent[i-1]` to `tangent[i]` via the **Rodrigues rotation formula**,
  using the minimum-rotation axis $\text{tangent}_{i-1} \times \text{tangent}_{i}$.
  After each step the result is re-orthogonalised against `tangent[i]`
  to suppress drift. Finally `e2 = tangent × e1`.

This keeps the cross-section orthogonal to the local tangent everywhere
without spurious rolling. For a planar curve, the Bishop frame coincides
with the natural in-plane / out-of-plane decomposition.

### 4.3 Sweep to 3D Points

With the frame in hand, every disk node is placed in 3D:

```
points[i, j] = cl3[i] + r_axial[i] * (nodes_2d[j,0]*e1[i] + nodes_2d[j,1]*e2[i])
```

where `r_axial[i] = d_axial[i] / 2` is the local bore radius. The result is
flattened into `points_flat` of shape `(n_axial * n_disk, 3)`.

### 4.4 Prism-to-Tet Split

Adjacent cross-sections (axial stations $i$ and $i+1$) form one layer of
triangular **prisms**: every disk triangle $(a, b, c)$ at level $i$
pairs with its image $(a', b', c')$ at level $i+1$ to make a prism.
A prism splits into **3 tetrahedra**.

To keep the global mesh **conforming** (adjacent tets share full faces),
the diagonals on the prism's three rectangular faces must be consistent
across every prism that shares one of them. The convention used here is:
on a rectangular face with corners $(v_i, v_j, v_i', v_j')$, the splitting
diagonal goes from `min(v_i, v_j)` at the low level to `max(v_i, v_j)` at
the high level. Equivalently, each disk triangle's vertex indices are
sorted ascending (`a < b < c`) before the split, and the fixed pattern

* tet 0: $(a, b, c, c')$,
* tet 1: $(a, b, c', b')$,
* tet 2: $(a, b', c', a')$

is applied. Any prism that shares a rectangular face with a neighbour
will see the same diagonal regardless of how the triangle vertices were
originally labelled, because the rule only depends on the actual node
indices.

After the split, `_fix_tet_orientation` flips any tetrahedron whose
signed volume is negative (which can happen when the sort reverses the
disk-template orientation), so all tets are positively oriented.

### 4.5 Final skfem Mesh

```
mesh = fem.MeshTet(
    np.ascontiguousarray(points_flat.T),   # shape (3, N_nodes)
    np.ascontiguousarray(tets.T),          # shape (4, N_tets)
)
```

`mouth_nodes` is the disk at axial 0; `bell_nodes` is the disk at
axial $L$. The centre node of `mouth_nodes` (index 0) is the on-axis
probe used to read out the impedance.

## 5. Sanity Checks

Each backend's mesh has a few invariants worth knowing about:

* **1D FEM.** Mesh is just `np.linspace`; the only thing that can vary is
  the area callback. Resolution is fixed at 600 nodes.
* **2D FEM.** Total triangle count is $2\,(n_\text{axial}-1)\,(n_\text{radial}-1)$.
  Node count is $n_\text{axial}\,n_\text{radial}$ (half strip) or
  same again (full strip). `r_perp` ranges from 0 (axis) to $d(x)/2$ (wall).
* **3D FEM.** Tet count is $3\,(n_\text{axial}-1)\,T_\text{disk}$, where
  $T_\text{disk} = n_\text{circ} + 2\,(n_\text{radial}-2)\,n_\text{circ}$.
  Node count is $n_\text{axial}\,\bigl(1 + (n_\text{radial}-1)\,n_\text{circ}\bigr)$.
  All tets are positively oriented after `_fix_tet_orientation`.

The bent-bore variants additionally rely on the centerline curve being
non-self-intersecting after the cross-section sweep; the example
notebooks under `doc/examples/bent_shapes/` show the trim knob and how
to inspect raw curves before meshing.
