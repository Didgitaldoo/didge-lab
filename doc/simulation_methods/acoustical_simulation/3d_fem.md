# 3D Finite Element Method

## 1. The Governing Equation: 3D Helmholtz

For an arbitrary bore geometry — including bends, twists, and asymmetric
cross-sections — the right equation is the full 3D Helmholtz equation:

$$\nabla^2 p + k^2 p = 0,$$

solved on the actual volume of the air column. No rotational symmetry is
assumed; no Webster reduction to an axial parameter. The bend lives in the
geometry of the mesh, and the wave operator sees it directly. The
implementation lives in `didgelab/sim/fem3d.py`.

This is the **gold-standard physics** in this codebase. The TLM, 1D FEM,
and 2D FEM all collapse a 3D problem to a 1D parameter (sometimes via an
axisymmetric 2D detour); the 3D FEM does not.

## 2. Weak Form

Multiply by a test function $v$, integrate over the bore volume $\Omega$,
integrate by parts:

$$\int_\Omega \nabla p \cdot \nabla v\,\mathrm{d}V
 \;-\; k^2 \int_\Omega p\,v\,\mathrm{d}V
 \;=\; \text{source}.$$

There is no radial weight — every quadrature integral is just $\mathrm{d}V$,
because the geometry is already 3D. The two bilinear forms reduce to:

| Matrix | Form | Code |
|---|---|---|
| Stiffness $K$ | $\int \nabla u \cdot \nabla v$ | `dot(grad(u), grad(v))` |
| Mass $M$      | $\int u\,v$                    | `u * v`                  |

## 3. Discretisation

The bore is meshed as a tetrahedral volume built by sweeping a 2D disk
template along the centerline (see
[`mesh_construction.md`](mesh_construction.md) for the full algorithm).
With `n_axial` axial stations, `n_radial` radial layers, and `n_circ`
circumferential nodes per non-center ring, each cross-section has
$1 + (n_\text{radial}-1)\,n_\text{circ}$ disk nodes, and the volume has

* $n_\text{axial}\,\bigl(1 + (n_\text{radial}-1)\,n_\text{circ}\bigr)$ nodes,
* roughly $3\,(n_\text{axial}-1)\,T_\text{disk}$ tetrahedra (where $T_\text{disk}$
  is the number of disk triangles).

The same matrix equation as the other FEM backends:

$$(K - k^2 M)\,\mathbf{p} = \mathbf{b}.$$

`fem.ElementTetP1` linear tetrahedra are used.

## 4. Complex Wave Number and Damping

The same Mapes-Riordan boundary-layer model used in the TLM and 1D/2D FEM
is applied. The frequency-dependent complex wave number uses the mean bore
radius $R_\text{avg}$ as a damping proxy — same expression, same units
conversion from m$^{-1}$ to mm$^{-1}$:

$$k_\text{m} = \frac{\omega}{c}\!\left(1 + \frac{1.045}{r_v}\right)
            \;-\; i\,\frac{\omega}{c}\!\cdot\!\frac{1.045}{r_v},
\qquad r_v = R_\text{avg}\sqrt{\frac{\rho\,\omega}{\mu}}.$$

A 3D solver could in principle resolve the actual local cross-section in
its damping, but for consistency with the other backends the mean-radius
proxy is kept here.

## 5. Boundary Conditions

| Boundary | Where                          | Condition                       | How |
|---|---|---|---|
| Mouth   | $z=0$ disk                       | source                          | `b_mouth[mouth_nodes] = 1.0` |
| Bell    | $z=L$ disk                       | Dirichlet $p=0$ (open end)      | `fem.condense(..., D=bell_nodes)` |
| Wall    | outer surface of the swept tube | natural Neumann (rigid wall)    | automatic — default for no specified flux |

The wall BC is implicit: any boundary face whose nodes are not in
`mouth_nodes` or `bell_nodes` carries no flux term in the weak form. Since
the outer ring of disk nodes lies on the bore surface and is not pinned, the
operator naturally treats those faces as rigid.

## 6. Bent Bores — No Curvature Correction

This is the headline advantage of the 3D solver. The 2D FEM is
*axisymmetric*: its $r$-weighted operator treats every cross-section as a
circle rotated about the centerline, so the curvature of the centerline
does not enter the wave equation — only the geometry of the planar 2D strip
does. To produce the bent-tube pitch shift in 2D you have to bolt on a
heuristic correction (`bent_shortcut`, scaling the squared wavenumber by
the inner/outer arc-length ratio).

The 3D FEM does not need any of that. A bent centerline produces a curved
3D volume, and the lowest-order acoustic mode shortcuts to the inside of
the bend because of the actual geometry. The pitch shift emerges from the
Helmholtz operator on the swept tetrahedral mesh, with no model parameter.

See [`mesh_construction.md`](mesh_construction.md) for how the disk is
parallel-transported (Bishop frame) along the centerline.

## 7. Calculating Impedance

The pressure response is read at the **on-axis mouth node** — the center
node of the mouth cross-section (disk index 0, which by construction sits
exactly on the centerline at $s=0$):

$$|Z| \;\approx\; \bigl|p(\text{mouth axis})\bigr|.$$

As with the 2D FEM, the source $U$ is set to a uniform unit value at the
mouth disk, so the ratio $p/U$ collapses to $p$. Absolute magnitudes are
not comparable across backends (different normalisations); peak
frequencies are.

For a straight bore, the 3D FEM agrees with the 2D FEM (and the 1D Webster
horn) to within a couple of cents on the lowest peaks. For a bent bore it
*does not* — by design, because it captures a physical effect the others
miss.

## 8. Mesh Convergence

The bent-bore pitch shift converges from above as the mesh refines, because
coarser axial discretisation makes each cross-section see a larger heading
change between adjacent disks, slightly amplifying the apparent shortcut.
A reasonable default (`n_axial=80`, `n_radial=3`, `n_circ=10`, ~1700 nodes,
~7k tets) sweeps ~600 frequencies in a few seconds and is accurate enough
for design iteration. Push the resolution up — `n_axial=120-150`,
`n_circ=12-16` — for a high-fidelity answer on a particular instrument.

## 9. When to Use Which Backend

| Backend       | Best for                                                                          |
|---|---|
| `tlm_cython`  | Anything performance-sensitive; default in DidgeLab.                              |
| `tlm_python`  | Readable reference for debugging the TLM physics.                                 |
| `1d_fem`      | Cross-check of the TLM with an independent solver.                                |
| `2d_fem`      | Cheap bent-bore experiments (with the shortcut heuristic), mode-shape pictures.   |
| `3d_fem`      | Predicting impedance of arbitrary bent bores from first principles, no fitting.   |

The 3D FEM is the slowest backend — roughly an order of magnitude slower
than the 2D FEM per frequency at comparable accuracy — because the
linear system is larger and denser. For straight bores all four other
backends are faster and equivalent; use the 3D FEM when the geometry of
the bend matters.

See `doc/examples/bent_shapes/fem3d_bent_didge.ipynb` for a worked example
(straight vs. curve.csv-bent didgeridoo) and a side-by-side comparison
with the 2D FEM.
