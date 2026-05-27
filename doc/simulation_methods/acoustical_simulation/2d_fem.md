# 2D Finite Element Method (axisymmetric)

## 1. The Governing Equation: Axisymmetric Helmholtz

A didgeridoo is (to good approximation) a body of revolution: a tube whose
cross-section is a disk of varying radius $R(z)=d(z)/2$ along its axis.
Exploiting that rotational symmetry, the 3D Helmholtz equation reduces to a
2D equation in the meridional half-plane $(r,z)$ — $r$ being the perpendicular
distance from the centerline, $z$ the axial coordinate:

$$\frac{1}{r}\frac{\partial}{\partial r}\!\left(r\,\frac{\partial p}{\partial r}\right)
  + \frac{\partial^2 p}{\partial z^2}
  + k^2 p = 0$$

This is just the 3D Laplacian in cylindrical coordinates with $\partial/\partial\theta=0$.
The implementation lives in `didgelab/sim/fem2d.py`.

## 2. Why a Radial Weight?

In Cartesian 2D, the volume element is $\mathrm{d}A=\mathrm{d}r\,\mathrm{d}z$.
In cylindrical 3D with rotational symmetry, integrating over $\theta\!\in\![0,2\pi)$
contributes a factor of $2\pi r$, so the natural volume element of the meridional
half-plane is

$$\mathrm{d}V = 2\pi\,r\,\mathrm{d}r\,\mathrm{d}z.$$

Dropping the constant $2\pi$, **every integral in the weak form picks up an
$r$ weight**. This is the single most important detail of the 2D FEM. A
"plain" 2D Helmholtz on the same triangular mesh describes a planar
waveguide — a flat slab — and gives wrong resonance frequencies (by tens to
hundreds of cents for typical bore profiles).

## 3. The Weak Form

Multiply by a test function $v$, integrate over the half-meridional domain
$\Omega=\{(r,z)\!:\!0\le r\le R(z),\;0\le z\le L\}$, integrate by parts:

$$\int_\Omega r\,\nabla p\cdot\nabla v\,\mathrm{d}A
 \;-\; k^2\int_\Omega r\,p\,v\,\mathrm{d}A
 \;=\; \text{source}.$$

In the code, this is split into two `r`-weighted bilinear forms:

| Matrix | Form | Code |
|---|---|---|
| Stiffness $K$ | $\int r\,\nabla u\cdot\nabla v$ | `w['r'] * dot(grad(u), grad(v))` |
| Mass $M$      | $\int r\,u\,v$                  | `w['r'] * u * v`                  |

The weight `w['r']` is a P1 finite-element field giving each quadrature point's
perpendicular distance to the centerline. It is built by
`build_2d_mesh(..., full_strip=False)`, which returns the `r_perp` array
alongside the mesh.

## 4. Discretisation

The bore is meshed as a structured grid of $N_{\text{axial}} \times N_{\text{radial}}$
nodes (default $200\times10$) over the half-meridional plane
$r\in[0,R(z)]$. Each quad is split into two `ElementTriP1` triangles, giving
$2(N_{\text{axial}}-1)(N_{\text{radial}}-1)$ elements.

The discrete problem is the same matrix equation as the 1D FEM:

$$(K - k^2 M)\,\mathbf{p} = \mathbf{b}.$$

`K`, `M` are global sparse matrices assembled from the `r`-weighted forms.
`b` is the source vector, with `1.0` at every mouth node.

## 5. Complex Wave Number and Damping

The same Mapes-Riordan boundary-layer model used in the TLM and 1D FEM is
applied here: a frequency-dependent complex wave number

$$k_{\text{m}} = \frac{\omega}{c}\!\left(1 + \frac{1.045}{r_v}\right)
            \;-\; i\,\frac{\omega}{c}\!\cdot\!\frac{1.045}{r_v},
\qquad r_v = R_{\text{avg}}\sqrt{\frac{\rho\,\omega}{\mu}}.$$

$R_{\text{avg}}$ is the mean bore radius (in metres). The result is converted
from m$^{-1}$ to mm$^{-1}$ before squaring, because the mesh lives in mm.

## 6. Boundary Conditions

| Boundary | Where | Condition | How |
|---|---|---|---|
| Mouth   | $z=0$         | source                          | `b_mouth[mouth_nodes] = 1.0` |
| Bell    | $z=L$         | Dirichlet $p=0$ (open end)      | `fem.condense(..., D=bell_nodes)` |
| Axis    | $r=0$         | natural Neumann (symmetry)      | automatic — $r=0$ makes the surface integral vanish |
| Wall    | $r=R(z)$      | natural Neumann (rigid wall)    | automatic — default for no specified flux |

The axis BC is the elegant payoff of the $r$ weighting: the singular
$1/r$ term in the strong form disappears in the weak form because the
volume element kills it, so no special treatment of $r=0$ is needed.

## 7. Bent Centerlines

`build_2d_mesh(..., centerline=...)` extrudes the bore along an arbitrary
planar polyline instead of the straight $z$-axis. Mesh nodes are placed at
`cl[i] + normals[i] * offset * d_axial[i] * 0.5`, and `r_perp` is just the
absolute offset times $d/2$ — independent of the curve's shape.

Physically this is a *swept-disk* approximation: every cross-section is a
disk of radius $d(s)/2$ perpendicular to the local centerline. The
axisymmetric weight is applied around the (curved) centerline rather than
around a fixed axis. For modest bends the approximation is sound; for tight
loops it ignores the inner/outer-radius asymmetry of the toroidal Jacobian.

See `src/experiments/2dfem/fem2d_bent_didge.ipynb` for a worked example
(straight vs. curve.csv-bent didgeridoo).

## 8. Calculating Impedance

The pressure response is read at the **on-axis mouth node** (the node at
$r=0,\,z=0$, which by construction has linear index `mouth_nodes[0]`):

$$|Z| \;\approx\; \bigl|p(0, 0)\bigr|.$$

Because the input source $U$ is set to a uniform unit value at the mouth, the
ratio $p/U$ collapses to $p$ itself. Absolute magnitudes are not comparable
to the TLM (different normalisation), but the *peak frequencies* are: by
construction the lossless modes of the $r$-weighted operator are identical to
those of the 1D Webster horn

$$\frac{1}{S(z)}\frac{\mathrm{d}}{\mathrm{d}z}\!\left(S(z)\frac{\mathrm{d}p}{\mathrm{d}z}\right) + k^2 p = 0,$$

since $\int_0^{R(z)}r\,\mathrm{d}r = R(z)^2/2 = S(z)/(2\pi)$. Numerically, on
a realistic didgeridoo bore the 2D FEM's first peak lands within ~4 cents of
the TLM result (see `tests/test_acoustical_simulation.py
::test_all_backends_agree_on_fundamental`).

## 9. When to Use Which Backend

| Backend       | Best for                                                  |
|---|---|
| `tlm_cython`  | Anything performance-sensitive; default in DidgeLab.       |
| `tlm_python`  | Readable reference for debugging the TLM physics.          |
| `1d_fem`      | Cross-check of the TLM with an independent solver.         |
| `2d_fem`      | Bent-bore experiments and visualising the mode shapes.     |

The 2D FEM is roughly an order of magnitude slower than the 1D FEM per
frequency (it solves a sparse system that's $N_{\text{radial}}\!\times$
larger). For sweeps over many frequencies you usually want one of the TLM
backends; the 2D FEM earns its cost only when the extra spatial information
matters.
