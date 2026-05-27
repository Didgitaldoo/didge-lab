# 1D Finite Element Method

Source: [`src/didgelab/sim/fem1d.py`](../../../src/didgelab/sim/fem1d.py).

## 1. The Governing Equation: Webster Horn

The code models the didgeridoo using the **Webster horn equation** — a 1D
reduction of the Helmholtz equation that accounts for a varying cross-sectional
area $S(x)$ along the bore axis:

$$\frac{1}{S(x)} \frac{\mathrm{d}}{\mathrm{d}x}\!\left( S(x)\,\frac{\mathrm{d}p}{\mathrm{d}x} \right) + k^2 p = 0$$

- $S(x) = \pi\,(d(x)/2)^2$ — cross-sectional area at axial position $x$, in mm². The diameter $d(x)$ is linearly interpolated from the input geometry.
- $k$ — complex wave number (defined in §4), in mm$^{-1}$.
- $p$ — complex acoustic pressure.

The mesh and all geometric quantities are in **mm**. The physical constants
($\rho$, $\mu$, $c$) are passed in **SI units** (kg/m³, Pa·s, m/s); the only
place units cross is when the wave number is converted from m$^{-1}$ to mm$^{-1}$
before squaring.

## 2. The Weak Form

Multiplying by a test function $v$, integrating over the bore length $L$, and
integrating by parts gives

$$\int_0^L S(x)\,\frac{\mathrm{d}p}{\mathrm{d}x}\,\frac{\mathrm{d}v}{\mathrm{d}x}\,\mathrm{d}x
 \;-\; k^2 \int_0^L S(x)\,p\,v\,\mathrm{d}x \;=\; 0.$$

In the code this is split into two area-weighted bilinear forms:

| Matrix | Form | Code |
|---|---|---|
| Stiffness $K$ | $\int S(x)\,p'\,v'$ | `get_area(w.x[0]) * dot(grad(u), grad(v))` |
| Mass $M$      | $\int S(x)\,p\,v$   | `get_area(w.x[0]) * u * v`                  |

## 3. Discretisation

The bore is meshed as **600 linear `ElementLineP1` elements** evenly spaced
from $x=0$ to $x=L$:

```python
mesh = fem.MeshLine(np.linspace(0, x_coords[-1], 600))
```

That turns the weak form into the sparse matrix equation

$$(K - k^2 M)\,\mathbf{p} = \mathbf{b},$$

with $\mathbf{p}$ the unknown pressures at each node and $\mathbf{b}$ the
source vector.

## 4. Complex Wave Number and Damping (Mapes-Riordan)

Without damping the resonances would be infinitely sharp (the matrix is
singular at every eigenfrequency). The code uses the same Mapes-Riordan
boundary-layer model as the TLM and the 2D FEM: a frequency- and geometry-
dependent **viscothermal parameter**

$$r_v \;=\; R_{\text{avg}}\,\sqrt{\frac{\rho\,\omega}{\mu}}$$

where $R_{\text{avg}}$ is the mean bore radius (in metres, computed from the
mean of the input diameters). $r_v$ then enters the complex wave number as

$$k_{\text{m}} \;=\; \frac{\omega}{c}\!\left(1 + \frac{1.045}{r_v}\right)
              \;-\; i\,\frac{\omega}{c}\,\frac{1.045}{r_v}.$$

The real part is a small phase-velocity reduction; the imaginary part is the
attenuation. The minus sign on the imaginary part matches the $e^{j\omega t}$
time convention used by the linear solver.

The result is in m$^{-1}$. The mesh is in mm, so before assembling $A=K-k^2 M$:

```python
k_mm = k_complex_m / 1000.0
k_sq = k_mm ** 2
```

This is the only place a unit conversion happens in the solver.

## 5. Boundary Conditions

| Boundary | $x$ | Condition | How |
|---|---|---|---|
| Mouthpiece | $x=0$ | source: unit volume velocity | `b_mouth[mouth_indices] = 1.0` |
| Bell       | $x=L$ | Dirichlet $p=0$ (open end)   | `fem.condense(A, b, D=bell_dofs)` |

The Dirichlet bell condition is the standard idealisation of an open pipe end
radiating into an infinite atmosphere — pressure drops to zero there.

## 6. Calculating Impedance

Acoustic impedance is

$$Z = \frac{p}{U}.$$

Because the input source $U$ is set to a constant `1.0` at the mouth, the
returned spectrum is simply

$$|Z(f)| \;=\; \bigl|p_c(\text{mouth\_indices[0]})\bigr|$$

for each frequency $f$ in the sweep. Absolute magnitudes are not comparable
to the TLM (different normalisations); peak frequencies are — and they agree
with the TLM to within ~5 cents on a realistic didgeridoo geometry (see
`tests/test_acoustical_simulation.py::test_all_backends_agree_on_fundamental`).

## 7. The Simulator Class

`FiniteElementsModeling1D` is the `AcousticSimulationInterface` wrapper:

```python
class FiniteElementsModeling1D(AcousticSimulationInterface):
    def get_impedance_spectrum(self, geo, frequencies):
        return fem1d(
            np.array(geo.geo), frequencies,
            p=self.constants.air_density,
            n=self.constants.dynamic_viscosity,
            c=self.constants.speed_of_sound,
        )
```

It inherits the constants-handling constructor from the base class, so
`FiniteElementsModeling1D(constants=AcousticConstants(speed_of_sound=340.0))`
overrides whichever air properties you like. The default (when constants is
None) is `compute_moist_air_properties()` — 28 °C, 100 % relative humidity,
1 atm — i.e. saturated breath.
