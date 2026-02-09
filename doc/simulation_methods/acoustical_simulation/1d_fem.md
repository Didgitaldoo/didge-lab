# 1D Finite Element Method

## 1. The Governing Equation: Helmholtz Equation

The code models the didgeridoo using the Webster Horn Equation, which is a 1D version of the Helmholtz equation that accounts for a changing cross-sectional area $S(x)$.
The physics follows this second-order differential equation for acoustic pressure $p$:
$$\frac{1}{S(x)} \frac{d}{dx} \left( S(x) \frac{dp}{dx} \right) + k^2 p = 0$$
$S(x)$: Cross-sectional area at position $x$.
$k$: The complex wave number ($k = \frac{\omega}{c}$).
$p$: Acoustic pressure.

## 2. The Weak Form (Finite Element Formulation)

To solve this with FEM, the code converts the differential equation into a "weak form" by multiplying by a test function $v$ and integrating over the length of the tube $L$. After applying integration by parts, we get:
$$\int_0^L S(x) \frac{dp}{dx} \frac{dv}{dx} dx - k^2 \int_0^L S(x) p v dx = 0$$
In the code, this is split into two bilinear forms:
Stiffness Matrix ($K$): get_area(w.x[0]) * dot(grad(u), grad(v)) — Represents the spatial variation of pressure.
Mass Matrix ($M$): get_area(w.x[0]) * u * v — Represents the "storage" of acoustic energy in the volume.

## 3. Discretization and Linear Algebra

The code uses skfem to divide the instrument into 600 small linear elements (ElementLineP1). This turns the continuous calculus problem into a discrete matrix equation:
$$(K - k^2 M) \mathbf{p} = \mathbf{b}$$
$K$ and $M$ are global matrices assembled from the segment geometries.
$\mathbf{p}$ is the vector of unknown pressures at each node.
$\mathbf{b}$ is the "source" vector (the input from your lips at the mouthpiece).

## 4. Complex Wave Number and Damping

In an ideal world, $k = \omega / c$. However, without damping, the resonances would be infinitely sharp (mathematically singular). The code introduces an imaginary component to $k$ to simulate viscothermal losses:
k = (omega / c) - 1j * (2e-6 * np.sqrt(f))
This imaginary term represents energy absorbed by the walls of the didgeridoo. The $\sqrt{f}$ dependency is mathematically accurate for boundary layer losses in acoustic tubes.

## 5. Boundary Conditions

The code defines how the sound behaves at both ends:
The Mouthpiece ($x=0$): It sets a source term b_mouth[mouth_indices] = 1.0. This effectively simulates a unit volume velocity input.
The Bell ($x=L$): It sets p = 0 (Dirichlet condition) using bell_dofs. In physics, this is an "open end" approximation where pressure drops to zero because the tube meets the infinite atmosphere.

## 6. Calculating Impedance

Acoustic Impedance $Z$ is defined as the ratio of Pressure $p$ to Volume Velocity $U$:
$$Z = \frac{p}{U}$$
Because the code sets the input source ($U$) to a constant $1.0$, the resulting pressure value at the first node (p[mouth_indices[0]]) is the impedance. The impedance_magnitude is simply the absolute value of this complex result.



