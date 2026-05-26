import numpy as np
import skfem as fem
from skfem.helpers import dot, grad

from .sim_interface import AcousticSimulationInterface
from ..geo import Geo


def fem1d(geo, frequencies, p=1.225, n=1.81e-5, c=343.0):
    """
    1D FEM acoustic impedance solver.
    Expects p (density), n (viscosity), and c (speed of sound) in standard SI units.
    """
    x_coords = geo[:, 0]
    diameters = geo[:, 1]

    def get_area(x):
        d = np.interp(x, x_coords, diameters)
        return np.pi * (d / 2.0) ** 2

    # Calculate average radius in METERS for the viscothermal damping calculation
    avg_d_m = np.mean(diameters) / 1000.0
    r_avg = avg_d_m / 2.0

    # 2. Mesh and Basis
    mesh = fem.MeshLine(np.linspace(0, x_coords[-1], 600))
    element = fem.ElementLineP1()
    basis = fem.Basis(mesh, element)

    # 3. Define Forms
    def stiffness_fun(u, v, w):
        return get_area(w.x[0]) * dot(grad(u), grad(v))

    def mass_fun(u, v, w):
        return get_area(w.x[0]) * u * v

    K = fem.BilinearForm(stiffness_fun).assemble(basis)
    M = fem.BilinearForm(mass_fun).assemble(basis)

    # 4. Boundary Setup
    bell_dof_data = basis.get_dofs(lambda x: np.isclose(x[0], x_coords[-1])).nodal
    bell_dofs = (
        np.concatenate([v for v in bell_dof_data.values()])
        if isinstance(bell_dof_data, dict)
        else bell_dof_data
    )

    b_mouth = np.zeros(basis.N)
    mouth_dof_data = basis.get_dofs(lambda x: np.isclose(x[0], 0)).nodal
    if isinstance(mouth_dof_data, dict):
        mouth_indices = np.concatenate([v for v in mouth_dof_data.values()]).astype(int)
    else:
        mouth_indices = np.array(mouth_dof_data).astype(int)

    b_mouth[mouth_indices] = 1.0

    # 5. Frequency Sweep
    impedance_magnitude = []

    for f in frequencies:
        omega = 2 * np.pi * f

        # 1. Viscothermal parameter (r_v) matching Mapes-Riordan physics
        # r_v = r * sqrt(rho * omega / mu)
        r_v = r_avg * np.sqrt(p * omega / n)

        # 2. Complex wave number in standard SI units (m^-1)
        kw_m = omega / c
        k_real_m = kw_m * (1.0 + 1.045 / r_v)          # Phase velocity reduction
        k_imag_m = kw_m * (1.045 / r_v)                # Attenuation (damping)

        # Subtract imaginary part for e^(j*omega*t) convention damping
        k_complex_m = k_real_m - 1j * k_imag_m

        # 3. Convert wave number to mm^-1 to match the skfem coordinate system
        k_mm = k_complex_m / 1000.0
        k_sq = k_mm ** 2

        A = K - k_sq * M

        # Solve for complex pressure p_c
        p_c = fem.solve(*fem.condense(A, b_mouth, D=bell_dofs))

        z_val = p_c[mouth_indices[0]]
        impedance_magnitude.append(np.abs(z_val))

    return np.array(impedance_magnitude)


class FiniteElementsModeling1D(AcousticSimulationInterface):
    """
    FiniteElementsModeling1D simulator.

    Inherits the AcousticConstants-based constructor from
    AcousticSimulationInterface; pass ``constants=AcousticConstants(...)`` to
    override the default (currently the breath-temperature moist-air mix
    computed by ``compute_moist_air_properties``). The constants flow as SI
    units (kg/m^3, Pa*s, m/s) into ``fem1d``, which handles the m->mm wave-
    number conversion internally.
    """

    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array):
        """Return list of impedance magnitudes at each frequency in Hz."""
        impedances = fem1d(
            np.array(geo.geo),
            frequencies,
            p=self.constants.air_density,
            n=self.constants.dynamic_viscosity,
            c=self.constants.speed_of_sound,
        )
        return impedances
