import numpy as np
import skfem as fem
from skfem.helpers import dot, grad
from .sim_interface import AcousticSimulationInterface
from ..geo import Geo

DEFAULT_C_MM_S = 343000.0  # Speed of sound in mm/s (matches 343.37 m/s up to legacy rounding)
c = DEFAULT_C_MM_S  # retained for backwards compatibility with anyone importing this name

def fem1d(geo, frequencies, c=DEFAULT_C_MM_S):


    x_coords = geo[:, 0]
    diameters = geo[:, 1]

    def get_area(x):
        d = np.interp(x, x_coords, diameters)
        return np.pi * (d / 2.0)**2

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
    # Bell end: Pressure = 0 (Dirichlet)
    bell_dof_data = basis.get_dofs(lambda x: np.isclose(x[0], x_coords[-1])).nodal
    # Extract indices from dictionary if necessary
    bell_dofs = np.concatenate([v for v in bell_dof_data.values()]) if isinstance(bell_dof_data, dict) else bell_dof_data

    # Mouthpiece: Setup the source vector
    b_mouth = np.zeros(basis.N)

    # FIXED: Safely extract nodal indices from the Dofs object (handling dict structure)
    mouth_dof_data = basis.get_dofs(lambda x: np.isclose(x[0], 0)).nodal
    if isinstance(mouth_dof_data, dict):
        mouth_indices = np.concatenate([v for v in mouth_dof_data.values()]).astype(int)
    else:
        mouth_indices = np.array(mouth_dof_data).astype(int)

    b_mouth[mouth_indices] = 1.0 

    # 5. Frequency Sweep
    impedance_magnitude = []

    # Add viscothermal damping to prevent infinite peaks and simulate wall friction
    # damping factor alpha is proportional to sqrt(frequency)
    for f in frequencies:
        omega = 2 * np.pi * f
        k = (omega / c) - 1j * (2e-6 * np.sqrt(f)) 
        k_sq = k**2
        
        A = K - k_sq * M
        
        # Solve for complex pressure p
        p = fem.solve(*fem.condense(A, b_mouth, D=bell_dofs))
        
        # Input Impedance Z = p_mouth / U_mouth. Since U=1, Z = p[mouth]
        z_val = p[mouth_indices[0]]
        impedance_magnitude.append(np.abs(z_val))

    impedance_magnitude = np.array(impedance_magnitude)
    return impedance_magnitude

class FiniteElementsModeling1D(AcousticSimulationInterface):

    """FiniteElementsModeling1D simulator.

    Reads the speed of sound from ``self.constants`` (SI, m/s) and converts to
    mm/s for the internal mm-based mesh. The viscothermal damping coefficient
    (2e-6) is kept hard-coded and is not configurable through this interface.
    """

    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array):

        """Return list of impedance magnitudes at each frequency in Hz."""
        c_mm_s = self.constants.speed_of_sound * 1000.0
        impedances = fem1d(np.array(geo.geo), frequencies, c=c_mm_s)
        return impedances