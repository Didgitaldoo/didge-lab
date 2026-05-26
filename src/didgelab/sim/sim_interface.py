"""
Abstract interface for acoustic simulation backends in DidgeLab.

Any simulator (e.g. TLM Python or Cython) implements get_impedance_spectrum(geo, frequencies).
Physical constants used by the simulators are bundled in `AcousticConstants` and
injected per simulator instance via the constructor. Defaults match the values
historically hard-coded in each backend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
import math

from ..geo import Geo


@dataclass(frozen=True)
class AcousticConstants:
    """Physical constants for acoustic simulation. SI units (m, s, kg).

    air_density: kg/m^3 (was `p`)
    dynamic_viscosity: Pa*s (was `n`)
    speed_of_sound: m/s (was `c`)
    """

    air_density: float = 1.2929
    dynamic_viscosity: float = 1.708e-5
    speed_of_sound: float = 343.37


class AcousticSimulationInterface(ABC):
    """Interface for computing acoustic impedance spectrum of a didgeridoo geometry."""

    def __init__(self, constants: Optional[AcousticConstants] = None):
        self.constants = constants if constants is not None else AcousticConstants()

    @abstractmethod
    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array) -> np.array:
        """Return impedance values at each frequency in Hz for the given geometry."""
        pass

def compute_air_properties(temp_celsius, pressure_pa=101325.0):
    """
    Computes air density, dynamic viscosity, and speed of sound 
    based on temperature (Celsius) and absolute pressure (Pascals).
    
    Returns:
        p (float): Air density in kg/m^3
        n (float): Dynamic viscosity in Pa*s
        c (float): Speed of sound in m/s
    """
    # Thermodynamic constants for dry air
    R_specific = 287.058  
    gamma = 1.402         
    
    # Sutherland's Law constants
    mu_0 = 1.716e-5       
    T_0 = 273.15          
    S = 110.4             
    
    # Convert Celsius to Kelvin
    T_K = temp_celsius + 273.15
    
    # Speed of sound (c)
    c = math.sqrt(gamma * R_specific * T_K)
    
    # Air density (p)
    p = pressure_pa / (R_specific * T_K)
    
    # Dynamic viscosity (n)
    n = mu_0 * ((T_K / T_0) ** 1.5) * ((T_0 + S) / (T_K + S))
    
    return AcousticConstants(
        air_density=p,
        dynamic_viscosity=n,
        speed_of_sound=c
    )
