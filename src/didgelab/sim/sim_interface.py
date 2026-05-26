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
        self.constants = constants if constants is not None else compute_moist_air_properties()

    @abstractmethod
    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array) -> np.array:
        """Return impedance values at each frequency in Hz for the given geometry."""
        pass

def compute_moist_air_properties(temp_celsius=28.0, rel_humidity=1.0, pressure_pa=101325.0):
    """
    Computes density, dynamic viscosity, and speed of sound for moist air.
    
    Args:
        temp_celsius (float): Temperature in Celsius.
        rel_humidity (float): Relative humidity (0.0 to 1.0). Default is 1.0 (fully saturated breath).
        pressure_pa (float): Absolute atmospheric pressure in Pascals.
        
    Returns:
        p (float): Air density in kg/m^3
        n (float): Dynamic viscosity in Pa*s
        c (float): Speed of sound in m/s
    """
    # Thermodynamic constants
    R_d = 287.058     # Specific gas constant for dry air
    R_v = 461.495     # Specific gas constant for water vapor
    gamma = 1.402     # Heat capacity ratio for dry air
    
    # Sutherland's Law constants (for viscosity)
    mu_0 = 1.716e-5       
    T_0 = 273.15          
    S = 110.4             
    
    T_K = temp_celsius + 273.15
    
    # 1. Vapor Pressure Calculations (Tetens formula)
    p_sat = 610.78 * math.exp((17.27 * temp_celsius) / (temp_celsius + 237.3))
    p_v = rel_humidity * p_sat
    p_dry = pressure_pa - p_v
    
    # 2. Air Density (p)
    p = (p_dry / (R_d * T_K)) + (p_v / (R_v * T_K))
    
    # 3. Speed of Sound (c)
    # Calculate dry speed of sound first
    c_dry = math.sqrt(gamma * R_d * T_K)
    # Apply humidity correction
    c = c_dry * math.sqrt(1 + 0.3192 * (p_v / pressure_pa))
    
    # 4. Dynamic Viscosity (n) 
    # (Using Sutherland's law for dry air, as humidity impact is < 0.5%)
    n = mu_0 * ((T_K / T_0) ** 1.5) * ((T_0 + S) / (T_K + S))
    
    return AcousticConstants(
        air_density=p,
        dynamic_viscosity=n,
        speed_of_sound=c
    )