"""
Abstract interface for acoustic simulation backends in DidgeLab.

Any simulator (e.g. TLM Python or Cython) implements get_impedance_spectrum(geo, frequencies).
"""

from abc import ABC, abstractmethod
import numpy as np

from ..geo import Geo


class AcousticSimulationInterface(ABC):
    """Interface for computing acoustic impedance spectrum of a didgeridoo geometry."""

    @abstractmethod
    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array) -> np.array:
        """Return impedance values at each frequency in Hz for the given geometry."""
        pass