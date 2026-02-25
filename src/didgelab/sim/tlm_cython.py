"""
Cython-backed transmission-line model for didgeridoo acoustics.

Uses the compiled _cadsd extension (from _cadsd.pyx) for faster impedance
spectrum computation. Build with: ``pip install -e .`` from the package root
(or ``python setup.py build_ext --inplace`` in ``sim/tlm_cython_lib/``).

If the extension is not built, this module still loads (e.g. for pdoc); using
``TransmissionLineModelCython`` will raise at runtime.
"""

try:
    from .tlm_cython_lib._cadsd import create_segments_from_geo, cadsd_Ze
    _CADSD_AVAILABLE = True
except ImportError:
    create_segments_from_geo = None  # type: ignore
    cadsd_Ze = None  # type: ignore
    _CADSD_AVAILABLE = False

from .sim_interface import AcousticSimulationInterface
from ..geo import Geo
import numpy as np


class TransmissionLineModelCython(AcousticSimulationInterface):
    """TLM simulator using Cython-compiled CADSD core."""

    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array):
        """Return list of impedance magnitudes at each frequency in Hz."""
        if not _CADSD_AVAILABLE:
            raise ImportError(
                "didgelab.sim.tlm_cython_lib._cadsd is not built. "
                "Install the package with 'pip install -e .' from the package root to compile the Cython extension."
            )
        segments = create_segments_from_geo(geo.geo)
        impedances = np.array([cadsd_Ze(segments, f) for f in frequencies])
        return impedances