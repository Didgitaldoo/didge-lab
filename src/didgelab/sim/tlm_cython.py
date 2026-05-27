"""
Cython-backed transmission-line model for didgeridoo acoustics.

Uses the compiled _cadsd extension (from _cadsd.pyx) for faster impedance
spectrum computation. Build with: ``pip install -e .`` from the package root
(or ``python setup.py build_ext --inplace`` in ``sim/tlm_cython_lib/``).

If the extension is not built, this module still loads (e.g. for pdoc); using
``TransmissionLineModelCython`` will raise at runtime.
"""

try:
    from .tlm_cython_lib import _cadsd as _cadsd_mod
    from .tlm_cython_lib._cadsd import create_segments_from_geo, cadsd_Ze
    _CADSD_AVAILABLE = True
except ImportError:
    _cadsd_mod = None  # type: ignore
    create_segments_from_geo = None  # type: ignore
    cadsd_Ze = None  # type: ignore
    _CADSD_AVAILABLE = False

# cadsd_Ze_array is the bulk-evaluation entry point added in the perf refactor.
# If it isn't present (very old .so, or the MagicMock used by test_TairuaLoss
# without that attribute set), fall back to a Python loop over cadsd_Ze.
_cadsd_Ze_array = getattr(_cadsd_mod, "cadsd_Ze_array", None)

from .sim_interface import AcousticSimulationInterface, AcousticConstants
from ..geo import Geo
import numpy as np
from unittest.mock import MagicMock

_DEFAULT_CONSTANTS = AcousticConstants()


def _is_magicmock(obj):
    return isinstance(obj, MagicMock)


class TransmissionLineModelCython(AcousticSimulationInterface):
    """TLM simulator using Cython-compiled CADSD core.

    Note: the Cython backend stores physical constants as module-level state in
    ``_cadsd``. ``get_impedance_spectrum`` updates them before each call, so
    concurrent use of multiple ``TransmissionLineModelCython`` instances with
    different constants in the same process will race.
    """

    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array):
        """Return list of impedance magnitudes at each frequency in Hz."""
        if not _CADSD_AVAILABLE:
            raise ImportError(
                "didgelab.sim.tlm_cython_lib._cadsd is not built. "
                "Install the package with 'pip install -e .' from the package root to compile the Cython extension."
            )
        _cadsd_mod.set_constants(
            self.constants.air_density,
            self.constants.dynamic_viscosity,
            self.constants.speed_of_sound,
        )
        segments = create_segments_from_geo(geo.geo)
        # Re-resolve the bulk function every call so tests that swap in a
        # MagicMock for the _cadsd module are honoured.
        bulk = getattr(_cadsd_mod, "cadsd_Ze_array", None)
        if bulk is not None and not _is_magicmock(bulk):
            return bulk(segments, np.ascontiguousarray(frequencies, dtype=np.float64))
        # Legacy / mocked fallback: per-frequency Python loop.
        return np.array([cadsd_Ze(segments, f) for f in frequencies])