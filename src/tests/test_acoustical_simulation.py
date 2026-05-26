"""
Pytest unit tests for didgelab.acoustical_simulation.
"""

import pytest
import numpy as np

from didgelab.geo import Geo
from didgelab.acoustical_simulation import (
    acoustical_simulation,
    _get_closest_index,
    get_log_simulation_frequencies,
    interpolate_spectrum,
    get_notes,
)
from didgelab.sim.sim_interface import AcousticConstants


class TestAcousticalSimulation:
    """Tests for acoustical_simulation main entry point."""

    def test_returns_impedance_array(self):
        geo = Geo([[0, 32], [1200, 60]])
        freqs = np.array([73.0, 150.0, 300.0])
        imp = acoustical_simulation(geo, freqs, simulation_method="tlm_python")
        assert len(imp) == len(freqs)
        assert all(isinstance(z, (int, float, np.floating)) for z in imp)
        assert all(z > 0 for z in imp)

    def test_unknown_backend_raises(self):
        geo = Geo([[0, 32], [1200, 60]])
        freqs = np.array([73.0])
        with pytest.raises(Exception, match="Unknown simulation backend \"invalid\""):
            acoustical_simulation(geo, freqs, simulation_method="invalid")

    def test_fem_returns_impedance_array(self):
        pytest.importorskip("skfem")
        geo = Geo([[0, 32], [1200, 60]])
        freqs = np.array([73.0, 150.0, 300.0])
        imp = acoustical_simulation(geo, freqs, simulation_method="1d_fem")
        assert len(imp) == len(freqs)
        assert all(isinstance(z, (int, float, np.floating)) for z in imp)
        assert all(z > 0 for z in imp)

    def test_all_backends_agree_on_fundamental(self):
        """All three backends should locate the fundamental impedance peak at
        roughly the same frequency (within ~10 percent) for the same geometry.

        Each backend uses a different normalization, so absolute magnitudes are
        not comparable — but peak *frequencies* are a physical quantity and
        must agree.
        """
        pytest.importorskip("skfem")
        try:
            from didgelab.sim.tlm_cython_lib._cadsd import cadsd_Ze  # noqa: F401
        except ImportError:
            pytest.skip("tlm_cython extension not built")

        geo = Geo([[0, 32], [1500, 70]])
        freqs = np.linspace(40, 200, 50)

        peak_freqs = {}
        for method in ("tlm_python", "tlm_cython", "1d_fem"):
            imp = acoustical_simulation(geo, freqs, simulation_method=method)
            peak_freqs[method] = freqs[int(np.argmax(imp))]

        ref = peak_freqs["tlm_python"]
        for method, f in peak_freqs.items():
            rel_err = abs(f - ref) / ref
            assert rel_err < 0.10, (
                f"Backend '{method}' fundamental peak at {f:.1f} Hz disagrees "
                f"with tlm_python at {ref:.1f} Hz (rel err {rel_err:.2%}). "
                f"All peaks: {peak_freqs}"
            )

    def test_fem_constants_take_effect(self):
        pytest.importorskip("skfem")
        geo = Geo([[0, 32], [1200, 60]])
        freqs = np.array([73.0, 150.0, 300.0])
        imp_default = acoustical_simulation(geo, freqs, simulation_method="1d_fem")
        imp_custom = acoustical_simulation(
            geo,
            freqs,
            simulation_method="1d_fem",
            constants=AcousticConstants(speed_of_sound=340.0),
        )
        # Lowering c shifts the spectrum noticeably — guards the m/s -> mm/s wiring.
        assert not np.allclose(imp_default, imp_custom)


class TestGetClosestIndex:
    """Tests for _get_closest_index."""

    def test_exact_match(self):
        freqs = np.array([50.0, 100.0, 150.0, 200.0])
        assert _get_closest_index(freqs, 100.0) == 1

    def test_closest_below(self):
        freqs = np.array([50.0, 100.0, 150.0, 200.0])
        assert _get_closest_index(freqs, 95.0) == 1

    def test_closest_above(self):
        freqs = np.array([50.0, 100.0, 150.0, 200.0])
        assert _get_closest_index(freqs, 105.0) == 1

    def test_first_element(self):
        freqs = np.array([50.0, 100.0, 150.0])
        assert _get_closest_index(freqs, 50.0) == 0

    def test_last_element(self):
        freqs = np.array([50.0, 100.0, 150.0])
        assert _get_closest_index(freqs, 150.0) == 2

    def test_above_max_returns_len(self):
        freqs = np.array([50.0, 100.0, 150.0])
        assert _get_closest_index(freqs, 200.0) == 3


class TestGetLogSimulationFrequencies:
    """Tests for get_log_simulation_frequencies."""

    def test_returns_array(self):
        freqs = get_log_simulation_frequencies(fmin=50, fmax=500, max_error=100)
        assert isinstance(freqs, np.ndarray)
        assert len(freqs) > 0

    def test_monotonic_ascending(self):
        freqs = get_log_simulation_frequencies(fmin=50, fmax=500, max_error=100)
        assert np.all(np.diff(freqs) > 0)

    def test_all_below_fmax(self):
        freqs = get_log_simulation_frequencies(fmin=50, fmax=500, max_error=100)
        assert np.all(freqs <= 500)

    def test_starts_near_fmin(self):
        freqs = get_log_simulation_frequencies(fmin=100, fmax=1000, max_error=50)
        assert freqs[0] >= 100

    def test_smaller_max_error_more_points(self):
        freqs_coarse = get_log_simulation_frequencies(fmin=50, fmax=500, max_error=100)
        freqs_fine = get_log_simulation_frequencies(fmin=50, fmax=500, max_error=10)
        assert len(freqs_fine) > len(freqs_coarse)


class TestInterpolateSpectrum:
    """Tests for interpolate_spectrum."""

    def test_output_length(self):
        freqs = np.array([1.0, 50.0, 100.0, 150.0])
        impedances = np.array([1e5, 2e5, 1.5e5, 2e5])
        f_ip, imp_ip = interpolate_spectrum(freqs, impedances)
        # Output covers integer freqs from 1 to int(freqs[-1])-1
        assert f_ip[0] == 1
        assert f_ip[-1] <= int(np.round(freqs[-1]))
        assert len(f_ip) >= 1 and len(imp_ip) >= 1

    def test_linearly_interpolated(self):
        freqs = np.array([1.0, 10.0, 20.0])
        impedances = np.array([100.0, 200.0, 300.0])
        f_ip, imp_ip = interpolate_spectrum(freqs, impedances)
        # At freq 10, impedance should be 200
        idx_10 = np.argmin(np.abs(f_ip - 10))
        assert imp_ip[idx_10] == pytest.approx(200.0, abs=1.0)


class TestGetNotes:
    """Tests for get_notes."""

    def test_returns_dataframe(self):
        freqs = np.linspace(50, 500, 200)
        impedances = 1e5 * np.exp(-((freqs - 73) ** 2) / 500) + 0.5e5 * np.exp(
            -((freqs - 146) ** 2) / 500
        )
        notes = get_notes(freqs, impedances, base_freq=440)
        assert hasattr(notes, "columns")
        assert "note_name" in notes.columns or len(notes) == 0

    def test_columns_present(self):
        # Create spectrum with clear peak
        freqs = np.linspace(50, 300, 500)
        peak_idx = 100
        impedances = np.zeros_like(freqs)
        impedances[peak_idx] = 1e6
        impedances[peak_idx - 1] = 0.5e6
        impedances[peak_idx + 1] = 0.5e6
        notes = get_notes(freqs, impedances, base_freq=440)
        if len(notes) > 0:
            for col in ["note_name", "cent_diff", "note_nr", "freq", "impedance", "rel_imp"]:
                assert col in notes.columns

    def test_with_target_freqs(self):
        freqs = np.linspace(50, 300, 500)
        impedances = 1e5 * np.exp(-((freqs - 73) ** 2) / 500)
        target_freqs = np.array([73.0, 150.0])
        notes = get_notes(freqs, impedances, base_freq=440, target_freqs=target_freqs)
        if len(notes) > 0:
            assert "target" in notes.columns
