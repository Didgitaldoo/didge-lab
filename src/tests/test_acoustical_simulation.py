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

    def test_fem2d_returns_impedance_array(self):
        pytest.importorskip("skfem")
        geo = Geo([[0, 32], [1200, 60]])
        freqs = np.array([73.0, 150.0, 300.0])
        imp = acoustical_simulation(geo, freqs, simulation_method="2d_fem")
        assert len(imp) == len(freqs)
        assert all(isinstance(z, (int, float, np.floating)) for z in imp)
        assert all(z > 0 for z in imp)

    def test_all_backends_agree_on_fundamental(self):
        """All four backends should locate the first impedance peak at the
        same frequency, within 5 cents.

        Uses the geometry and log-spaced frequency grid from
        ``doc/examples/acoustical_simulations.ipynb`` so this test guards
        against regressions on a realistic didgeridoo shape. Each backend has
        a different normalization, so absolute magnitudes are not compared —
        only the *frequency* of the first impedance peak, which is the
        fundamental and a physical quantity that must agree.

        5 cents (~0.29%) is a musically meaningful tolerance: anything looser
        and two backends would be heard as tuned differently.

        The 2D FEM backend solves the axisymmetric Helmholtz equation (r-
        weighted forms over the meridional half-plane), which is equivalent
        to the 1D Webster horn and matches the TLM transfer-matrix result on
        the fundamental.
        """
        pytest.importorskip("skfem")
        try:
            from didgelab.sim.tlm_cython_lib._cadsd import cadsd_Ze  # noqa: F401
        except ImportError:
            pytest.skip("tlm_cython extension not built")

        # Some sibling test modules (e.g. test_TairuaLoss) replace
        # sys.modules["didgelab.sim.tlm_cython_lib._cadsd"] with a MagicMock
        # whose ``cadsd_Ze`` returns a constant. If that's happened, skip — we
        # cannot compare against a flat spectrum.
        import sys as _sys
        from unittest.mock import MagicMock as _MagicMock
        _cmod = _sys.modules.get("didgelab.sim.tlm_cython_lib._cadsd")
        if isinstance(_cmod, _MagicMock):
            pytest.skip("tlm_cython is mocked in this session")

        from scipy.signal import argrelextrema

        # Geometry copied verbatim from doc/examples/acoustical_simulations.ipynb
        geo = Geo([
            [0.0, 30.0], [51.46899363230762, 33.7784524142553],
            [85.30423566381049, 27.778931402719003],
            [146.18854416328128, 29.93556480319699],
            [179.09381040534882, 28.176128566813905],
            [259.6802069488814, 28.958055614249236],
            [285.8190435948692, 32.523925651740726],
            [371.25522284885295, 28.657844125210218],
            [419.84135591763703, 29.58434155089998],
            [466.51643935886017, 29.918981345641743],
            [527.3904014120826, 31.90101706990902],
            [531.1184504606866, 34.99886886158793],
            [602.3168197366772, 32.507540421468754],
            [662.7828119015719, 36.93351761826125],
            [696.8927839128953, 35.045862969399295],
            [746.6898985029628, 36.84736191316634],
            [815.4460816003736, 39.05244109202236],
            [832.2110323436855, 43.81877757673142],
            [848.9759830869973, 48.23024903217608],
            [865.7409338303091, 51.74295627153961],
            [882.505884573621, 54.20771815807003],
            [899.2708353169328, 55.56194698443228],
            [916.0357860602446, 55.32924009903286],
            [932.8007368035564, 53.87356959783409],
            [949.5656875468683, 57.597070298366724],
            [966.3306382901801, 60.18361651506194],
            [999.0345487680597, 64.33061402881069],
            [1015.6645962439811, 65.73163097400256],
            [1032.2946437199025, 65.17490857100057],
            [1048.924691195824, 62.75626540369139],
            [1065.5547386717453, 58.79571998981954],
            [1082.1847861476667, 54.17039891202623],
            [1119.7483374925102, 50.13599574849312],
            [1145.100343149991, 52.07082961850034],
            [1196.9667987935184, 52.785312404146595],
            [1275.3325429113545, 51.631734726631564],
            [1310.4749184562197, 49.258375148257805],
            [1360.5050929455278, 55.86662095119183],
            [1432.9324204516163, 57.034846444202294],
            [1447.2181347373307, 57.6709182777631],
            [1461.503849023045, 58.180684544167576],
            [1475.7895633087592, 58.54248286994988],
            [1490.0752775944734, 58.97267620477503],
            [1504.3609918801878, 59.524557312640155],
            [1518.646706165902, 60.26754363477322],
            [1532.9324204516163, 62.288373855176175],
            [1547.2181347373307, 64.72655249320907],
            [1561.503849023045, 67.72346991377412],
            [1575.7895633087592, 70.39265102058911],
            [1590.0752775944734, 73.44970066456183],
            [1604.3609918801878, 77.47427770386149],
            [1618.646706165902, 82.64877489218449],
        ])
        frequencies = get_log_simulation_frequencies(30, 1000, 5)

        def _first_peak(imp):
            # First local maximum; argrelextrema returns increasing-then-
            # decreasing indices, so [0] is the lowest-frequency peak.
            imp = np.asarray(imp, dtype=float)
            ex = argrelextrema(imp, np.greater)[0]
            assert ex.size, "no peaks found"
            i = int(ex[0])
            # Parabolic interpolation in log-frequency for sub-bin precision.
            # The frequency grid step (5 cents) is the test tolerance, so the
            # raw bin index alone has ~5-cent uncertainty. cents = 1200*log2(f)
            # so log(f) is the natural axis for the interpolation.
            if 0 < i < len(imp) - 1:
                y0, y1, y2 = imp[i - 1], imp[i], imp[i + 1]
                denom = y0 - 2.0 * y1 + y2
                if denom != 0.0:
                    offset = 0.5 * (y0 - y2) / denom  # in samples (-0.5..0.5)
                    log_f = np.log(frequencies)
                    return float(np.exp(
                        log_f[i] + offset * (log_f[i + 1] - log_f[i])
                    ))
            return float(frequencies[i])

        # Method list = notebook's list ("tlm_python", "tlm_cython", "1d_fem")
        # plus "2d_fem" which now uses an axisymmetric solver and agrees with
        # the others. Pass explicit AcousticConstants() so every backend uses
        # literally the same numbers — without this, each backend falls through
        # the base ABC's default of compute_moist_air_properties(), and even
        # though that is deterministic, tiny differences in the per-backend
        # damping models multiplied by the (different) moist-air c shift the
        # result enough to cross the 5-cent boundary.
        constants = AcousticConstants()
        peak_freqs = {}
        for method in ("tlm_python", "tlm_cython", "1d_fem", "2d_fem"):
            imp = acoustical_simulation(
                geo, frequencies, simulation_method=method, constants=constants,
            )
            peak_freqs[method] = _first_peak(imp)

        # Compare every pair; assert max separation < 5 cents.
        # cents = 1200 * log2(f1 / f2)
        vals = list(peak_freqs.values())
        max_cents = 1200.0 * np.log2(max(vals) / min(vals))
        assert max_cents < 5.0, (
            f"First-peak frequencies disagree by {max_cents:.2f} cents "
            f"(> 5 cent tolerance). Peaks: "
            + ", ".join(f"{m}={f:.3f} Hz" for m, f in peak_freqs.items())
        )

    def test_temperature_humidity_changes_all_methods(self):
        """For every simulation backend, swapping cold/dry (20 C, 0% RH) air
        for warm/wet (30 C, 100% RH) breath must shift the fundamental.

        Warm moist air has a higher speed of sound (~6% higher than 20 C dry),
        which raises every resonance. This guards against any backend that
        silently ignores the ``constants`` argument — a regression we'd
        otherwise only catch by listening.

        Source: temperature_and_humidity.ipynb.
        """
        pytest.importorskip("skfem")
        try:
            from didgelab.sim.tlm_cython_lib._cadsd import cadsd_Ze  # noqa: F401
        except ImportError:
            pytest.skip("tlm_cython extension not built")

        import sys as _sys
        from unittest.mock import MagicMock as _MagicMock
        _cmod = _sys.modules.get("didgelab.sim.tlm_cython_lib._cadsd")
        if isinstance(_cmod, _MagicMock):
            pytest.skip("tlm_cython is mocked in this session")

        from scipy.signal import argrelmax
        from didgelab.sim.sim_interface import compute_moist_air_properties

        geo = Geo([[0, 32], [1500, 70]])
        frequencies = get_log_simulation_frequencies(30, 1000, 5)

        cold_dry = compute_moist_air_properties(temp_celsius=20.0, rel_humidity=0.0)
        warm_wet = compute_moist_air_properties(temp_celsius=30.0, rel_humidity=1.0)

        # Speed of sound must differ between the two conditions — otherwise
        # this test isn't really testing anything.
        assert cold_dry.speed_of_sound != warm_wet.speed_of_sound

        methods = ("tlm_python", "tlm_cython", "1d_fem", "2d_fem")
        for method in methods:
            imp_cold = acoustical_simulation(
                geo, frequencies, simulation_method=method, constants=cold_dry,
            )
            imp_warm = acoustical_simulation(
                geo, frequencies, simulation_method=method, constants=warm_wet,
            )
            f_cold = float(frequencies[argrelmax(imp_cold)[0][0]])
            f_warm = float(frequencies[argrelmax(imp_warm)[0][0]])
            assert f_cold != f_warm, (
                f"Backend '{method}' produced the same fundamental "
                f"({f_cold:.3f} Hz) for cold/dry and warm/wet air — "
                "the constants argument is being ignored."
            )
            # Sanity: warm air (higher c) should give a higher fundamental
            assert f_warm > f_cold, (
                f"Backend '{method}' fundamental moved the wrong way: "
                f"cold/dry={f_cold:.3f} Hz, warm/wet={f_warm:.3f} Hz "
                "(expected warm > cold since c grows with temperature)."
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
