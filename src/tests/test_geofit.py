import numpy as np
import pytest

from didgelab.synth import Bore, fit_geometry, render_song_loop_from_peaks
from didgelab.synth.geofit import find_peaks

GEO = [[0, 32], [800, 34], [1100, 42], [1450, 75]]


def test_find_peaks_matches_bore_resonances():
    b = Bore(GEO, fmax=2000)
    pk = find_peaks(GEO, 1200, b.constants)
    for r in b.resonances[:5]:
        p = min(pk, key=lambda p: abs(p.freq - r.freq))
        assert abs(1200 * np.log2(p.freq / r.freq)) < 1.0
        assert abs(p.impedance / r.impedance - 1) < 0.02


def test_fit_geometry_reproduces_peaks_of_a_bore():
    b = Bore(GEO)
    fit = fit_geometry(b.resonances[:4], mouth_diameter=32)
    assert fit.max_cents < 3.0
    assert np.all(np.abs(np.log(fit.impedance_ratio)) < np.log(1.5))
    assert fit.geometry[0] == [0.0, 32.0]
    assert all(b2[0] > a[0] for a, b2 in zip(fit.geometry, fit.geometry[1:]))


def test_fit_geometry_flags_an_unrealistic_target():
    # a drone and a "toot" only 30 cents above it cannot both be resonances of a bore
    fit = fit_geometry([(70.0, 2e7, 20.0), (71.2, 2e7, 20.0)])
    assert fit.max_cents > 50


@pytest.mark.parametrize("peaks", [[], [(70, -1, 10)], [(0, 1e7, 10)]])
def test_fit_geometry_rejects_invalid_peaks(peaks):
    with pytest.raises(ValueError):
        fit_geometry(peaks)


def test_render_song_loop_from_peaks_with_geometry():
    peaks = [(r.freq, r.impedance, r.q) for r in Bore(GEO).resonances[:3]]
    y, fs = render_song_loop_from_peaks(peaks, "tiki-du", tempo=120, method="geometry")
    assert len(y) > fs and np.isfinite(y).all() and 0 < np.abs(y).max() <= 1.0
    with pytest.raises(ValueError):
        render_song_loop_from_peaks(peaks, "tiki-du", method="nope")
