import numpy as np
import pytest

from didgelab.synth import Bore, ModalBore, Resonance, render_song_loop_from_peaks

GEO = [[0, 32], [1500, 72]]


def test_modal_bore_peaks_are_the_given_peaks():
    peaks = [(70.0, 2e7, 15.0), (190.0, 8e6, 25.0), (310.0, 5e6, 30.0)]
    b = ModalBore(peaks)
    assert [r.freq for r in b.resonances] == [70.0, 190.0, 310.0]
    assert b.drone == 70.0 and b.toot(1) == 190.0
    for f, z, q in peaks:
        ff = np.linspace(f * 0.95, f * 1.05, 2001)
        zz = b.didgelab_impedance(ff)
        assert abs(ff[np.argmax(zz)] / f - 1) < 0.005
        assert abs(zz.max() / z - 1) < 0.1
        # -3 dB bandwidth gives back Q
        above = ff[zz > zz.max() / np.sqrt(2)]
        assert abs(f / (above[-1] - above[0]) / q - 1) < 0.15


def test_modal_bore_accepts_dicts_and_resonances_and_sorts():
    b1 = ModalBore([{"freq": 200, "impedance": 5e6, "q": 20}, {"freq": 70, "impedance": 2e7, "q": 15}])
    b2 = ModalBore([Resonance(70, 2e7, 15), Resonance(200, 5e6, 20)])
    assert b1.drone == 70 and np.allclose(b1.Zin, b2.Zin)


@pytest.mark.parametrize("peaks", [[], [(70, -1, 10)], [(0, 1e7, 10)], [(70, 1e7, 0)]])
def test_modal_bore_rejects_invalid_peaks(peaks):
    with pytest.raises(ValueError):
        ModalBore(peaks)


def test_modal_bore_transfer_function_peaks_at_resonances():
    b = ModalBore(Bore(GEO, fmax=2000).resonances[:5], mouth_diameter=32)
    for res in b.resonances[:3]:
        f = np.linspace(res.freq * 0.9, res.freq * 1.1, 400)
        h = np.abs(b.interp("H", f))
        assert abs(f[np.argmax(h)] / res.freq - 1) < 0.02


def test_render_song_loop_from_peaks():
    peaks = [(r.freq, r.impedance, r.q) for r in Bore(GEO, fmax=3000).resonances]
    y, fs = render_song_loop_from_peaks(peaks, "tiki-du", tempo=120)
    assert fs == 48000 and len(y) > fs
    assert np.isfinite(y).all() and 0 < np.abs(y).max() <= 1.0
    with pytest.raises(ValueError):
        render_song_loop_from_peaks(peaks, "no-such-song")


def _drone_brightness(bore):
    from didgelab.synth import ModelParams
    from didgelab.synth.model import harmonic_spectrum
    from didgelab.synth.techniques import score_controls

    params = ModelParams()
    vt = np.median(np.asarray(score_controls(bore, "tempo = 60\ndoo:3", params, seed=0).vt), 0)
    a = np.abs(harmonic_spectrum(bore, bore.drone, vt, params, 100)) ** 2
    return (a * np.arange(1, 101)).sum() / a.sum()


def test_modal_bore_brightness_close_to_geometry():
    # The peaks alone do not fix the timbre: on the didge-archive and generated
    # bores a ModalBore drone is 0.43..1.6 x as bright as the geometry with the
    # same peaks (0.85 x on average).
    from tests.test_bore import GEOS

    for geo in GEOS:
        b = Bore(geo)
        m = ModalBore(b.resonances, mouth_diameter=geo[0][1])
        assert 0.4 < _drone_brightness(m) / _drone_brightness(b) < 1.7
