import numpy as np

from didgelab.synth.bore import Bore, chain_matrix

GEOS = [
    [[0, 32], [1500, 72]],
    [[0, 30], [72.6, 24], [173, 26.7], [257.7, 36], [353, 24], [483, 30.3], [545, 36.2],
     [628, 36.5], [760.6, 25.7], [853, 26.6], [945.8, 37.3], [998, 29.3], [1119, 49.8],
     [1183.5, 55.5], [1296, 60.4], [1388.7, 77]],
    [[0, 32], [800, 32], [900, 38], [970, 42], [1050, 40], [1180, 48], [1350, 60], [1500, 72]],
]


def test_impedance_matches_didgelab():
    for geo in GEOS:
        b = Bore(geo, fmax=3000)
        f = np.arange(30, 3000, 3.7)
        z_dl = b.didgelab_impedance(f)
        A, B, C, D, Zr = chain_matrix(b.geo, f, b.constants)
        z = np.abs((A * Zr + B) / (C * Zr + D))
        assert np.max(np.abs(z - z_dl) / z_dl) < 1e-9


def test_resonances_are_impedance_peaks():
    b = Bore(GEOS[0], fmax=2000)
    r = b.resonances
    assert 60 < r[0].freq < 80
    assert all(r[i].freq < r[i + 1].freq for i in range(len(r) - 1))
    # a cone's toots are roughly (but not exactly) harmonic
    assert 2.0 < r[1].freq / r[0].freq < 3.2


def test_transfer_function_peaks_at_resonances():
    b = Bore(GEOS[0], fmax=2000)
    for res in b.resonances[:3]:
        f = np.linspace(res.freq * 0.9, res.freq * 1.1, 400)
        h = np.abs(b.interp("H", f))
        assert abs(f[np.argmax(h)] / res.freq - 1) < 0.02


def test_loss_factor_reduces_q():
    q1 = Bore(GEOS[0], fmax=1000).resonances[0].q
    z1 = Bore(GEOS[0], fmax=1000).interp("Zin", [Bore(GEOS[0], fmax=1000).drone])
    b2 = Bore(GEOS[0], fmax=1000, loss_factor=3.0)
    z2 = b2.interp("Zin", [b2.drone])
    assert abs(z2[0]) < abs(z1[0])
