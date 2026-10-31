import numpy as np

from didgelab.synth import synthesize
from didgelab.synth.bore import Bore, ModalBore
from didgelab.synth import impedance as imp

GEO = [[0, 32], [800, 32], [900, 38], [970, 42], [1050, 40], [1180, 48], [1350, 60], [1390, 68], [1500, 72]]


def test_matrix_pencil_recovers_damped_modes():
    fs = 4000
    t = np.arange(int(0.12 * fs)) / fs
    y = np.exp(-30 * t) * np.cos(2 * np.pi * 75 * t) + 0.5 * np.exp(-60 * t) * np.cos(2 * np.pi * 210 * t + 1)
    m = imp.matrix_pencil(y, fs, order=4)
    f = np.sort(m[:, 0])
    assert np.allclose(f, [75, 210], atol=0.5)
    a = m[np.argsort(m[:, 0]), 1]
    assert np.allclose(a, [30, 60], rtol=0.05)


def test_geometry_parametrisation():
    theta = [1500, 35, 40, 50, 60, 70, 80]
    g = imp.geometry(theta)
    assert g[0] == [0.0, imp.MOUTHPIECE_MM]
    assert abs(g[-1][0] - 1500) < 1e-9 and abs(g[-1][1] - 80) < 1e-9
    assert len(g) == imp.N_SEG + 1


def test_true_bore_fits_its_own_drone():
    y, fs = synthesize(GEO, "tempo = 60\ndoo:3", seed=0, reverb=0.0)
    obs = imp.observe(y, fs)
    b = Bore(GEO, fmax=3000)
    assert abs(1200 * np.log2(obs.f0 / b.drone)) < 20
    fw = imp._Forward(obs, imp.ModelParams())
    terms, pk = imp.geo_terms(GEO, fw, imp.Weights())
    other, _ = imp.geo_terms([[0, 31], [1500, 120]], fw, imp.Weights())
    assert terms["harm"] < other["harm"]
    assert abs(pk[0] / b.drone - 1) < 0.005


def test_estimate_finds_the_drone_resonance():
    y, fs = synthesize(GEO, "tempo = 60\ndoo:3", seed=0, reverb=0.0)
    est = imp.estimate(y, fs, restarts=1, maxiter=8)
    b = Bore(GEO, fmax=3000)
    assert abs(1200 * np.log2(est.resonances[0][0] / b.drone)) < 30
    assert est.band_db.shape == (2, len(est.freqs))


def test_estimate_peaks_feed_a_modal_bore():
    y, fs = synthesize(GEO, "tempo = 60\ndoo:3", seed=0, reverb=0.0)
    est = imp.estimate(y, fs, restarts=1, maxiter=8)
    peaks = est.peaks
    assert peaks[0].freq == est.resonances[0][0]
    assert abs(peaks[0].impedance - est.resonances[0][1] * 1e6) < 1
    mb = ModalBore(peaks[:6])
    assert abs(mb.drone / est.resonances[0][0] - 1) < 1e-9
