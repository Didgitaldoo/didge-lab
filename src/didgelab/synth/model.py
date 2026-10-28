"""
The source-filter model of the played didgeridoo.

The lips act as a periodic flow source U_s with a finite internal (dynamic)
impedance Z_s. The vocal tract (upstream) and the bore (downstream) are in
series with the lips, so the acoustic flow through the lips is

    U(f) = U_s(f) * Z_s / (Z_s + Z_vt(f) + Z_in(f))

and the far-field pressure radiated from the bell is

    P(f) ~ j*omega * H(f) * U(f),     H = U_bell / U_mouth   (from the bore).

Peaks of Z_vt choke the flow at those frequencies and produce the
characteristic formant-like bands of the didgeridoo sound
(Tarnopolsky et al., Nature 2005).
"""

from dataclasses import dataclass, asdict, replace

import numpy as np

from . import vocaltract
from .excitation import source_harmonics


@dataclass(frozen=True)
class ModelParams:
    """Parameters of the player/recording model.

    Defaults are the values calibrated on the Didge Archive
    (see ``didgesynth.calibrate`` and the article).
    """
    zs_mpa: float = 0.1          # lip source impedance, MPa s/m^3
    loss_factor: float = 1.0     # wall-loss multiplier of the bore (1 = DidgeLab)
    source: str = "lips"         # 'lips' (beating lips) or 'hindle' (Didge2Wav)
    oq: float = 0.45             # open quotient of the lips
    skew: float = 0.85           # opening fraction of the open phase
    alpha: float = 0.85          # shape of the Hindle source
    tilt_db_oct: float = 1.0     # on-axis directivity of the bell / microphone, dB/octave (calibrated)
    pitch_cents: float = 0.0     # played drone relative to the impedance peak
    toot_cents: float = -45.0    # played toots relative to the impedance peak (archive: -40..-58)
    noise_db: float = -30.0      # breath noise relative to the tone

    def to_dict(self):
        return asdict(self)

    def with_(self, **kw):
        return replace(self, **kw)


def filter_gain(bore, freqs, vt, params: ModelParams, zvt=None):
    """Complex transfer from source flow U_s to radiated pressure at ``freqs``.

    ``vt`` may be None (no vocal tract), a VocalTract, a (4,) array, or a
    (K, 4) array together with ``freqs`` of shape (K, N). A precomputed
    vocal-tract impedance (same shape as ``freqs``) can be passed as ``zvt``.
    """
    freqs = np.asarray(freqs, dtype=float)
    zs = params.zs_mpa * 1e6
    zin = bore.interp("Zin", freqs)
    H = bore.interp("H", freqs)
    if zvt is None:
        zvt = vocaltract.impedance(vt, freqs) if vt is not None else 0.0
    G = 1j * freqs / 100.0 * H * zs / (zs + zin + zvt)
    if params.tilt_db_oct:
        G = G * 10 ** (params.tilt_db_oct * np.log2(np.maximum(freqs, 1) / 1000) / 20)
    return np.where(freqs < bore.freqs[-1], G, 0)


def source(params: ModelParams, n_harm, oq=None, skew=None):
    return source_harmonics(n_harm,
                            oq=params.oq if oq is None else oq,
                            skew=params.skew if skew is None else skew,
                            model=params.source, alpha=params.alpha)


def harmonic_spectrum(bore, f0, vt, params: ModelParams, n_harm=40):
    """Complex amplitudes of the first n_harm harmonics of the radiated sound."""
    n = np.arange(1, n_harm + 1)
    return source(params, n_harm) * filter_gain(bore, n * f0, vt, params)


def harmonic_db(bore, f0, vt, params, n_harm=40):
    return 20 * np.log10(np.abs(harmonic_spectrum(bore, f0, vt, params, n_harm)) + 1e-30)
