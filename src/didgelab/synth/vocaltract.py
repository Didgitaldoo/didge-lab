"""
Vocal tract model.

The player's vocal tract is the acoustic load *upstream* of the lips. It is
modelled as a chain of short lossy cylinders (lips -> glottis) whose areas
follow a parametric area function, terminated by the glottis and the
subglottal airway. The input impedance Z_vt(f) seen from the lips enters the
synthesis in series with the bore impedance (see ``synth.py``).

Articulatory parameters (all in [0, 1]):

* ``tongue_pos``    0 = tongue tip / front (alveolar), 1 = back (velar/pharyngeal)
* ``tongue_height`` 0 = tongue low / open, 1 = full closure (stop consonant)
* ``mouth``         jaw opening / size of the oral cavity, 0 = nearly closed, 1 = wide
* ``glottis``       0 = closed (voiced / held), 1 = wide open (breathing)
"""

from dataclasses import dataclass, replace, fields

import numpy as np

RHO = 1.1556
C = 350.2

VT_LENGTH = 0.165      # m, lips to glottis
N_SECTIONS = 24
TRACHEA_AREA = 2.5e-4  # m^2
TRACHEA_LENGTH = 0.12  # m
GLOTTIS_MIN = 5e-6     # m^2, nearly closed glottis
GLOTTIS_MAX = 1.5e-4   # m^2, wide open
GLOTTIS_LENGTH = 0.01  # m, effective length incl. end corrections
WALL_B0 = 30.0         # Hz, frequency-independent formant bandwidth (soft walls)
WALL_B1 = 30.0         # Hz at 1 kHz, sqrt(f) viscous/thermal part
AREA_SCALE = 0.2       # global scale of the area function (calibrated, see article)
MOUTH_RANGE = 2.5      # octaves of overall area change over the 'mouth' range


@dataclass(frozen=True)
class VocalTract:
    tongue_pos: float = 0.5
    tongue_height: float = 0.3
    mouth: float = 0.5
    glottis: float = 0.7

    def lerp(self, other, a):
        return VocalTract(*[(1 - a) * getattr(self, f.name) + a * getattr(other, f.name)
                            for f in fields(self)])

    def with_(self, **kw):
        return replace(self, **kw)

    def as_array(self):
        return np.array([getattr(self, f.name) for f in fields(self)])


# Vowel-like presets ("mouth shapes" used by didgeridoo players).
VOWELS = {
    # "neutral" is the relaxed, wide-open mouth (open glottis) that best
    # explains the neutral drones of the Didge Archive. The articulated
    # mouth shapes have a nearly closed jaw and lie on the tongue-position
    # axis; their output formant (prominence 12-18 dB) falls monotonically
    # from front to back (values for the Matema bore).
    "neutral": VocalTract(0.30, 0.20, 1.00, 0.6),
    "i":   VocalTract(0.40, 0.50, 0.45, 0.3),   # "ee": tongue front   (~2500 Hz)
    "e":   VocalTract(0.50, 0.50, 0.45, 0.3),   #                       (~2100 Hz)
    "ae":  VocalTract(0.60, 0.50, 0.60, 0.3),   # open "wa"            (~1800 Hz)
    "a":   VocalTract(0.70, 0.50, 0.55, 0.3),   #                       (~1550 Hz)
    "o":   VocalTract(0.80, 0.50, 0.45, 0.3),   #                       (~1350 Hz)
    "u":   VocalTract(0.90, 0.50, 0.45, 0.3),   # "oo": tongue back    (~1180 Hz)
}

# Place of articulation for consonants: (tongue_pos of the closure, closure type)
CONSONANTS = {
    "t": dict(tongue_pos=0.05, kind="stop", voiced=False),
    "d": dict(tongue_pos=0.05, kind="stop", voiced=True),
    "k": dict(tongue_pos=0.62, kind="stop", voiced=False),
    "g": dict(tongue_pos=0.62, kind="stop", voiced=True),
    "p": dict(tongue_pos=0.0, kind="stop", voiced=False),  # cheek/lip pulse, not a closure of the tract
    "r": dict(tongue_pos=0.08, kind="tap", voiced=True),     # single tongue flap ("ri-ti")
    "rr": dict(tongue_pos=0.08, kind="trill", voiced=True),  # rolled r
    "ng": dict(tongue_pos=0.75, kind="coda", voiced=True),   # syllable end "dong"
    "n": dict(tongue_pos=0.05, kind="coda", voiced=True),
    "y": dict(tongue_pos=0.15, kind="glide", voiced=True),
    "w": dict(tongue_pos=0.65, kind="glide", voiced=True),
    "h": dict(tongue_pos=0.5, kind="aspirate", voiced=False),
    "ch": dict(tongue_pos=0.12, kind="fricative", voiced=False),
    "s": dict(tongue_pos=0.04, kind="fricative", voiced=False),
}


def _params(vt):
    """VocalTract or array (..., 4) -> array (..., 4)."""
    if isinstance(vt, VocalTract):
        return vt.as_array()
    return np.asarray(vt, dtype=float)


def area_function(vt, n=N_SECTIONS):
    """Cross-sectional areas (m^2) of ``n`` sections from lips (index 0) to glottis.

    ``vt`` is a VocalTract or an array of shape (..., 4); the result has shape (..., n).
    """
    P = _params(vt)
    tp, th, mo = P[..., 0:1], np.clip(P[..., 1:2], 0, 1), P[..., 2:3]
    x = (np.arange(n) + 0.5) / n * VT_LENGTH
    # neutral tract, piecewise linear through (x, area) knots; the area of the
    # front cavity (knots 0 and 1) scales with 'mouth' (cheeks, jaw)
    front = (2.0 + 4.5 * mo) * 1e-4
    knots_x = [0.0, 0.02, 0.07, 0.12, VT_LENGTH]
    hat = [np.interp(x, knots_x, np.eye(5)[k]) for k in range(5)]
    base = (hat[0] * (1.2e-4 + 0.6 * front) + hat[1] * front
            + hat[2] * 3.2e-4 + hat[3] * 2.6e-4 + hat[4] * 1.8e-4)
    # tongue constriction: Gaussian bump, x_c from 1.5 cm (tip) to 13 cm (pharynx)
    xc = 0.015 + tp * 0.115
    width = 0.018 + 0.012 * tp
    a_min = 1e-7 + 0.5e-4 * (1 - th) ** 1.5
    # Gaussian with a flattened core: the central sections close completely
    bump = np.minimum(1.0, 1.3 * np.exp(-0.5 * ((x - xc) / width) ** 2))
    A = base - (base - np.minimum(base, a_min)) * bump * np.minimum(1.0, 1.6 * th + 0.2)
    # jaw opening scales the whole oral tract: a relaxed open mouth is wide,
    # an articulating didgeridoo player keeps the jaw nearly closed
    return np.maximum(A * AREA_SCALE * 2.0 ** (MOUTH_RANGE * (mo - 0.5)), 1e-7)


def _mul(m, ch, sh, Zc):
    a, b, c, d = m
    return (a * ch + b * sh / Zc, a * Zc * sh + b * ch,
            c * ch + d * sh / Zc, c * Zc * sh + d * ch)


def impedance(vt, freqs, n=N_SECTIONS):
    """Input impedance of the vocal tract seen from the lips, Pa s / m^3.

    Shapes: ``vt`` (4,) or VocalTract with ``freqs`` (N,) -> (N,);
    ``vt`` (K, 4) with ``freqs`` (K, N) -> (K, N).
    """
    P = _params(vt)
    freqs = np.asarray(freqs, dtype=float)
    w = 2 * np.pi * freqs
    f = np.maximum(freqs, 1.0)
    A_s = area_function(P, n)                       # (..., n)
    dl = VT_LENGTH / n
    # wall vibration + viscous/thermal loss -> formant bandwidths of 50-150 Hz
    alpha = np.pi * (WALL_B0 + WALL_B1 * np.sqrt(f / 1000.0)) / C
    gamma = alpha + 1j * w / C
    ch, sh = np.cosh(gamma * dl), np.sinh(gamma * dl)

    m = (np.ones_like(gamma), np.zeros_like(gamma), np.zeros_like(gamma), np.ones_like(gamma))
    for i in range(n):
        Zc = (RHO * C / A_s[..., i])[..., None] if P.ndim > 1 else RHO * C / A_s[..., i]
        m = _mul(m, ch, sh, Zc)

    # glottis (short orifice) followed by the trachea, terminated by the lungs
    g = P[..., 3]
    Ag = GLOTTIS_MIN + np.clip(g, 0, 1) ** 2 * GLOTTIS_MAX
    Zg = (RHO * C / Ag)[..., None] if P.ndim > 1 else RHO * C / Ag
    m = _mul(m, np.cosh(gamma * GLOTTIS_LENGTH), np.sinh(gamma * GLOTTIS_LENGTH), Zg)
    cht, sht = np.cosh(gamma * TRACHEA_LENGTH / 6), np.sinh(gamma * TRACHEA_LENGTH / 6)
    for _ in range(6):
        m = _mul(m, cht, sht, RHO * C / TRACHEA_AREA)
    a, b, c, d = m
    Zt = RHO * C / TRACHEA_AREA
    return (a * Zt + b) / (c * Zt + d)
