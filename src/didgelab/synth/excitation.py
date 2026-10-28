"""
Lip excitation (and voice source).

Two lip-flow models are provided.

``hindle``  -- the "dijj" excitation of Hindle & Posnett used in Didge2Wav:

    u(t) = p0/R - (p0^2/R^3) / (a0 + a sin(2 pi f t))^2

  With alpha = a/a0 this is, up to scale and offset, -1/(1 + alpha sin)^2.
  The waveform is analytic, so its harmonics decay *exponentially*: the
  sound gets very dull above ~1 kHz.

``lips`` (default) -- beating lips. The lips are closed for a fraction
  (1 - OQ) of the cycle. During the open phase the flow follows a
  Rosenberg-type pulse whose opening part takes a fraction ``skew`` of the
  open phase:

      u(tau) = (1 - cos(pi tau / T_p)) / 2          0 <= tau < T_p
      u(tau) = cos(pi/2 (tau - T_p) / T_n)          T_p <= tau < T_p + T_n

  The slope discontinuity at closure gives the power-law decay (~n^-2) of
  the harmonics that is found in the Didge Archive recordings.

All coefficient sets are normalised to unit AC power, so the shape
parameters only change the timbre; loudness is controlled separately.
"""

from functools import lru_cache

import numpy as np

N_PERIOD = 4096
N_TABLE_HARM = 256


def _normalised_harmonics(u, n_harm=N_TABLE_HARM):
    U = np.fft.rfft(u)[1:n_harm + 1] * 2 / len(u)
    return U / np.sqrt(np.sum(np.abs(U) ** 2) / 2 + 1e-30)


# --------------------------------------------------------------------------
# Hindle & Posnett
def hindle_waveform(alpha, n=N_PERIOD):
    th = 2 * np.pi * np.arange(n) / n
    u = -1.0 / (1.0 + alpha * np.sin(th)) ** 2
    return u - u.mean()


@lru_cache(maxsize=None)
def _hindle_table(n_alpha=64):
    alphas = np.linspace(0.0, 0.97, n_alpha)
    return alphas, np.stack([_normalised_harmonics(hindle_waveform(a)) for a in alphas])


def hindle_harmonics(alpha, n_harm):
    alphas, table = _hindle_table()
    return _interp1(table, alphas, alpha)[..., :n_harm]


# --------------------------------------------------------------------------
# beating lips
def lips_waveform(oq, skew=0.6, n=N_PERIOD):
    """One period of beating-lip flow; open for a fraction ``oq`` of the period."""
    tau = np.arange(n) / n
    tp = oq * skew
    tn = oq * (1 - skew)
    u = np.zeros(n)
    m1 = tau < tp
    u[m1] = 0.5 * (1 - np.cos(np.pi * tau[m1] / tp))
    m2 = (tau >= tp) & (tau < tp + tn)
    u[m2] = np.cos(0.5 * np.pi * (tau[m2] - tp) / tn)
    return u - u.mean()


OQS = np.linspace(0.2, 0.95, 31)
SKEWS = np.linspace(0.3, 0.9, 13)


@lru_cache(maxsize=None)
def _lips_table():
    return np.stack([[_normalised_harmonics(lips_waveform(o, s)) for s in SKEWS] for o in OQS])


def lips_harmonics(oq, skew, n_harm):
    """Complex harmonic coefficients, shape broadcast(oq, skew) + (n_harm,)."""
    table = _lips_table()[:, :, :n_harm]
    oq, skew = np.broadcast_arrays(np.asarray(oq, float), np.asarray(skew, float))
    po = np.clip((oq - OQS[0]) / (OQS[1] - OQS[0]), 0, len(OQS) - 1.0001)
    ps = np.clip((skew - SKEWS[0]) / (SKEWS[1] - SKEWS[0]), 0, len(SKEWS) - 1.0001)
    io, is_ = po.astype(int), ps.astype(int)
    ao, as_ = (po - io)[..., None], (ps - is_)[..., None]
    return ((1 - ao) * (1 - as_) * table[io, is_] + ao * (1 - as_) * table[io + 1, is_]
            + (1 - ao) * as_ * table[io, is_ + 1] + ao * as_ * table[io + 1, is_ + 1])


def _interp1(table, grid, x):
    x = np.clip(np.asarray(x, dtype=float), grid[0], grid[-1])
    pos = (x - grid[0]) / (grid[1] - grid[0])
    i = np.clip(pos.astype(int), 0, len(grid) - 2)
    a = (pos - i)[..., None]
    return table[i] * (1 - a) + table[i + 1] * a


def source_harmonics(n_harm, oq=0.5, skew=0.6, model="lips", alpha=0.85):
    if model == "hindle":
        return hindle_harmonics(alpha, n_harm)
    return lips_harmonics(oq, skew, n_harm)


# --------------------------------------------------------------------------
# voice
def voice_harmonics(n_harm, tilt_db_per_oct=-12.0):
    """Glottal source spectrum of a sung/hummed tone (simple spectral tilt)."""
    k = np.arange(1, n_harm + 1)
    return 10 ** (tilt_db_per_oct * np.log2(k) / 20) * (-1j)
