"""
Acoustics of the didgeridoo bore.

The input impedance Z_in(f) is computed with DidgeLab
(``didgelab.acoustical_simulation``).  For synthesis we additionally need the
*complex* quantities of the same transmission-line model, most importantly the
volume-flow transfer function from the mouthpiece to the bell

    H(f) = U_bell / U_mouth = 1 / (C * Z_rad + D),

where [[A, B], [C, D]] is the chained transfer matrix of all conical segments.
DidgeLab only returns |Z_in|, so this module re-implements DidgeLab's
transmission-line formulas (Mapes-Riordan cones, CADSD losses and radiation
impedance) vectorised over frequency.  ``tests/test_bore.py`` checks that
|Z_in| of this implementation equals DidgeLab's result.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.signal import find_peaks

from ..geo import Geo
from ..acoustical_simulation import acoustical_simulation
from ..sim.sim_interface import AcousticConstants, compute_moist_air_properties


def _as_geo(geo):
    if isinstance(geo, Geo):
        return geo
    return Geo([[float(x), float(d)] for x, d in geo])


def chain_matrix(geo, freqs, constants: AcousticConstants, loss_factor=1.0):
    """Transfer matrix of the bore for every frequency.

    Returns arrays A, B, C, D (complex, shape ``freqs.shape``) and the
    radiation impedance Z_rad at the bell.  Formulas are those of
    ``didgelab.sim.tlm_python`` (CADSD), vectorised.
    """
    rho = constants.air_density
    eta = constants.dynamic_viscosity
    c = constants.speed_of_sound

    w = 2 * np.pi * np.asarray(freqs, dtype=np.float64)
    shape = np.asarray(geo.geo, dtype=np.float64) / 1000.0  # mm -> m

    A = np.ones_like(w, dtype=np.complex128)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    D = np.ones_like(A)

    seg = None
    for (x0_, d0), (x1_, d1) in zip(shape[:-1], shape[1:]):
        L = x1_ - x0_
        if L <= 0:
            continue
        a0 = np.pi * d0 * d0 / 4
        a01 = np.pi * (d0 + d1) ** 2 / 16
        phi = np.arctan((d1 - d0) / (2 * L))
        s = 2 * np.sin(phi)
        if s == 0:
            s = 1e-20
        l = (d1 - d0) / s
        x1 = d1 / s
        x0 = x1 - l
        r0 = rho * c / a0

        rvw = np.sqrt(rho * w * a01 / (eta * np.pi))
        kw = w / c
        rv = rvw / loss_factor
        T = kw * 1.045 / rv + 1j * kw * (1.0 + 1.045 / rv)
        Zc = r0 * (1.0 + 0.369 / rv) - 1j * r0 * 0.369 / rv

        if d0 != d1:
            ch, sh = np.cosh(T * l), np.sinh(T * l)
            y00 = x1 / x0 * (ch - sh / (T * x1))
            y01 = x0 / x1 * Zc * sh
            y10 = ((x1 / x0 - 1.0 / (T * T * x0 * x0)) * sh + T * l / (T * x0) ** 2 * ch) / Zc
            y11 = x0 / x1 * (ch + sh / (T * x0))
        else:
            ch, sh = np.cosh(T * L), np.sinh(T * L)
            y00, y01, y10, y11 = ch, Zc * sh, sh / Zc, ch

        A, B, C, D = (A * y00 + B * y10, A * y01 + B * y11,
                      C * y00 + D * y10, C * y01 + D * y11)
        seg = (L, d1, Zc)

    L, d1, Zc = seg
    # radiation impedance at the bell, identical to didgelab's Za()
    Zrad = 0.5 * Zc * (w * w * d1 * d1 / c / c + 1j * 0.6 * L * w * d1 / c)
    return A, B, C, D, Zrad


@dataclass
class Resonance:
    freq: float
    impedance: float
    q: float


class Bore:
    """A didgeridoo bore with cached acoustic spectra.

    Parameters
    ----------
    geo : list of [x_mm, diameter_mm] or ``didgelab.Geo``
    fmax : highest frequency that is simulated (Hz)
    df : frequency resolution of the dense spectrum (Hz)
    loss_factor : wall-loss multiplier of the complex spectra (1 = DidgeLab)
    """

    def __init__(self, geo, fmax=12000.0, df=0.5, constants=None, loss_factor=1.0):
        self.geo = _as_geo(geo)
        self.constants = constants or compute_moist_air_properties()
        self.loss_factor = loss_factor
        self.freqs = np.arange(df, fmax + df, df)
        A, B, C, D, Zrad = chain_matrix(self.geo, self.freqs, self.constants, loss_factor)
        self.Zin = (A * Zrad + B) / (C * Zrad + D)
        self.H = 1.0 / (C * Zrad + D)
        # characteristic impedance at the mouthpiece, used for normalisation
        d0 = self.geo.geo[0][1] / 1000
        self.Z0 = self.constants.air_density * self.constants.speed_of_sound / (np.pi * d0 * d0 / 4)

    # -- DidgeLab ---------------------------------------------------------------
    def didgelab_impedance(self, freqs):
        """|Z_in| computed by DidgeLab's own (Cython) simulation."""
        return np.asarray(acoustical_simulation(self.geo, np.asarray(freqs, dtype=float),
                                                constants=self.constants))

    @cached_property
    def resonances(self):
        """Impedance maxima (drone, toots) computed with DidgeLab.

        A coarse DidgeLab sweep finds the peaks, each peak is refined with a
        0.05 Hz sweep, and the quality factor is estimated from the -3 dB width.
        """
        f = np.arange(20.0, 3000.0, 1.0)
        z = self.didgelab_impedance(f)
        idx, _ = find_peaks(z)
        out = []
        for i in idx:
            ff = np.arange(f[i] - 2.0, f[i] + 2.0, 0.05)
            zz = self.didgelab_impedance(ff)
            j = int(np.argmax(zz))
            fpk, zpk = ff[j], zz[j]
            # -3 dB bandwidth from the dense complex spectrum
            mag = np.abs(self.Zin)
            k = int(np.argmin(np.abs(self.freqs - fpk)))
            lo = k
            while lo > 0 and mag[lo] > zpk / np.sqrt(2):
                lo -= 1
            hi = k
            while hi < len(mag) - 1 and mag[hi] > zpk / np.sqrt(2):
                hi += 1
            bw = max(self.freqs[hi] - self.freqs[lo], 1e-3)
            out.append(Resonance(float(fpk), float(zpk), float(fpk / bw)))
        return out

    @property
    def drone(self):
        return self.resonances[0].freq

    def toot(self, n=1):
        """Frequency of the n-th toot (n=1 is the second impedance peak)."""
        return self.resonances[n].freq

    # -- interpolation helpers -------------------------------------------------
    def interp(self, name, f):
        arr = getattr(self, name)
        f = np.asarray(f, dtype=float)
        re = np.interp(f, self.freqs, arr.real, left=arr.real[0], right=0.0)
        im = np.interp(f, self.freqs, arr.imag, left=arr.imag[0], right=0.0)
        return re + 1j * im
