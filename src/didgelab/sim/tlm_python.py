"""
Pure-Python transmission-line model (TLM) for didgeridoo acoustics (CADSD-style).

Implements conical/cylindrical segment chain and impedance at the mouthpiece.
Based on transmission line modeling (Mapes-Riordan) and Frank Geipel's CADSD.
Geometry is in m; constants: air density p, viscosity n, speed of sound c.
"""

import math
import cmath
import numpy as np
import pandas as pd
from ..geo import Geo

from .sim_interface import AcousticSimulationInterface

# Physical constants (float64 preferred but caused issues on some MacOS)
p = np.float64(1.2929)   # air density kg/m³
n = np.float64(1.708e-5) # dynamic viscosity
c = np.float64(343.37)   # speed of sound m/s
PI = np.float64(np.pi)


class Segment:
    """Single conical/cylindrical segment: length L, diameters d0 (input), d1 (output)."""

    def __init__(self, L, d0, d1):
        self.L = L
        self.d0 = d0
        self.d1 = d1

        self.a0 = PI * d0 * d0 / 4
        self.a01 = PI * (d0 + d1) * (d0 + d1) / 16
        self.a1 = PI * d1 * d1 / 4
        self.phi = math.atan ((d1 - d0) / (2 * L))

        x = (2 * math.sin (self.phi))
        if x==0:
            x=1e-20
        self.l = (d1 - d0) / x
        self.x1 = d1 / x
        self.x0 = self.x1 - self.l
        self.r0 = p * c / self.a0

    @classmethod
    def create_segments_from_geo(cls, geo):
        """Build list of Segment from geometry (list of [x_mm, d_mm]); converts mm to m."""

        segments=[]
        shape=[[np.float64(x)[0]/1000, np.float64(x)[1]/1000] for x in geo]
        for i_seg in range(1, len(shape)):
            seg1=shape[i_seg]
            seg0=shape[i_seg-1]
            L=seg1[0]-seg0[0]
            d0=seg0[1]
            d1=seg1[1]
            seg=Segment(L, d0, d1)
            segments.append(seg)
        return segments

def create_segments_from_geo(geo):
    """Convenience wrapper for Segment.create_segments_from_geo."""
    return Segment.create_segments_from_geo(geo)


def ap(w, segments):
    """Chain transfer matrices for angular frequency w; returns 2x2 product matrix."""
    x = [[1, 0], [0, 1]]
    y=[[0,0], [0,0]]
    z=[[0,0],[0,0]]

    for t_seg in segments:

        L=t_seg.L
        d0=t_seg.d0
        d1=t_seg.d1

        a0 = t_seg.a0
        a01 = t_seg.a01
        a1 = t_seg.a1
        l = t_seg.l
        x0 = t_seg.x0
        x1 = t_seg.x1
        r0 = t_seg.r0

        rvw = math.sqrt (p * w * a01 / (n * PI))
        kw = w / c
        Tw =np.complex128(kw * 1.045 / rvw + (kw * (1.0 + 1.045 / rvw))*1j)
        Zcw = np.complex128(r0 * (1.0 + 0.369 / rvw) - 1j*r0 * 0.369 / rvw)

        ccoshlwl = cmath.cosh(Tw * l)
        csinhlwl = cmath.sinh(Tw * l)
        ccoshlwL = cmath.cosh(Tw * L)
        csinhlwL = cmath.sinh(Tw * L)

        if (d0 != d1):
            y[0][0] = x1 / x0 * (ccoshlwl - csinhlwl / (Tw * x1))
            y[0][1] = x0 / x1 * Zcw * csinhlwl
            y[1][0] = ((x1 / x0 - 1.0 / (Tw * Tw * x0 * x0)) * csinhlwl + Tw * l / ((Tw * x0) * (Tw * x0)) * ccoshlwl) / Zcw
            y[1][1] = x0 / x1 * (ccoshlwl + csinhlwl / (Tw * x0))
        else:
            y[0][0] = ccoshlwL
            y[0][1] = Zcw * csinhlwL
            y[1][0] = csinhlwL / Zcw
            y[1][1] = ccoshlwL

        # dot product
        z[0][0] = x[0][0] * y[0][0] + x[0][1] * y[1][0]
        z[0][1] = x[0][0] * y[0][1] + x[0][1] * y[1][1]
        z[1][0] = x[1][0] * y[0][0] + x[1][1] * y[1][0]
        z[1][1] = x[1][0] * y[0][1] + x[1][1] * y[1][1]

        x[0][0] = z[0][0]
        x[0][1] = z[0][1]
        x[1][0] = z[1][0]
        x[1][1] = z[1][1]
    return z


def Za(w, segments):
    """Radiation impedance at the bell (last segment) for angular frequency w."""
    t_seg = segments[-1]

    L = t_seg.L
    d1 = t_seg.d1
    a01 = t_seg.a01
    r0 = t_seg.r0

    rvw = math.sqrt (p * w * a01 / (n * PI))
    Zcw = np.complex128(r0*(1.0 + 0.369 / rvw) - 1j*r0 * 0.369 / rvw)

    res = 0.5 * Zcw * np.complex128(w * w * d1 * d1 / c / c + 1j*0.6 * L * w * d1 / c)  # from Geipel
    return res


def cadsd_Ze(segments, f):
    """Input impedance at mouthpiece (magnitude) for frequency f Hz."""
    w = 2.0 * PI * f
    a = Za(w, segments)
    b = ap(w, segments)
    Ze = abs((a * b[0][0] + b[0][1]) / (a * b[1][0] + b[1][1]))
    return Ze


def geo_fft(geo, gmax, offset):
    """Legacy: build impedance dict and peak/valley arrays; returns fft dict."""
    fft = {
        "impedance": {},
        "overblow": {},
        "ground": {}
    }

    for key in fft.keys(): 
        fft[key][0]=0

    segments=Segment.create_segments_from_geo(geo)
    for f in range(1, gmax):
        fft["impedance"][f] = cadsd_Ze(segments, f)
        fft["overblow"][f] = 0
        fft["ground"][f] = 0


    # search for peaks and valleys
    peaks=[0,0]
    vally=[0,0]

    up=False
    npeaks=0
    nvally=0

    freqs=fft["impedance"].keys()
    for i in range(2, len(freqs)):
        if fft["impedance"][i] > fft["impedance"][i-1]:
            if npeaks and not up:
                vally[nvally]=i-1
                nvally+=1
            up=True
        else:
            if up:
                peaks[npeaks] = i-1
                npeaks+=1
            up=False 

        if nvally>1:
            break

    if peaks[0]<0:
        return None

    k = 0.0001

    mem0 = peaks[0]
    mem0a = peaks[0]

    mem0b = mem0a
    return fft


class TransmissionLineModelPython(AcousticSimulationInterface):
    """TLM simulator implemented in pure Python (no Cython)."""

    def get_impedance_spectrum(self, geo: Geo, frequencies: np.array):
        """Return list of impedance magnitudes at each frequency in Hz."""
        segments = Segment.create_segments_from_geo(geo.geo)
        impedances = np.array([cadsd_Ze(segments, f) for f in frequencies])
        return impedances