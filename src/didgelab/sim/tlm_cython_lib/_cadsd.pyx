#!python
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# Build with Cython >= 3.1 for NumPy 2.x compatibility (PyArray_Descr subarray API).
#
# Fast transmission-line-model (CADSD) impedance solver.
#
# Design notes
# ------------
# The earlier version of this file looked Cython but ran near-Python-speed
# because (a) segments lived in a Python list of dicts (one hash-table lookup
# per field per segment, per frequency), (b) all inner functions were ``def``
# so every call crossed the Python-C boundary, and (c) the 2x2 transfer
# matrices were numpy arrays whose elements went through ``__getitem__`` /
# ``__setitem__`` per access.
#
# This rewrite removes all three:
#   - ``Segment`` is a packed ``cdef`` struct that is layout-compatible with a
#     numpy structured dtype ``SEG_DTYPE``. ``create_segments_from_geo`` builds
#     one of those arrays directly. Field access from Cython becomes a single
#     pointer dereference.
#   - The whole per-frequency math lives in a ``cdef nogil`` function with
#     stack-allocated ``double complex`` matrix entries.
#   - A new bulk entry point ``cadsd_Ze_array(segments, frequencies)`` evaluates
#     a whole frequency sweep in one Python -> C call; this is what
#     ``TransmissionLineModelCython`` now uses.
#
# Precision: switched from ``long double`` to ``double``. Long double is
# x86-only 80-bit; on ARM / macOS it's plain 64-bit anyway. The numerical
# answer differs in the last digit at most, which is far below the 5-cent
# cross-backend agreement test.

import math
import cmath
cimport numpy as np
cimport cython
import numpy as npy

from libc.math cimport sqrt, atan, sin, M_PI, fabs

np.import_array()


# ---------------------------------------------------------------------------
# C99 complex math (no GIL needed)
# ---------------------------------------------------------------------------

cdef extern from "complex.h" nogil:
    double complex ccosh(double complex z)
    double complex csinh(double complex z)
    double cabs(double complex z)


# ---------------------------------------------------------------------------
# Module-level physical constants
# ---------------------------------------------------------------------------

cdef double p = 1.2929
cdef double n = 1.708e-5
cdef double c = 343.37
cdef double PI = 3.14159265358979323846
# Minimum length/diameter (m) to avoid zero division; geometry is mm then converted to m
cdef double EPS_GEO = 1e-12

DEFAULT_P = 1.2929
DEFAULT_N = 1.708e-5
DEFAULT_C = 343.37


def set_constants(p_, n_, c_):
    """Override the module-level physical constants used by all simulations.

    Note: this mutates module-level state. Concurrent simulations in the same
    process that need different constants will race. Single-threaded callers
    should call this before invoking ``cadsd_Ze_array`` /
    ``create_segments_from_geo``.
    """
    global p, n, c
    p = p_
    n = n_
    c = c_


def reset_constants():
    """Restore physical constants to their default values."""
    set_constants(DEFAULT_P, DEFAULT_N, DEFAULT_C)


# ---------------------------------------------------------------------------
# Segment storage
# ---------------------------------------------------------------------------

cdef packed struct Segment:
    double L
    double d0
    double d1
    double a0
    double a01
    double a1
    double phi
    double l
    double x1
    double x0
    double r0


# Numpy dtype with byte-for-byte the same layout as ``cdef packed struct Segment``.
# Field order, sizes, and absence of padding all match (``align=False``).
SEG_DTYPE = npy.dtype([
    ("L", npy.float64),
    ("d0", npy.float64),
    ("d1", npy.float64),
    ("a0", npy.float64),
    ("a01", npy.float64),
    ("a1", npy.float64),
    ("phi", npy.float64),
    ("l", npy.float64),
    ("x1", npy.float64),
    ("x0", npy.float64),
    ("r0", npy.float64),
], align=False)


def create_segments_from_geo(geo):
    """Build a typed structured array of Segment from geometry.

    ``geo`` is a sequence of ``[x_mm, diameter_mm]`` points. The returned
    numpy array has dtype :data:`SEG_DTYPE`; pass it directly to
    :func:`cadsd_Ze_array`.
    """
    cdef np.ndarray[double, ndim=2] g = npy.asarray(geo, dtype=npy.float64)
    cdef Py_ssize_t n_pts = g.shape[0]
    cdef Py_ssize_t n_seg = n_pts - 1
    if n_seg < 1:
        raise ValueError("Geometry must have at least 2 points")

    out = npy.zeros(n_seg, dtype=SEG_DTYPE)
    cdef Segment[:] segs = out

    cdef Py_ssize_t i
    cdef double L, d0, d1, phi
    for i in range(n_seg):
        L = (g[i + 1, 0] - g[i, 0]) / 1000.0
        d0 = g[i, 1] / 1000.0
        d1 = g[i + 1, 1] / 1000.0

        if L <= 0.0:
            L = EPS_GEO
        if d0 <= 0.0:
            d0 = EPS_GEO
        if d1 <= 0.0:
            d1 = EPS_GEO

        segs[i].L = L
        segs[i].d0 = d0
        segs[i].d1 = d1
        segs[i].a0 = PI * d0 * d0 / 4.0
        segs[i].a01 = PI * (d0 + d1) * (d0 + d1) / 16.0
        segs[i].a1 = PI * d1 * d1 / 4.0
        phi = atan((d1 - d0) / (2.0 * L))
        segs[i].phi = phi

        if d1 == d0:
            # Cylindrical: l and x1 unused (the cylindrical branch in the
            # inner loop reads only L/d0/d1/a01/r0). NaN flags "do not use".
            segs[i].l = npy.nan
            segs[i].x1 = npy.nan
            segs[i].x0 = npy.nan
        else:
            segs[i].l = (d1 - d0) / (2.0 * sin(phi))
            segs[i].x1 = d1 / (2.0 * sin(phi))
            segs[i].x0 = segs[i].x1 - segs[i].l

        segs[i].r0 = p * c / segs[i].a0

    return out


# ---------------------------------------------------------------------------
# Inner kernel: impedance magnitude at one angular frequency.
#
# All arithmetic is in C; no Python objects, no GIL.
# ---------------------------------------------------------------------------

cdef inline double _cadsd_Ze_one(Segment[:] segs, double f) nogil:
    cdef double w = 2.0 * PI * f
    cdef Py_ssize_t n_seg = segs.shape[0]
    cdef Py_ssize_t i

    cdef double L, d0, d1, a01, l, x0, x1, r0
    cdef double rvw, kw

    cdef double complex Tw, Zcw
    cdef double complex ccoshlwl, csinhlwl, ccoshlwL, csinhlwL
    cdef double complex y00, y01, y10, y11
    cdef double complex z00, z01, z10, z11

    # x starts as the identity matrix and accumulates the product
    cdef double complex x00 = 1.0 + 0.0j
    cdef double complex x01 = 0.0 + 0.0j
    cdef double complex x10 = 0.0 + 0.0j
    cdef double complex x11 = 1.0 + 0.0j

    for i in range(n_seg):
        L = segs[i].L
        d0 = segs[i].d0
        d1 = segs[i].d1
        a01 = segs[i].a01
        l = segs[i].l
        x0 = segs[i].x0
        x1 = segs[i].x1
        r0 = segs[i].r0

        rvw = sqrt(p * w * a01 / (n * PI))
        kw = w / c

        # Tw = kw * 1.045 / rvw + i * (kw * (1 + 1.045/rvw))
        Tw = (kw * 1.045 / rvw) + 1j * (kw * (1.0 + 1.045 / rvw))
        # Zcw = r0 * (1 + 0.369/rvw) - i * r0 * 0.369/rvw
        Zcw = (r0 * (1.0 + 0.369 / rvw)) - 1j * (r0 * 0.369 / rvw)

        ccoshlwl = ccosh(Tw * l)
        csinhlwl = csinh(Tw * l)
        ccoshlwL = ccosh(Tw * L)
        csinhlwL = csinh(Tw * L)

        if d0 != d1:
            y00 = x1 / x0 * (ccoshlwl - csinhlwl / (Tw * x1))
            y01 = x0 / x1 * Zcw * csinhlwl
            y10 = ((x1 / x0 - 1.0 / (Tw * Tw * x0 * x0)) * csinhlwl
                   + Tw * l / ((Tw * x0) * (Tw * x0)) * ccoshlwl) / Zcw
            y11 = x0 / x1 * (ccoshlwl + csinhlwl / (Tw * x0))
        else:
            y00 = ccoshlwL
            y01 = Zcw * csinhlwL
            y10 = csinhlwL / Zcw
            y11 = ccoshlwL

        # x = x . y
        z00 = x00 * y00 + x01 * y10
        z01 = x00 * y01 + x01 * y11
        z10 = x10 * y00 + x11 * y10
        z11 = x10 * y01 + x11 * y11
        x00 = z00
        x01 = z01
        x10 = z10
        x11 = z11

    # Radiation impedance at bell (Za), computed from the last segment
    cdef double L_last = segs[n_seg - 1].L
    cdef double d1_last = segs[n_seg - 1].d1
    cdef double a01_last = segs[n_seg - 1].a01
    cdef double r0_last = segs[n_seg - 1].r0

    cdef double rvw_a = sqrt(p * w * a01_last / (n * PI))
    cdef double complex Zcw_a = (r0_last * (1.0 + 0.369 / rvw_a)) - 1j * (r0_last * 0.369 / rvw_a)
    cdef double complex a = 0.5 * Zcw_a * (
        w * w * d1_last * d1_last / c / c
        + 1j * 0.6 * L_last * w * d1_last / c
    )

    cdef double complex num = a * x00 + x01
    cdef double complex den = a * x10 + x11
    return cabs(num / den)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def cadsd_Ze_array(segments, frequencies):
    """Compute impedance magnitudes for a whole frequency sweep.

    This is the *fast path*. One Python -> C transition per call, regardless of
    how many frequencies are in the sweep.

    Args:
        segments: structured array returned by :func:`create_segments_from_geo`.
        frequencies: 1-D array-like of frequencies in Hz.

    Returns:
        1-D ``np.float64`` array of impedance magnitudes, one per frequency.
    """
    cdef Segment[:] segs = segments
    cdef double[::1] freqs = npy.ascontiguousarray(frequencies, dtype=npy.float64)
    cdef Py_ssize_t n_freq = freqs.shape[0]

    out = npy.empty(n_freq, dtype=npy.float64)
    cdef double[::1] o = out

    cdef Py_ssize_t i
    with nogil:
        for i in range(n_freq):
            o[i] = _cadsd_Ze_one(segs, freqs[i])

    return out


def cadsd_Ze(segments, f):
    """Single-frequency impedance magnitude (legacy API).

    Kept for back-compat with callers that loop over frequencies in Python.
    Prefer :func:`cadsd_Ze_array` for sweeps — it avoids one Python -> C
    boundary crossing per frequency.
    """
    cdef Segment[:] segs = segments
    cdef double freq = float(f)
    cdef double result
    with nogil:
        result = _cadsd_Ze_one(segs, freq)
    return result
