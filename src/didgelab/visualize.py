"""
Visualization of didgeridoo bore geometry, impedance spectra, and tuning (notes).

Provides:
- Bore geometry plot from a Geo object
- Impedance spectrum computation and display for a Geo
- Tuning display using get_notes
- Combined one-row-three-column layout (bore | impedance | notes)
- Support for multiple geometries (multiple rows, three columns each).
"""

import numpy as np
import matplotlib.pyplot as plt

from didgelab.geo import Geo

try:
    from didgelab.sim.tlm_cython_lib.sim import (
        get_log_simulation_frequencies,
        create_segments,
        compute_impedance,
        get_notes,
    )
    _SIM_AVAILABLE = True
except ImportError:
    _SIM_AVAILABLE = False


def _ensure_geo(geo):
    """Return Geo instance; accept Geo or list of segments."""
    if isinstance(geo, Geo):
        return geo
    return Geo(geo=geo)


def plot_bore(geo, ax=None, half_bore=True, **kwargs):
    """
    Plot bore geometry (cross-section): position along bore (mm) vs diameter (mm).

    Args:
        geo: Didgeridoo geometry (Geo instance or list of [x, d] segments).
        ax: Matplotlib axes. If None, current axes or new figure is used.
        half_bore: If True, plot as half-profile (y = ±d/2). If False, plot diameter vs x.
        **kwargs: Passed to plot/fill_between (e.g. color, label).

    Returns:
        matplotlib axes used.
    """
    geo = _ensure_geo(geo)
    if ax is None:
        ax = plt.gca()
    x = np.array([s[0] for s in geo.geo])
    d = np.array([s[1] for s in geo.geo], dtype=float)
    if half_bore:
        r = d / 2
        ax.fill_between(x, -r, r, **kwargs)
        ax.vlines(x, -r, r, colors="lightgray", linewidth=0.5)
    else:
        ax.plot(x, d, **kwargs)
        ax.vlines(x, 0, d, colors="lightgray", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_title("Bore geometry")
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    plt.axis('equal')
    return ax


def plot_impedance_spectrum(
    geo,
    ax=None,
    fmin=1,
    fmax=1000,
    max_error=5,
    **kwargs
):
    """
    Compute and plot impedance spectrum for a geometry.

    Args:
        geo: Didgeridoo geometry (Geo or list of segments).
        ax: Matplotlib axes. If None, current axes or new figure is used.
        fmin, fmax: Frequency range (Hz).
        max_error: Log-spacing error (cents) for simulation frequencies.
        **kwargs: Passed to ax.plot (e.g. color, label).

    Returns:
        (ax, freqs, impedance) – axes and computed data.
    """
    if not _SIM_AVAILABLE:
        raise RuntimeError("Simulation backend not available; cannot compute impedance.")
    geo = _ensure_geo(geo)
    if ax is None:
        ax = plt.gca()
    freqs = get_log_simulation_frequencies(fmin, fmax, max_error)
    segments = create_segments(geo)
    impedance = compute_impedance(segments, freqs)
    ax.plot(freqs, impedance, **kwargs)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Impedance")
    ax.set_title("Impedance spectrum")
    ax.grid(True, alpha=0.3)
    return ax, freqs, impedance


def plot_notes(
    geo,
    ax=None,
    base_freq=440,
    fmin=1,
    fmax=1000,
    max_error=5,
    **kwargs
):
    """
    Compute impedance, get notes, and display tuning in the given axes.

    Args:
        geo: Didgeridoo geometry (Geo or list of segments).
        ax: Matplotlib axes. If None, current axes or new figure is used.
        base_freq: A4 tuning (Hz) for note names.
        fmin, fmax, max_error: Passed to simulation.
        **kwargs: Unused (for API consistency).

    Returns:
        (ax, notes_df) – axes and notes DataFrame.
    """
    if not _SIM_AVAILABLE:
        raise RuntimeError("Simulation backend not available; cannot compute notes.")
    geo = _ensure_geo(geo)
    if ax is None:
        ax = plt.gca()
    freqs = get_log_simulation_frequencies(fmin, fmax, max_error)
    segments = create_segments(geo)
    impedance = compute_impedance(segments, freqs)
    notes = get_notes(freqs, impedance, base_freq=base_freq)
    ax.axis("off")
    ax.set_title("Tuning (notes)")
    if notes.empty:
        ax.text(0.5, 0.5, "No peaks", ha="center", va="center", transform=ax.transAxes)
        return ax, notes
    # Compact table: note_name, freq, cent_diff
    cell_text = []
    for _, row in notes.iterrows():
        cent = row["cent_diff"]
        sign = "+" if cent >= 0 else ""
        cell_text.append([row["note_name"], f"{row['freq']:.1f}", f"{sign}{cent:.1f}"])
    table = ax.table(
        cellText=cell_text,
        colLabels=["Note", "Freq (Hz)", "Cents"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    return ax, notes


def plot_geo_impedance_notes(
    geos,
    fmin=1,
    fmax=1000,
    max_error=5,
    base_freq=440,
    figsize=None,
    titles=None,
):
    """
    Plot one row per geometry with three columns: bore | impedance | notes.

    Args:
        geos: Single Geo (or list of segments) or list of Geo/list-of-segments.
        fmin, fmax, max_error: Simulation frequency range and spacing.
        base_freq: For note names.
        figsize: (width, height) per row; default (12, 4) for one row.
        titles: Optional list of row titles (same length as geos).

    Returns:
        (fig, axes_2d) – figure and array of axes [n_rows, 3].
    """
    if not _SIM_AVAILABLE:
        raise RuntimeError("Simulation backend not available.")
    # Normalize to list of Geo: single Geo or list-of-segments -> [geo]
    if isinstance(geos, Geo):
        geos = [geos]
    elif isinstance(geos, list) and len(geos) > 0:
        first = geos[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            geos = [geos]  # one geometry as list of [x, d] segments
    geos = [_ensure_geo(g) for g in geos]
    n = len(geos)
    if figsize is None:
        figsize = (12, 4 * n)
    fig, axes = plt.subplots(n, 3, figsize=figsize)
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, geo in enumerate(geos):
        row_title = (titles[i] if titles and i < len(titles) else None) or f"Geometry {i + 1}"
        plot_bore(geo, ax=axes[i, 0])
        axes[i, 0].set_title(f"{row_title}\nBore")
        plot_impedance_spectrum(geo, ax=axes[i, 1], fmin=fmin, fmax=fmax, max_error=max_error)
        plot_notes(geo, ax=axes[i, 2], base_freq=base_freq, fmin=fmin, fmax=fmax, max_error=max_error)
    plt.tight_layout()
    return fig, axes


def vis_didge(geo, ax=None):
    """
    Plot didgeridoo bore geometry (half cross-section). Used by analysis and notebooks.

    Args:
        geo: Didgeridoo geometry (Geo instance or list of [x, d] segments).
        ax: Matplotlib axes. If None, creates a new figure and axes.

    Returns:
        matplotlib axes used.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    return plot_bore(geo, ax=ax, half_bore=True)
