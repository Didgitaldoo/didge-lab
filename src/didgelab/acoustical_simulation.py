"""
Acoustical simulation entry point for DidgeLab.

This module provides a single function to run the transmission-line-based
acoustical simulation for a didgeridoo geometry over a set of frequencies.
It is used for computing impedance spectra and resonant frequencies (e.g. drone
and toot peaks). The simulation model follows transmission-line theory and
CADSD-style implementations (see `didgelab.sim.tlm_python` and `didgelab.sim.tlm_cython`).
"""

from didgelab import sim
from .sim.sim_interface import AcousticSimulationInterface
from .geo import Geo
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from .conv import *

def acoustical_simulation(
    geo: Geo,
    frequencies: np.ndarray,
    simulation_method: str = "tlm_cython",
):
    """
    Compute acoustic impedance at the given frequencies for a didgeridoo geometry.

    Uses a transmission-line model of the bore. Impedance peaks correspond to
    resonances (drone, toots). The result has the same length as `frequencies`.

    Args:
        geo: Didgeridoo geometry (bore profile as list of segments). Instance of
            `didgelab.geo.Geo` with `geo.geo` as list of `[x_mm, diameter_mm]`.
        frequencies: 1D array of frequencies in Hz at which to evaluate impedance.
        simulation_method: Which backend to use. `"tlm_python"` uses the pure-Python
            implementation; `"tlm_cython"` uses the compiled Cython extension (faster,
            requires a successful build of `didgelab.sim.tlm_cython_lib._cadsd`).

    Returns:
        Impedance magnitude at each frequency, in the same order as `frequencies`.
        Type matches the backend (list or array); length equals `len(frequencies)`.

    Raises:
        Exception: If `simulation_method` is not `"tlm_python"` or `"tlm_cython"`.

    Example:
        >>> from didgelab.acoustical_simulation import acoustical_simulation
        >>> from didgelab.geo import Geo
        >>> import numpy as np
        >>> geo = Geo([[0, 32], [1200, 60]])  # 1.2 m cone, 32 mm mouth, 60 mm bell
        >>> freqs = np.array([73, 150, 300])
        >>> imp = acoustical_simulation(geo, freqs, simulation_method="tlm_python")
        >>> len(imp) == 3
        True
    """
    if simulation_method == "tlm_python":
        from .sim.tlm_python import TransmissionLineModelPython
        simulator = TransmissionLineModelPython()
        return simulator.get_impedance_spectrum(geo, frequencies)
    elif simulation_method == "tlm_cython":
        from .sim.tlm_cython import TransmissionLineModelCython
        simulator = TransmissionLineModelCython()
        return simulator.get_impedance_spectrum(geo, frequencies)
    elif simulation_method == "1d_fem":
        from .sim.fem import FiniteElementsModeling1D
        simulator = FiniteElementsModeling1D()
        return simulator.get_impedance_spectrum(geo, frequencies)
    else:
        raise Exception(f"Unknown simulation backend \"{simulation_method}\"")


# helper function for compute_ground
def _get_closest_index(freqs, f):
    """
    Return the index in `freqs` whose value is closest to `f`.

    Assumes `freqs` is sorted ascending. If `f` is above the maximum frequency,
    returns the last index (or len(freqs) when f > freqs[-1]).

    Args:
        freqs: 1D array of frequencies in Hz (ascending).
        f: Target frequency in Hz.

    Returns:
        int: Index such that freqs[index] is closest to f.
    """
    for i in range(len(freqs)):
        m2=np.abs(freqs[i]-f)
        if i==0:
            m1=m2
            continue
        if m2>m1:
            return i-1
        m1=m2

    if f>freqs[-1]:
        return len(freqs)
    else:
        return len(freqs)-1

# helper function for compute_ground
# find first maximum in a list of numbers
def _find_first_maximum_index(impedance):

    peaks=[0,0]
    vally=[0,0]

    up = 0
    npeaks = 0
    nvally = 0

    for i in range(1, len(impedance)):
        if impedance[i] > impedance[i-1]:
            if npeaks and not up:
                vally[nvally] = i - 1
                nvally+=1
            up = 1
        else:
            if up:
                peaks[npeaks] = i - 1
                npeaks+=1
            up = 0
        if nvally > 1:
            break

    if peaks[0]<0:
        raise Exception("bad fft")

    return peaks[0]

        
# compute ground spektrum from impedance spektrum
# warning: frequencies must be evenly spaced
def compute_ground_spektrum(freqs, impedance):
    """
    Compute a "ground" spectrum (envelope / baseline) from an impedance spectrum.

    Used for visualization or analysis of the resonance envelope. The algorithm
    builds a baseline from the fundamental and its harmonics, then returns
    values in dB (20*log10 of a scaled product of impedance and ground).

    Args:
        freqs: 1D array of frequencies in Hz. Must be evenly spaced.
        impedance: 1D array of impedance magnitude at each frequency (same length as freqs).

    Returns:
        np.ndarray: Ground spectrum in dB, same length as freqs.
    """

    impedance = impedance.copy() / 1e-6
    fmin = 1
    fmax = freqs[-1]
    fundamental_i = _find_first_maximum_index(impedance)
    fundamental_freq = freqs[fundamental_i]

    ground = np.zeros(len(freqs))
    indizes = np.concatenate((np.arange(1,fundamental_freq), np.arange(fundamental_freq,fmin-1,-1)))
    window_right = impedance[indizes]

    k = 0.0001
    for i in range(fundamental_freq, fmax, fundamental_freq):

        il = _get_closest_index(freqs, i-fundamental_freq+1)
        ir = np.min((len(freqs)-1, il+len(window_right)))

        window_left = impedance[il:ir]
        if ir-il!=len(window_right):
            window_right = window_right[0:ir-il]

        ground[il:ir] += window_right*np.exp(i*k)

    for i in range(len(ground)):
        ground[i] = impedance[i] * ground[i] * 1e-6

    for i in range(len(ground)):
        x=ground[i]*2e-5
        ground[i] = 0 if x<1 else 20*np.log10(x) 
        # impedance[i] *= 1e-6

    return np.array(ground)

def get_log_simulation_frequencies(fmin: float = 30.0, fmax: float = 1000.0, max_error: float = 5):
    """
    Generate a logarithmically spaced set of frequencies for simulation.

    Frequencies are distributed so that the maximum error (e.g. in cents) between
    adjacent points is bounded by max_error. Useful for efficient impedance
    sweeps with roughly constant resolution per octave.

    Args:
        fmin: Minimum frequency in Hz.
        fmax: Maximum frequency in Hz.
        max_error: Target maximum error (used to derive step size; 1200 steps per octave
            are scaled by max_error).

    Returns:
        np.ndarray: 1D array of frequencies in Hz, ascending, all <= fmax.
    """
    frequencies = []
    stepsize = max_error/1200
    start_freq = fmin
    end_freq = start_freq
    octave = 0

    while end_freq < fmax:
        notes = np.arange(0,1,stepsize) + octave
        frequencies.extend(start_freq*np.power(2, notes))
        end_freq = frequencies[-1]
        octave += 1
        
    frequencies = np.array(list(filter(lambda x:x<=fmax, frequencies)))
    return frequencies



# compute the impedance spektrum iteratively with high precision only
# around the peaks
impedance_iteratively_start_freqs = None
def compute_impedance_iteratively(geo: Geo, fmax=1000, n_precision_peaks=3, simulation_method='tlm_cython'):
    """
    Compute the impedance spectrum with adaptive resolution around peaks.

    Runs an initial coarse simulation, then refines with a denser frequency grid
    around the first few impedance peaks (e.g. drone and toots) to improve
    accuracy there. Results are merged and sorted by frequency.

    Args:
        geo: Didgeridoo geometry (instance of didgelab.geo.Geo).
        fmax: Maximum frequency in Hz for the sweep (default 1000).
        n_precision_peaks: Number of peaks to refine with denser sampling (default 3).
        simulation_method: Backend for the transmission-line simulation
            ('tlm_python' or 'tlm_cython', default 'tlm_cython').

    Returns:
        tuple: (freqs, impedances) — 1D arrays of frequencies in Hz and
            impedance magnitudes, sorted by frequency.
    """
    # start simulation with a low grid size
    global impedance_iteratively_start_freqs
    if impedance_iteratively_start_freqs is None:
        impedance_iteratively_start_freqs = np.concatenate((
            np.arange(1,50,10),
            np.arange(50, 100, 5),
            get_log_simulation_frequencies(fmin=101, fmax=fmax, max_error=10)
        ))
    freqs = [impedance_iteratively_start_freqs]

    impedances = acoustical_simulation(geo, freqs, simulation_method=simulation_method)

    # compute a preciser simulation at the peaks
    extrema = get_max(impedances[0])
    for i in range(np.min((n_precision_peaks, len(extrema)))):
        f = freqs[0][extrema[i]]
        extra_freqs = get_log_simulation_frequencies(fmin=0.9*f, fmax=1.1*f, max_error=2)
        impedances.append(acoustical_simulation(geo, freqs, simulation_method=simulation_method))
        freqs.append(extra_freqs)

    # join and sort
    impedances = np.concatenate(impedances)
    freqs = np.concatenate(freqs)
    i = np.arange(len(impedances))
    i = sorted(i, key=lambda x : freqs[x])
    freqs=freqs[i]
    impedances=impedances[i]

    return freqs, impedances

# interpolate the spectrum that is evenly spaced from freq 1 - fmax
def interpolate_spectrum(freqs, impedances):
    """
    Interpolate an impedance spectrum onto an integer grid from 1 Hz to fmax.

    Produces linearly interpolated impedance values at every integer frequency
    from 1 to the maximum of freqs. Useful for downstream functions that expect
    evenly spaced data (e.g. compute_ground_spektrum).

    Args:
        freqs: 1D array of frequencies in Hz (ascending).
        impedances: 1D array of impedance magnitude at each frequency (same length as freqs).

    Returns:
        tuple: (freq_interpolated, impedance_interpolated) — 1D arrays with
            freq_interpolated = [1, 2, ..., int(freqs[-1])] and interpolated impedances.
    """
    freq_interpolated = np.arange(1, int(np.round(freqs[-1])))
    impedance_interpolated = [impedances[0]]

    i_orig=1
    f_ip = 2
    last_point=0
    while i_orig<len(freqs):

        if freqs[i_orig]>=f_ip or i_orig==len(freqs)-1:
            dx = freqs[i_orig]-freqs[last_point]
            a = (impedances[i_orig]-impedances[last_point]) / dx

            while freqs[i_orig]>=f_ip:
                val = a*(f_ip-freqs[last_point]) + impedances[last_point]
                impedance_interpolated.append(val)
                f_ip +=1

            last_point = i_orig

        i_orig += 1

    return freq_interpolated, np.array(impedance_interpolated)

# get a pandas table about the notes from the resonant spectrum
# you can pass a different base_freq for alternative, non-440 hz tuning
# you can also pass target_freqs to include the deviation from the target into the note list
def get_notes(freqs, impedances, base_freq=440, target_freqs=None):
    """
    Build a table of notes from the resonant peaks of an impedance spectrum.

    Detects local maxima in the impedance, converts their frequencies to note
    names and cent deviations from the nearest semitone, and returns a pandas
    DataFrame. Optionally adds a target column with deviation from given target
    frequencies (e.g. for tuning comparison).

    Args:
        freqs: 1D array of frequencies in Hz.
        impedances: 1D array of impedance magnitude (same length as freqs).
        base_freq: Reference frequency in Hz for note conversion (default 440).
        target_freqs: Optional 1D array of target frequencies in Hz; if provided,
            adds a "target" column with note name and cent deviation from nearest target.

    Returns:
        pd.DataFrame: Columns include note_name, cent_diff, note_nr, freq, impedance,
            rel_imp (impedance relative to max), and optionally target.
    """
    extrema = argrelextrema(impedances, np.greater)
    peak_freqs = freqs[extrema]
    note_and_cent = [freq_to_note_and_cent(f, base_freq=base_freq) for f in peak_freqs]

    peaks = {
        "note_name": [note_name(n[0]) for n in note_and_cent],
        "cent_diff": [n[1] for n in note_and_cent],
        "note_nr": [n[0] for n in note_and_cent],
        "freq": peak_freqs,
        "impedance": impedances[extrema],
    }

    peaks = pd.DataFrame(peaks)
    peaks["rel_imp"] = peaks.impedance / peaks.impedance.max()

    if target_freqs is not None:
        targets = []
        for freq in peak_freqs:
            i = np.argmin(np.abs(freq-target_freqs))
            target = target_freqs[i]
            target_note = note_name(freq_to_note(target))
            cent = cent_diff(target, freq)
            sign = "+" if cent>0 else ""
            cent = f"{sign}{cent:.2f}"
            targets.append(f"{target_note} ({cent})")
        peaks["target"] = targets

    return peaks

def quick_analysis(geo: Geo, fmin=1, fmax=1000, max_error=5, base_freq=440, simulation_method='tlm_cython'):
    """
    Run a full analysis pipeline: impedance sweep, note table, and ground spectrum.

    Computes the impedance spectrum over a log-spaced frequency grid, extracts
    resonant notes (peaks), interpolates to an integer grid, and computes the
    ground spectrum. Convenience function for a single geometry.

    Args:
        geo: Didgeridoo geometry (instance of didgelab.geo.Geo).
        fmin: Minimum frequency in Hz (default 1).
        fmax: Maximum frequency in Hz (default 1000).
        max_error: Parameter for log-spaced frequency resolution (default 5).
        base_freq: Reference frequency in Hz for note conversion (default 440).
        simulation_method: 'tlm_python' or 'tlm_cython' (default 'tlm_cython').

    Returns:
        dict: Keys "freqs", "impedance", "notes" (DataFrame), "ground_freqs",
            "ground_spectrum" (array in dB).
    """
    freqs = get_log_simulation_frequencies(fmin, fmax, max_error)
    impedance = acoustical_simulation(geo, freqs, simulation_method=simulation_method)
    notes = get_notes(freqs, impedance, base_freq=base_freq)
    ground_freqs, imp_ip = interpolate_spectrum(freqs, impedance)
    ground = compute_ground_spektrum(ground_freqs, imp_ip)
    result = {
        "freqs": freqs,
        "impedance": impedance,
        "notes": notes,
        "ground_freqs": ground_freqs,
        "ground_spectrum": ground
    }
    return result


def get_fundamental(geo : Geo, simulation_method : str ='tlm_cython', min_peak_f : float = 50.0):
    """Return frequency and fundamental of the fundamental of the didgeridoo.

    Args:
        geo (Geo): Didgeridoo geometry
        simulation_method (str, optional): Acoustical simulation method. Defaults to 'tlm_cython'.
        min_peak_f (float, optional): Minimum frequency of the fundamental. Defaults to 50.0.

    Returns:
        _type_: _description_
    """
    freqs = get_log_simulation_frequencies()
    impedances = acoustical_simulation(geo, freqs, simulation_method)
    i_peaks = find_peaks(impedances)[0]
    
    peak_f = freqs[i_peaks]
    i_peak = i_peaks[np.arange(len(peak_f))[peak_f > min_peak_f][0]]
    return freqs[i_peak], impedances[i_peak]
    