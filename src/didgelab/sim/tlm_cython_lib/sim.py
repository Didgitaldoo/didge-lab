import os

if os.getenv('CADSD_BACKEND') == "python":
    import didgelab.sim.tlm_cython_lib.cadsd_py as cadsd_imp
else:
    import didgelab.sim.tlm_cython_lib._cadsd as cadsd_imp
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from didgelab.geo import Geo
from didgelab.conv import freq_to_note_and_cent, note_name, note_to_freq, freq_to_note, cent_diff

from abc import abstractmethod

def create_segments(geo):
    return cadsd_imp.create_segments_from_geo(geo.geo)

# compute impedance spectrum
# the raw impedance is not properly leveled
def compute_impedance(segments, frequencies):
    impedance = [1e-6*cadsd_imp.cadsd_Ze(segments, freq) for freq in frequencies]
    impedance = np.array(impedance)
    return impedance

# helper function for compute_ground
def _get_closest_index(freqs, f):
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
def find_first_maximum_index(impedance):

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

    impedance = impedance.copy() / 1e-6
    fmin = 1
    fmax = freqs[-1]
    fundamental_i = find_first_maximum_index(impedance)
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

def get_log_simulation_frequencies(fmin:float, fmax:float, max_error:float):
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

# find maxima or minima of a numpy array
# scipy.signal import argrelextrema caused problems
# code from https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
def get_max(y, find="max"):
    assert find in ("max", "min")
    a = np.diff(np.sign(np.diff(y))).nonzero()[0] + 1 # local min+max

    if find == "min":
        return (np.diff(np.sign(np.diff(y))) > 0).nonzero()[0] + 1 # local min
    elif find == "max":
        return (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0] + 1 # local max

# compute the impedance spektrum iteratively with high precision only
# around the peaks
impedance_iteratively_start_freqs = None
def compute_impedance_iteratively(geo : Geo, fmax=1000, n_precision_peaks=3):

    segments = create_segments(geo)

    # start simulation with a low grid size
    global impedance_iteratively_start_freqs
    if impedance_iteratively_start_freqs is None:
        impedance_iteratively_start_freqs = np.concatenate((
            np.arange(1,50,10),
            np.arange(50, 100, 5),
            get_log_simulation_frequencies(fmin=101, fmax=fmax, max_error=10)
        ))
    freqs = [impedance_iteratively_start_freqs]

    impedances = [compute_impedance(segments, freqs[0])]

    # compute a preciser simulation at the peaks
    extrema = get_max(impedances[0])
    for i in range(np.min((n_precision_peaks, len(extrema)))):
        f = freqs[0][extrema[i]]
        extra_freqs = get_log_simulation_frequencies(fmin=0.9*f, fmax=1.1*f, max_error=2)
        impedances.append(compute_impedance(segments, extra_freqs))
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

def quick_analysis(geo : Geo, fmin=1, fmax=1000, max_error=5, base_freq=440):
    freqs = get_log_simulation_frequencies(fmin, fmax, max_error)
    segments = create_segments(geo)
    impedance = compute_impedance(segments, freqs)
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