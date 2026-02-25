"""
FFT and spectrum utilities for DidgeLab.

Used to compute FFT from WAV files, find harmonic maxima, fundamental frequency,
and peaks in impedance or audio spectra. Supports sliding-window averaging.
"""

import numpy as np
from scipy.io import wavfile
import scipy


def do_fft(infile, maxfreq=1000):
    """Load WAV file and return (freq, magnitude) up to maxfreq Hz. Uses left channel if stereo."""
    sampFreq, sound = wavfile.read(infile)
    
    # use only left channel if signal is stereo
    if len(sound.shape)==2:
        signal = sound[:,0]
    else:
        signal = sound

    size=len(signal)
    
    fft_spectrum = np.fft.rfft(signal, n=size)
    freq = np.fft.rfftfreq(size, d=1./sampFreq)
    fft_spectrum_abs = np.abs(fft_spectrum)
 
    i=0
    while i<len(freq) and freq[i]<=maxfreq:
        i+=1
    freq = freq[0:i]
    fft_spectrum_abs = fft_spectrum_abs[0:i]
    return freq, fft_spectrum_abs


def get_harmonic_maxima(freq, spectrum, min_freq=60):
    """Find maxima near expected harmonics (multiples of base_freq) up to ~1000 Hz."""
    i = 0
    maxima = []
    base_freq = min_freq
    while i*base_freq<1000:
        if i==0:
            window = freq>min_freq
        else:
            window = (freq>(i+0.5)*base_freq) & (freq<base_freq*(i+1.5))

        if window.astype(int).sum() == 0:
            break
        window_f = freq[window]
        window_s = spectrum[window]
        maxi = np.argmax(window_s)
        max_f = window_f[maxi]
        if i==0:
            base_freq=max_f

        maxima.append(max_f)
        i += 1
    return maxima


def sling_window_average_spectrum(freq, spectrum, window_size=5):
    """Downsample spectrum by averaging over sliding windows (window_size points)."""
    new_freqs = []
    new_spectrum = []
    
    for i in np.arange(window_size, len(freq), window_size):
        new_freqs.append(freq[i])
        new_spectrum.append(np.mean(spectrum[i - window_size:i]))
    return np.array(new_freqs), np.array(new_spectrum)


def get_fundamental(fft_freq, fft, minfreq=50, maxfreq=120, order=40):
    """Find fundamental as the largest local maximum in [minfreq, maxfreq] (argrelextrema order=order)."""
    i = scipy.signal.argrelextrema(fft, np.greater, order=order)
    freqs = fft_freq[i]
    freqs = freqs[freqs>minfreq]
    freqs = freqs[freqs<maxfreq]
    assert len(freqs) == 1

    fundamental_freq = freqs[0]
    fundamental_freq_i = np.argmin(np.abs(fft_freq - fundamental_freq))
    return fundamental_freq, fundamental_freq_i


def get_peaks(fft_freq, fft, return_indizes=False):
    """Peaks in fft above fundamental (within ~1.3x fundamental); optionally return indices."""
    fundamental_freq, fundamental_freq_i = get_fundamental(fft_freq, fft)
    order = 1
    while fft_freq[fundamental_freq_i+order] < fundamental_freq*1.3:
        order += 1
    peaks = [fundamental_freq]
    indizes = []
    for i in scipy.signal.argrelextrema(fft, np.greater, order=order)[0]:
        freq = fft_freq[i]
        indizes.append(i)
        if freq>fundamental_freq:
            peaks.append(freq)
    if return_indizes:
        return np.array(peaks), indizes
    else:
        return np.array(peaks)
