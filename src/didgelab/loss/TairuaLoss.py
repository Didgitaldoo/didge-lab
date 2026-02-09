"""
Tairua loss function for evolutionary didgeridoo design.

Penalises deviation of simulated resonances from target frequencies and
impedance ratios, with optional scale tuning and bonuses for more/higher peaks.
"""

from typing import List, Optional
import numpy as np
import logging

from ..evo import LossFunction
from ..acoustical_simulation import acoustical_simulation, get_log_simulation_frequencies
from ..conv import note_to_freq, cent_diff
from scipy.signal import find_peaks


class TairuaLoss(LossFunction):
    """
    Loss that compares simulated impedance peaks to target frequencies and impedances.

    Targets are specified as parallel lists (freqs, impedances, weights). Loss can
    include scale-tuning (pull peaks toward a musical scale), and optional terms
    for encouraging more peaks or higher relative impedance.
    """

    def __init__(
        self,
        target_freqs: List[float],
        target_impedances: List[float],
        freq_weights: List[float],
        impedance_weights: List[float],
        scale_key: Optional[int] = None,
        scale: Optional[List[int]] = None,
        scale_weight: Optional[float] = None,
        higher_peaks: bool = False,
        higher_peaks_weight: float = 1.0,
        more_peaks: bool = False,
        more_peaks_weight: float = 1.0,
        max_error: float = 5.0,
    ):
        """
        Args:
            target_freqs: Target resonance frequencies in Hz (one per peak).
            target_impedances: Target relative impedance (0..1) per peak; use < 0 to ignore.
            freq_weights: Weight per target for frequency error in the total loss.
            impedance_weights: Weight per target for impedance error in the total loss.
            scale_key: Base note number for scale (e.g. A4=0); used only if scale and scale_weight set.
            scale: List of scale intervals in semitones (e.g. [0,2,4,5,7,9,11] for major).
            scale_weight: Weight for scale-tuning term (pull peaks toward scale).
            higher_peaks: If True, add a term that rewards higher mean relative impedance.
            higher_peaks_weight: Weight for the higher_peaks term.
            more_peaks: If True, add a term that rewards more detected peaks.
            more_peaks_weight: Weight for the more_peaks term.
            max_error: Maximum frequency error in cents for simulation grid (log spacing).
        """
        # Target tuning: stored in log2 for consistent cent-based comparison
        self.target_freqs: List[float] = np.log2(np.asarray(target_freqs))
        self.target_impedances: List[float] = list(target_impedances)
        self.freq_weights: List[float] = list(freq_weights)
        self.impedance_weights: List[float] = list(impedance_weights)
        self.max_error: float = max_error
        # Simulation frequency grid (log-spaced for tuning precision)
        self.freqs: np.ndarray = get_log_simulation_frequencies(1, 1000, max_error)

        # Scale tuning: scale_key = base note number, scale = list of intervals in semitones
        self.scale_key: Optional[int] = scale_key
        self.scale: Optional[List[int]] = scale
        self.scale_weight: Optional[float] = scale_weight

        # Optional loss terms
        self.higher_peaks: bool = higher_peaks
        self.higher_peaks_weight: float = higher_peaks_weight
        self.more_peaks: bool = more_peaks
        self.more_peaks_weight: float = more_peaks_weight

        self.tune_to_scale: bool = False
        if scale is not None and scale_key is not None and scale_weight is not None:
            self.scale_freqs: np.ndarray = self.compute_scale_frequencies()
            self.tune_to_scale = True

    def compute_scale_frequencies(self) -> np.ndarray:
        """
        Build list of scale frequencies (in log2) up to 1000 Hz from scale_key + scale intervals.

        Returns:
            Array of log2(frequency) for all scale notes from scale_key and scale up to 1000 Hz.
        """
        scale_freqs = []
        freq = -1
        i = 0
        while freq < 1000:
            octave = np.floor(i / len(self.scale))
            step_in_scale = i % len(self.scale)
            note = self.scale_key + 12 * octave + self.scale[step_in_scale]
            freq = note_to_freq(note)
            i += 1
            scale_freqs.append(freq)
        return np.log2(np.array(scale_freqs))

    def loss(self, shape):
        """
        Evaluate loss for a genome (shape): tuning + optional scale/more_peaks/higher_peaks.

        Converts genome to geometry, runs acoustical simulation, finds impedance peaks,
        then sums weighted frequency and impedance errors vs targets. Caches result on
        `shape.loss` if present.

        Args:
            shape: Genome-like object with `genome2geo()` and optional cached `loss` attribute.

        Returns:
            Dict with at least "total", "freq_loss", "imp_loss"; optionally "scale_losses",
            "more_peaks_loss", "higher_peaks_loss". On exception, returns dict with very large "total".
        """
        if getattr(shape, "loss", None) is not None:
            return shape.loss
        try:
            geo = shape.genome2geo()
            impedances = acoustical_simulation(geo, self.freqs)
            peak_indices = find_peaks(impedances)[0]

            # Peak frequencies and relative impedances (normalised by max)
            peak_freqs = np.array([self.freqs[i] for i in peak_indices])
            peak_freqs_log = np.log2(peak_freqs)
            peak_impedances = np.array([impedances[i] for i in peak_indices]) / impedances.max()

            # Deviation from target tuning table (each target matched to nearest peak)
            freq_loss = []
            imp_loss = []
            for target_freq, target_impedance in zip(self.target_freqs, self.target_impedances):
                i = np.argmin(np.abs(peak_freqs_log - target_freq))
                peak_freq_log = peak_freqs_log[i]
                peak_imp = peak_impedances[i]
                # cent_diff expects linear freq; we have log2, so use 2^log = peak_freq
                diff_cents = cent_diff(
                    np.power(2.0, target_freq),
                    np.power(2.0, peak_freq_log),
                )
                freq_loss.append(np.abs(diff_cents / 600.0))

                if target_impedance >= 0:
                    imp_loss.append(np.abs(peak_imp - target_impedance))

            freq_loss = np.sum(np.array(freq_loss) * self.freq_weights)
            imp_loss = np.sum(np.array(imp_loss) * self.impedance_weights)

            losses = {"freq_loss": freq_loss, "imp_loss": imp_loss}

            # Extra term: pull peaks toward scale frequencies
            if self.tune_to_scale:
                nearest_scale_ix = [np.argmin(np.abs(self.scale_freqs - f)) for f in peak_freqs_log]
                diffs = np.abs(peak_freqs_log - self.scale_freqs[nearest_scale_ix])
                scale_losses = diffs.sum() * self.scale_weight / max(len(diffs), 1) * 10
                losses["scale_losses"] = scale_losses

            if self.more_peaks:
                losses["more_peaks_loss"] = self.more_peaks_weight / max(len(peak_indices), 1)

            if self.higher_peaks:
                losses["higher_peaks_loss"] = self.higher_peaks_weight / np.mean(peak_impedances)

            losses["total"] = sum(losses.values())
            return losses

        except Exception as e:
            logging.exception(e)
            return {
                "total": 100_000_000,
                "tuning_losses": np.array([-1] * len(self.target_freqs)),
                "scale_losses": np.array([-1] * len(self.target_freqs)),
            }
