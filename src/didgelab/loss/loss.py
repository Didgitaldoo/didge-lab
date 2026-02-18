from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import logging
from abc import ABC, abstractmethod
from scipy.signal import find_peaks
from ..acoustical_simulation import acoustical_simulation, get_log_simulation_frequencies

# Assuming these are available in your environment
# from ..evo import LossFunction
# from ..acoustical_simulation import acoustical_simulation, get_log_simulation_frequencies
# from ..conv import note_to_freq, cent_diff

class LossComponent(ABC):
    """
    Abstract base for individual loss terms.
    Each component evaluates a specific psychoacoustic or physical property.
    """
    
    @abstractmethod
    def calculate(self, peak_freqs_log: np.ndarray, peak_impedances: np.ndarray, 
                  all_freqs: np.ndarray, all_impedances: np.ndarray, peak_indices: np.ndarray) -> float:
        """Computes the loss value based on spectral data."""
        pass

    @abstractmethod
    def get_formula(self) -> Tuple[str, List[str]]:
        """
        Returns a tuple:
        - [0]: LaTeX formula string
        - [1]: List of strings explaining each symbol used
        """
        pass

# --- Loss Components ---

class FrequencyTuningLoss(LossComponent):
    """
    PITCH & AMPLITUDE TUNING: 
    Aligns peaks to specific frequencies and optionally specific impedance magnitudes.
    Normalization: Frequency error is divided by 600 cents. Impedance error is absolute.
    """
    def __init__(self, 
                 target_freqs_log: np.ndarray, 
                 target_impedances: np.ndarray, 
                 weights: List[float]):
        """
        Args:
            target_freqs_log: Array of target frequencies in log2.
            target_impedances: Array of target normalized impedances [0, 1]. 
                               Set to -1 to ignore impedance loss for that peak.
            weights: Per-peak weights for the total loss calculation.
        """
        self.target_freqs_log = target_freqs_log
        self.target_impedances = target_impedances
        self.weights = weights

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        total_loss = 0.0
        
        for i, target_f_log in enumerate(self.target_freqs_log):
            # 1. Find the closest actual peak to this target frequency
            idx = np.argmin(np.abs(peak_freqs_log - target_f_log))
            actual_f_log = peak_freqs_log[idx]
            actual_amp = peak_impedances[idx]
            
            # 2. Calculate Frequency Loss (Normalized Cents)
            # 600 cents (a tritone) = 1.0 base loss unit
            freq_error_cents = 1200 * np.abs(target_f_log - actual_f_log)
            freq_loss = freq_error_cents / 600.0
            
            # 3. Calculate Impedance Loss (if target != -1)
            amp_loss = 0.0
            if self.target_impedances[i] != -1:
                amp_loss = np.abs(self.target_impedances[i] - actual_amp)
            
            # 4. Combine with per-peak weight
            total_loss += (freq_loss + amp_loss) * self.weights[i]
            
        return total_loss

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{tune} = \sum_{i=1}^{N} w_i \cdot \left( \frac{|1200 \Delta \log_2 f_i|}{600} + [Z_{target,i} \neq -1] \cdot |Z_{target,i} - Z_{peak,i}| \right)"
        explanations = [
            "w_i: Specific weight for the i-th peak.",
            "1200/600: Normalizes cent deviation so 1 semitone ≈ 0.166 and 1 tritone = 1.0.",
            "Z_target: Target normalized impedance (0.0 to 1.0).",
            "[Z != -1]: Indicator function; impedance loss is zero if target is -1."
        ]
        return formula, explanations

class QFactorLoss(LossComponent):
    """
    Q-FACTOR: Controls the sharpness/damping of resonances. 
    High Q creates a piercing tone; Low Q creates a warm, woody tone.
    """
    def __init__(self, target_q: float, weight: float):
        self.target_q = target_q
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        qs = []
        for p_idx in peak_indices:
            f_center = all_freqs[p_idx]
            target_amp = all_impedances[p_idx] / np.sqrt(2)
            
            left_side = all_impedances[:p_idx]
            f_low = all_freqs[np.argmin(np.abs(left_side - target_amp))]
            
            right_side = all_impedances[p_idx:]
            f_high = all_freqs[p_idx + np.argmin(np.abs(right_side - target_amp))]
            
            qs.append(f_center / (f_high - f_low + 1e-9))
            
        avg_q = np.mean(qs) if qs else 0.0
        return np.abs(avg_q - self.target_q) * self.weight

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_Q = w \cdot | \left( \frac{1}{N} \sum_{i=1}^{N} \frac{f_{c,i}}{\Delta f_{i,-3dB}} \right) - Q_{target} |"
        explanations = [
            "w: Global weight for Q-factor consistency.",
            "f_c: Center frequency of the peak.",
            "Δf_-3dB: Bandwidth at half-power (Full Width at Half Maximum).",
            "Q_target: The desired quality factor.",
            "N: Number of peaks analyzed."
        ]
        return formula, explanations

class ModalDensityLoss(LossComponent):
    """
    SHIMMERING CHORUS: Rewards the proximity of multiple peaks to create interference beats.
    """
    def __init__(self, cluster_range_cents: float, weight: float):
        self.cluster_range_cents = cluster_range_cents
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        if len(peak_freqs_log) < 2: return self.weight
        diffs = np.diff(peak_freqs_log) * 1200
        shimmer_score = np.sum(np.exp(-np.square(diffs - self.cluster_range_cents) / 100.0))
        return self.weight / (1.0 + shimmer_score)

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{shimmer} = \frac{w}{1 + \sum e^{-\frac{(\Delta c - C)^2}{\sigma^2}}}"
        explanations = [
            "Δc: Cent distance between adjacent peaks.",
            "C: Target cluster range (sweet spot for beating).",
            "σ: Smoothing factor (bandwidth of the shimmer reward).",
            "w: Importance of the shimmering texture."
        ]
        return formula, explanations

class HarmonicSplittingLoss(LossComponent):
    """
    NODAL TURBULENCE: Drives the EA to split a harmonic into a 'gritty' doublet.
    """
    def __init__(self, harmonic_index: int, split_width_hz: float, weight: float):
        self.h_idx = harmonic_index
        self.split_width = split_width_hz
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        if len(peak_freqs_log) <= self.h_idx: return self.weight
        f_target = np.power(2.0, peak_freqs_log[self.h_idx])
        close_peaks = np.sum(np.abs(np.power(2.0, peak_freqs_log) - f_target) < self.split_width)
        return 0.0 if close_peaks >= 2 else self.weight

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{split} = w \cdot [N(f_{peaks} \in [f_n \pm \delta]) < 2]"
        explanations = [
            "f_n: The target harmonic frequency to be split.",
            "δ: The frequency window (split width) in Hz.",
            "N(...): Count of peaks within that window.",
            "[...]: Iverson bracket (binary penalty if condition is true)."
        ]
        return formula, explanations

class IntegerHarmonicLoss(LossComponent):
    """
    PERFECT-INTEGER: Normalized by 600 cents.
    """
    def __init__(self, weight: float):
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        f0 = np.power(2.0, peak_freqs_log[0])
        # Calculate mean error in cents and divide by 600
        total_error_cents = sum(np.abs(1200 * np.log2(np.power(2.0, f_log) / (f0 * (i + 1)))) 
                               for i, f_log in enumerate(peak_freqs_log))
        avg_error_normalized = (total_error_cents / len(peak_freqs_log)) / 600.0
        return avg_error_normalized * self.weight

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{int} = w \cdot \frac{1}{600N} \sum_{n=1}^{N} |1200 \cdot \log_2 \left( \frac{f_n}{n \cdot f_0} \right)|"
        explanations = [
            "f_n: Frequency of the n-th peak.",
            "600: Normalization constant to balance cent-based loss."
        ]
        return formula, explanations

class NearIntegerLoss(LossComponent):
    """
    NEAR-INTEGER: Normalized by 600 cents.
    """
    def __init__(self, stretch_factor: float, weight: float):
        self.stretch = stretch_factor
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        f0 = np.power(2.0, peak_freqs_log[0])
        total_error_cents = sum(np.abs(1200 * np.log2(np.power(2.0, f_log) / (f0 * (i + 1) * (self.stretch ** i)))) 
                               for i, f_log in enumerate(peak_freqs_log))
        return (total_error_cents / 600.0) * self.weight

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{near} = w \cdot \frac{1}{600} \sum_{n=1}^{N} |1200 \cdot \log_2 \left( \frac{f_n}{n \cdot f_0 \cdot s^n} \right)|"
        return formula, ["s: Stretch factor", "600: Normalization divisor"]


class StretchedOddLoss(LossComponent):
    """
    STRETCHED/ODD: Normalized by 600 cents.
    """
    def __init__(self, weight: float):
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        f0 = np.power(2.0, peak_freqs_log[0])
        targets = [f0 * 1, f0 * 3.1, f0 * 5.2]
        total_loss_cents = sum(np.abs(1200 * np.log2(np.power(2.0, peak_freqs_log[np.argmin(np.abs(np.power(2.0, peak_freqs_log) - t))]) / t)) 
                              for t in targets)
        return (total_loss_cents / 600.0) * self.weight

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{odd} = \frac{w}{600} \cdot \sum |1200 \cdot \log_2 \left( \frac{f_{closest}}{f_{target}} \right)|"
        return formula, ["f_{target}: Stretched odd harmonic targets", "600: Normalization divisor"]


class HighInharmonicLoss(LossComponent):
    """
    HIGH INHARMONIC: Maximizes 'dissonance' by pushing ratios toward irrational numbers.
    """
    def __init__(self, weight: float):
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        f0 = np.power(2.0, peak_freqs_log[0])
        ratios = np.power(2.0, peak_freqs_log) / f0
        dist_to_int = np.abs(ratios - np.round(ratios)).mean()
        return self.weight * (0.5 - dist_to_int)

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{inharm} = w \cdot (0.5 - \text{avg}(| \frac{f_n}{f_0} - \lfloor \frac{f_n}{f_0} \rceil |))"
        explanations = [
            "⌊x⌉: Nearest integer (rounding).",
            "f_n/f_0: Harmonic ratio.",
            "0.5: Maximum possible deviation from an integer ratio.",
            "w: Weight for metallic/chaotic timbre."
        ]
        return formula, explanations

# --- Orchestrator ---

class CompositeTairuaLoss:
    """
    Orchestrates acoustic simulation and multi-objective loss distribution.
    Uses real acoustical_simulation when available (didgelab); falls back to dummy data otherwise.
    Compatible with Nuevolution: implement .loss(shape) returning a dict with "total".
    Set .target_freqs (1D array of target frequencies in Hz) when using with init_standard_evolution.
    """
    def __init__(self, max_error: float = 5.0):
        self.max_error = max_error
        self.components: Dict[str, LossComponent] = {}
        self.target_freqs: Optional[np.ndarray] = None  # for init_standard_evolution / EvolutionMonitor

    def add_component(self, name: str, component: LossComponent):
        self.components[name] = component

    def loss(self, shape) -> Dict[str, float]:
        if getattr(shape, "loss", None) is not None:
            return shape.loss

        try:
            geo = shape.genome2geo()
            freq_grid = get_log_simulation_frequencies(1, 1000, self.max_error)
            impedances = np.asarray(acoustical_simulation(geo, freq_grid))

            peak_indices, _ = find_peaks(impedances, prominence=0.05)

            if len(peak_indices) == 0:
                return {"total": 1_000_000}

            peak_freqs_log = np.log2(freq_grid[peak_indices])
            peak_impedances = impedances[peak_indices] / (impedances.max() + 1e-9)

            results = {}
            for name, comp in self.components.items():
                results[name] = comp.calculate(peak_freqs_log, peak_impedances, freq_grid, impedances, peak_indices)

            results["total"] = sum(results.values())
            shape.loss = results
            return results

        except Exception as e:
            logging.exception(f"EA Loss failed: {e}")
            return {"total": 1e8}


class ScaleTuningLoss(LossComponent):
    """
    SCALE TUNING: Normalized by 600 cents.
    """
    def __init__(self, base_note: int, intervals: List[int], weight: float):
        self.weight = weight
        self.base_note = base_note
        self.intervals = intervals
        self.scale_freqs_log = self._compute_scale_log(base_note, intervals)

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        total_dist_cents = 0
        for f_log in peak_freqs_log:
            dist = np.min(np.abs(self.scale_freqs_log - f_log))
            total_dist_cents += dist * 1200 
        avg_dist_normalized = (total_dist_cents / len(peak_freqs_log)) / 600.0
        return avg_dist_normalized * self.weight

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{scale} = w \cdot \frac{1}{600N} \sum_{i=1}^{N} \min |1200 \cdot (\log_2 f_{peak,i} - \log_2 F_{scale})|"
        return formula, ["F_scale: Allowed log2 frequencies", "600: Normalization factor"]

    def _compute_scale_log(self, base_note: int, intervals: List[int]) -> np.ndarray:
        # Generates a log2 frequency scale across the human hearing range
        freqs_log = []
        for octave in range(0, 5): # Cover 5 octaves
            for interval in intervals:
                # MIDI-style note calculation: note = base + octave*12 + interval
                note = base_note + (octave * 12) + interval
                # Conversion: freq = 440 * 2^((note-69)/12)
                freq = 440.0 * (2.0 ** ((note - 69.0) / 12.0))
                freqs_log.append(np.log2(freq))

        return np.array(freqs_log)

class PeakQuantityLoss(LossComponent):
    """
    PEAK QUANTITY (More Peaks): Penalizes 'dead' bore shapes that only support 
    one or two resonances. Encourages complex, multi-resonant geometry.
    """
    def __init__(self, target_count: int, weight: float):
        self.target_count = target_count
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        current_count = len(peak_indices)
        # Loss is high if we have fewer peaks than the target
        diff = max(0, self.target_count - current_count)
        return diff * self.weight

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{qty} = w \cdot \max(0, N_{target} - N_{actual})"
        explanations = [
            "N_target: Minimum desired number of resonance peaks.",
            "N_actual: Number of peaks currently detected in the simulation.",
            "w: Penalty per missing peak."
        ]
        return formula, explanations

class PeakAmplitudeLoss(LossComponent):
    """
    PEAK AMPLITUDE (Higher Peaks): Penalizes low-impedance resonances.
    Higher peaks generally correlate to better 'backpressure' and 
    more efficient energy transfer from the player's lips.
    """
    def __init__(self, target_min_amplitude: float, weight: float):
        self.target_min_amplitude = target_min_amplitude
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        # target_min_amplitude should be normalized relative to your simulation's max
        # Penalize the average peak height if it falls below target
        avg_amp = np.mean(peak_impedances)
        loss = max(0, self.target_min_amplitude - avg_amp)
        return loss * self.weight

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{amp} = w \cdot \max(0, A_{target} - \bar{A}_{peaks})"
        explanations = [
            "A_target: Desired average normalized impedance amplitude.",
            "A_bar: Mean amplitude of all detected peaks.",
            "w: Weight for resonance strength."
        ]
        return formula, explanations