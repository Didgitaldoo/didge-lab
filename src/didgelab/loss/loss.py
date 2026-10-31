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

    Components that set ``uses_peak_q = True`` also receive ``peak_qs`` (quality factor per
    peak, measured by CompositeTairuaLoss on the fine peak windows) as a keyword argument.

    Components whose loss is a weighted sum of absolute deviations can also implement
    ``residuals``: the signed deviations, so that sum(|r|) equals the loss. Least-squares
    refinement (``didgelab.evo.refine_least_squares``) minimizes sum(r^2) instead, which is
    smooth where the absolute values have a kink, at the target itself.
    """
    uses_peak_q = False

    @abstractmethod
    def calculate(self, peak_freqs_log: np.ndarray, peak_impedances: np.ndarray, 
                  all_freqs: np.ndarray, all_impedances: np.ndarray, peak_indices: np.ndarray) -> float:
        """Computes the loss value based on spectral data."""
        pass

    def residuals(self, peak_freqs_log: np.ndarray, peak_impedances: np.ndarray,
                  all_freqs: np.ndarray, all_impedances: np.ndarray,
                  peak_indices: np.ndarray) -> Optional[np.ndarray]:
        """Signed deviations with sum(|r|) == calculate(...), or None if the component has none."""
        return None

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

    def residuals(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        """Per target: the signed pitch error (cents / 600) and, if targeted, the signed
        impedance error, each times the target weight. Always the same length."""
        residuals = []
        for i, target_f_log in enumerate(self.target_freqs_log):
            idx = np.argmin(np.abs(peak_freqs_log - target_f_log))
            residuals.append(self.weights[i] * 1200 * (peak_freqs_log[idx] - target_f_log) / 600.0)
            if self.target_impedances[i] != -1:
                residuals.append(self.weights[i] * (peak_impedances[idx] - self.target_impedances[i]))
        return np.array(residuals)

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_{tune} = \sum_{i=1}^{N} w_i \cdot \left( \frac{|1200 \Delta \log_2 f_i|}{600} + [Z_{target,i} \neq -1] \cdot |Z_{target,i} - Z_{peak,i}| \right)"
        explanations = [
            "Δ log2 f_i: For each target i, the system finds the closest detected peak to that target; Δ log2 f_i = log2(f_target,i) − log2(f_peak,i) is the pitch error (positive when the peak is below the target).",
            "w_i: Weight for the i-th target peak.",
            "1200: Converts log2 ratio to cents.",
            "600: Normalization factor (6 semitones) so 1 tritone ≈ 1.0 loss.",
            "Z_target,i: Target normalized impedance (0.0 to 1.0) for peak i.",
            "Z_peak,i: Normalized impedance of the closest detected peak to target i.",
            "[Z ≠ −1]: Indicator; impedance loss is zero if target is −1 (ignore)."
        ]
        return formula, explanations

def half_power_q(all_freqs: np.ndarray, all_impedances: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    """Q = f_c / (f_high - f_low) per peak, from the half-power (|Z| / sqrt(2)) crossings
    nearest to the peak, linearly interpolated in log frequency. A side that reaches a valley
    or the end of the spectrum before crossing is cut there (overlapping peaks)."""
    log_f = np.log2(all_freqs)
    n = len(all_impedances)
    qs = []
    for p in peak_indices:
        threshold = all_impedances[p] / np.sqrt(2)
        edges = []
        for step in (-1, 1):
            i = p
            while 0 <= i + step < n and threshold < all_impedances[i + step] <= all_impedances[i]:
                i += step
            j = i + step
            if 0 <= j < n and all_impedances[j] <= threshold:
                frac = (all_impedances[i] - threshold) / (all_impedances[i] - all_impedances[j])
                edges.append(log_f[i] + (log_f[j] - log_f[i]) * frac)
            else:
                edges.append(log_f[i])
        f_low, f_high = np.power(2.0, edges)
        qs.append(all_freqs[p] / (f_high - f_low + 1e-9))
    return np.array(qs)


class QFactorLoss(LossComponent):
    """
    Q-FACTOR: Controls the sharpness/damping of resonances.
    High Q creates a piercing tone; Low Q creates a warm, woody tone.

    Q is the -3 dB quality factor f_c / bandwidth. Inside CompositeTairuaLoss it comes from
    the curvature of the refined peak (``peak_qs``); called directly, it is measured from
    the half-power crossings in the given spectrum.
    """
    uses_peak_q = True

    def __init__(self, target_q: float, weight: float):
        self.target_q = target_q
        self.weight = weight

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices,
                  peak_qs=None):
        if peak_qs is None:
            peak_qs = half_power_q(all_freqs, all_impedances, peak_indices)
        avg_q = np.mean(peak_qs) if len(peak_qs) else 0.0
        return np.abs(avg_q - self.target_q) * self.weight

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = r"L_Q = w \cdot | \left( \frac{1}{N} \sum_{i=1}^{N} \frac{f_{c,i}}{\Delta f_{i,-3dB}} \right) - Q_{target} |"
        explanations = [
            "w: Global weight for Q-factor consistency.",
            "f_c: Center frequency of the peak.",
            "Δf_-3dB: Bandwidth between the half-power points (|Z| / √2) around the peak.",
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
            "f_0: Fundamental (drone) frequency.",
            "n: Harmonic integer (1, 2, 3, …).",
            "N: Number of peaks.",
            "600: Normalization constant (6 semitones).",
            "w: Weight for mathematical purity.",
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
        explanations = [
            "s: Stretch factor (e.g., 1.002 for slightly sharp upper harmonics).",
            "f_n: Actual peak frequency.",
            "n: Harmonic rank.",
            "f_0: Fundamental frequency.",
            "600: Normalization divisor (6 semitones).",
            "w: Weight for stretched-harmonic timbre.",
        ]
        return formula, explanations


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
        explanations = [
            "f_closest: The peak nearest to the target odd harmonic (1st, 3rd, 5th).",
            "f_target: Stretched odd harmonic targets (1·f_0, 3.1·f_0, 5.2·f_0).",
            "600: Normalization divisor.",
            "w: Weight for odd-dominant timbre.",
        ]
        return formula, explanations


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

    Simulation runs in two passes: a coarse log grid (``coarse_step`` cents) finds the peaks,
    then a fine window (``max_error`` cents) around each peak is simulated and the peak
    position and height are refined by parabolic interpolation of log-impedance. This is
    ~6x fewer simulated frequencies than a full ``max_error`` grid and gives sub-cent peak
    frequencies instead of grid-quantized ones. ``all_freqs`` / ``all_impedances`` passed
    to the components are the coarse grid; ``peak_indices`` index into it.
    """
    def __init__(self, max_error: float = 5.0, fmin: float = 30.0, fmax: float = 1000.0,
                 coarse_step: float = 20.0):
        """
        Args:
            max_error: Resolution in cents of the fine window around each peak.
            fmin, fmax: Simulated band in Hz. Didgeridoo drones lie well above 30 Hz.
            coarse_step: Resolution in cents of the peak-finding grid (at least max_error).
        """
        self.max_error = max_error
        self.fmin = fmin
        self.fmax = fmax
        self.coarse_step = coarse_step
        self.components: Dict[str, LossComponent] = {}
        self.target_freqs: Optional[np.ndarray] = None  # for init_standard_evolution / EvolutionMonitor
        self._grid_key = None
        self._grid = None
        self._window_frac = None

    def add_component(self, name: str, component: LossComponent):
        self.components[name] = component

    def _freq_grid(self) -> np.ndarray:
        """Coarse simulation grid, cached until fmin/fmax/coarse_step/max_error change.

        Extends two steps above fmax so a peak just below fmax still has a right neighbour.
        """
        step = max(self.coarse_step, self.max_error)
        key = (self.fmin, self.fmax, step)
        if key != self._grid_key:
            fmax_ext = self.fmax * 2 ** (2 * step / 1200)
            self._grid = get_log_simulation_frequencies(self.fmin, fmax_ext, step)
            self._grid_key = key
        return self._grid

    def _refine_peaks(self, geo, freq_grid, impedances, peak_indices) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (log2 frequency, impedance, Q) of each peak, refined on a max_error window.

        Q comes from the three samples around the peak: for a resonance
        1/|Z|^2 = y0 * (1 + 4 Q^2 delta^2) with delta = (f - f_c) / f_c ~ ln 2 * (log2 f - log2 f_c),
        so a parabola K (x - x0)^2 + y0 through 1/|Z|^2 gives Q = sqrt(K / (4 y0)) / ln 2.
        Peaks where that parabola does not open upwards fall back to the half-power
        crossings on the grid.
        """
        step = max(self.coarse_step, self.max_error)
        n = int(np.ceil(2 * step / self.max_error)) + 1
        if self._window_frac is None or len(self._window_frac) != n:
            self._window_frac = np.linspace(0, 1, n)
        lo = np.log2(freq_grid[peak_indices - 1])
        hi = np.log2(freq_grid[peak_indices + 1])
        window_log = lo[:, None] + (hi - lo)[:, None] * self._window_frac[None, :]
        window_imp = np.asarray(acoustical_simulation(geo, np.power(2.0, window_log.ravel())))
        window_imp = window_imp.reshape(window_log.shape)

        rows = np.arange(len(peak_indices))
        j = np.clip(np.argmax(window_imp, axis=1), 1, n - 2)
        a, b, c = (np.log(window_imp[rows, j + k] + 1e-30) for k in (-1, 0, 1))
        denom = a - 2 * b + c
        delta = np.where(denom < 0, 0.5 * (a - c) / np.where(denom < 0, denom, 1.0), 0.0)
        delta = np.clip(delta, -1.0, 1.0)
        dx = window_log[:, 1] - window_log[:, 0]
        freqs_log = window_log[rows, j] + delta * dx
        imps = np.exp(b - 0.25 * (a - c) * delta)
        ya, yb, yc = (1.0 / window_imp[rows, j + k] ** 2 for k in (-1, 0, 1))
        curv = ya - 2 * yb + yc
        valid = curv > 0
        y0 = yb - (yc - ya) ** 2 / (8 * np.where(valid, curv, 1.0))
        valid &= y0 > 0
        qs = np.sqrt(np.where(valid, curv / (2 * dx ** 2 * 4 * np.where(valid, y0, 1.0)), 0.0)) / np.log(2)
        fallback = ~valid
        if np.any(fallback):
            qs[fallback] = half_power_q(freq_grid, impedances, peak_indices[fallback])
        return freqs_log, imps, qs

    def analyze(self, shape) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
        """
        Simulate ``shape`` and evaluate all components, without caching or error handling.

        Returns:
            (results, spectrum): results maps each component name and "total" to its value;
            spectrum holds the arguments the components are called with ("peak_freqs_log",
            "peak_impedances", "all_freqs", "all_impedances", "peak_indices") plus "peak_qs".
            If no peak is found, results is {"total": 1_000_000} and spectrum is None.
        """
        geo = shape.genome2geo()
        freq_grid = self._freq_grid()
        impedances = np.asarray(acoustical_simulation(geo, freq_grid))

        peak_indices, _ = find_peaks(impedances, prominence=0.05)
        peak_indices = peak_indices[freq_grid[peak_indices] <= self.fmax]

        if len(peak_indices) == 0:
            return {"total": 1_000_000}, None

        peak_freqs_log, peak_imps, peak_qs = self._refine_peaks(geo, freq_grid, impedances, peak_indices)
        peak_impedances = peak_imps / (max(impedances.max(), peak_imps.max()) + 1e-9)
        spectrum = {
            "peak_freqs_log": peak_freqs_log,
            "peak_impedances": peak_impedances,
            "all_freqs": freq_grid,
            "all_impedances": impedances,
            "peak_indices": peak_indices,
            "peak_qs": peak_qs,
        }

        results = {}
        for name, comp in self.components.items():
            results[name] = comp.calculate(*self._component_args(spectrum),
                                           **self._component_kwargs(comp, spectrum))
        results["total"] = sum(results.values())
        return results, spectrum

    def loss(self, shape) -> Dict[str, float]:
        if getattr(shape, "loss", None) is not None:
            return shape.loss

        try:
            results, spectrum = self.analyze(shape)
            if spectrum is None:
                return results
            shape.loss = results
            return results

        except Exception as e:
            logging.exception(f"EA Loss failed: {e}")
            return {"total": 1e8}

    def residuals(self, shape) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """
        Residual vector of ``shape`` for least-squares refinement, and its loss results.

        Components with ``residuals`` contribute their signed deviations; every other
        component contributes sqrt(value), so that sum(r^2) still includes it. The vector is
        None if no peak is found. Its length can change with the number of peaks
        (ScaleTuningLoss has one residual per peak).
        """
        results, spectrum = self.analyze(shape)
        if spectrum is None:
            return None, results
        parts = []
        for name, comp in self.components.items():
            r = comp.residuals(*self._component_args(spectrum))
            parts.append(np.sqrt([max(results[name], 0.0)]) if r is None else np.asarray(r, dtype=float))
        return np.concatenate(parts) if parts else np.zeros(0), results

    @staticmethod
    def _component_args(spectrum):
        return (spectrum["peak_freqs_log"], spectrum["peak_impedances"], spectrum["all_freqs"],
                spectrum["all_impedances"], spectrum["peak_indices"])

    @staticmethod
    def _component_kwargs(comp, spectrum):
        return {"peak_qs": spectrum["peak_qs"]} if getattr(comp, "uses_peak_q", False) else {}


class ScaleTuningLoss(LossComponent):
    """
    SCALE TUNING: Normalized by 600 cents.

    favor_lower_frequencies (p): per-peak weights α ∝ f^(1-p), renormalized so
    sum(α)=N. p=1 is neutral (uniform); p>1 favors lower frequencies; p<1 favors higher.
    """
    def __init__(
        self,
        base_note: int,
        intervals: List[int],
        weight: float,
        favor_lower_frequencies: float = 1.0,
    ):
        self.weight = weight
        self.base_note = base_note
        self.intervals = intervals
        self.favor_lower_frequencies = float(favor_lower_frequencies)
        self.scale_freqs_log = self._compute_scale_log(base_note, intervals)

    def calculate(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        peak_freqs_log = np.asarray(peak_freqs_log, dtype=float)
        n = len(peak_freqs_log)
        if n == 0:
            return 0.0

        if self.scale_freqs_log.size == 0:
            return 0.0

        dists_cents = np.abs(self._signed_dists_cents(peak_freqs_log))
        alphas = self._alphas(peak_freqs_log)
        avg_dist_normalized = float(np.sum(alphas * dists_cents) / n) / 600.0
        return avg_dist_normalized * self.weight

    def residuals(self, peak_freqs_log, peak_impedances, all_freqs, all_impedances, peak_indices):
        """Per peak: the signed distance to the nearest scale note, weighted as in calculate.
        One residual per peak, so the length changes with the number of peaks."""
        peak_freqs_log = np.asarray(peak_freqs_log, dtype=float)
        n = len(peak_freqs_log)
        if n == 0 or self.scale_freqs_log.size == 0:
            return np.zeros(n)
        alphas = self._alphas(peak_freqs_log)
        return self.weight * alphas * self._signed_dists_cents(peak_freqs_log) / (600.0 * n)

    def _signed_dists_cents(self, peak_freqs_log: np.ndarray) -> np.ndarray:
        """Signed cent distance of each peak from its nearest scale note."""
        offsets = peak_freqs_log[:, None] - self.scale_freqs_log[None, :]
        return 1200.0 * offsets[np.arange(len(peak_freqs_log)), np.argmin(np.abs(offsets), axis=1)]

    def _alphas(self, peak_freqs_log: np.ndarray) -> np.ndarray:
        """Per-peak weights ∝ f^(1-p), renormalized so sum(α) = N."""
        n = len(peak_freqs_log)
        p = self.favor_lower_frequencies
        if abs(p - 1.0) < 1e-12:
            return np.ones(n)
        raw = np.power(np.power(2.0, peak_freqs_log), 1.0 - p)
        raw_sum = float(np.sum(raw))
        if not np.isfinite(raw_sum) or raw_sum <= 0:
            return np.ones(n)
        return raw * (n / raw_sum)

    def get_formula(self) -> Tuple[str, List[str]]:
        formula = (
            r"L_{scale} = w \cdot \frac{1}{600N} \sum_{i=1}^{N} "
            r"\alpha_i \min |1200 \cdot (\log_2 f_{peak,i} - \log_2 F_{scale})|, "
            r"\quad \alpha_i \propto f_i^{1-p}"
        )
        explanations = [
            "F_scale: Set of allowed log2 frequencies (scale notes from base + intervals).",
            "f_peak,i: Detected resonance frequency.",
            "min|…|: Distance to the closest in-tune note in the scale.",
            "α_i: Per-peak weight ∝ f_i^(1-p), renormalized so sum(α)=N.",
            "p: favor_lower_frequencies (1=neutral; >1 favors lower; <1 favors higher).",
            "N: Number of peaks.",
            "600: Normalization factor (6 semitones).",
            "w: Weight for scale adherence.",
        ]
        return formula, explanations

    def _compute_scale_log(
        self,
        base_note: int,
        intervals: List[int],
        fmin: float = 1.0,
        fmax: float = 1000.0,
    ) -> np.ndarray:
        """Generate log2 scale frequencies covering the simulation band.

        ``base_note`` is a MIDI note whose pitch class is the scale root
        (evolution_runner uses C4=60 plus the key). Notes are generated in
        every octave that overlaps [fmin, fmax], so drone-range peaks
        (e.g. E2) are scored against the same scale as high toots.
        """
        if not intervals:
            return np.array([], dtype=float)

        root_pc = int(base_note) % 12
        scale_pcs = {(root_pc + int(iv)) % 12 for iv in intervals}

        freqs_log = []
        # MIDI 0 (C-1, ~8.2 Hz) through MIDI 96 (C7, ~2093 Hz).
        for midi in range(0, 97):
            if (midi % 12) not in scale_pcs:
                continue
            freq = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
            if fmin <= freq <= fmax:
                freqs_log.append(np.log2(freq))

        return np.array(freqs_log, dtype=float)

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