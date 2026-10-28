"""
Synthesis engine.

The sound is rendered from *control signals* sampled at a control rate of
typically 500 Hz (see :class:`Controls`):

* the lip tone is an additive bank of harmonics n*f0(t), each with the complex
  amplitude  A_n(t) = amp(t) * S_n(oq(t), skew(t)) * G(n f0(t), vt(t)),
  where S_n is the lip source spectrum and G the filter of ``model.py``;
* vocalisation (singing into the instrument) modulates the lip flow and adds
  sum- and difference tones n*f0 + k*fv (k = +-1, +-2, ...);
* breath noise and frication bursts are filtered by the same G in the STFT
  domain; the nasal inhalation of circular breathing bypasses the bore.
"""

from dataclasses import dataclass, field

import numpy as np

from . import vocaltract
from .model import ModelParams, filter_gain, source
from .excitation import voice_harmonics


@dataclass
class Controls:
    """Control signals, all arrays of length ``n`` sampled at ``rate`` Hz."""
    rate: float
    f0: np.ndarray            # lip frequency, Hz
    amp: np.ndarray           # lip oscillation amplitude (linear)
    oq: np.ndarray            # open quotient
    skew: np.ndarray          # pulse skew
    vt: np.ndarray            # (n, 4) vocal tract parameters
    breath: np.ndarray        # breath (lip) noise amplitude
    frication: np.ndarray     # frication / burst noise amplitude
    fric_freq: np.ndarray     # centre frequency of frication noise, Hz
    voice: np.ndarray         # vocalisation amplitude (0 = off)
    voice_f0: np.ndarray      # vocalisation pitch, Hz
    sniff: np.ndarray         # nasal inhalation noise (circular breathing)
    labels: list = field(default_factory=list)   # (start_s, end_s, label) of the score events
    gestures: list = field(default_factory=list) # (start_s, end_s, label) of articulatory gestures (ground truth)

    @property
    def n(self):
        return len(self.f0)

    @property
    def duration(self):
        return self.n / self.rate

    @classmethod
    def constant(cls, duration, rate, f0, params: ModelParams, vt):
        n = int(round(duration * rate))
        one = np.ones(n)
        vt = np.asarray(vt.as_array() if hasattr(vt, "as_array") else vt, float)
        return cls(rate, f0 * one, one.copy(), params.oq * one, params.skew * one,
                   np.tile(vt, (n, 1)), 10 ** (params.noise_db / 20) * one,
                   0 * one, 3000 * one, 0 * one, 150 * one, 0 * one)


def _upsample(x, n_out, rate, fs):
    """Linear interpolation of a control signal (n,) or (n, k) to audio rate."""
    t_c = np.arange(len(x)) / rate
    t_a = np.arange(n_out) / fs
    if x.ndim == 1:
        if np.iscomplexobj(x):
            return np.interp(t_a, t_c, x.real) + 1j * np.interp(t_a, t_c, x.imag)
        return np.interp(t_a, t_c, x)
    return np.stack([_upsample(x[:, i], n_out, rate, fs) for i in range(x.shape[1])], 1)


class Synthesizer:
    """Render :class:`Controls` for a given bore.

    Parameters
    ----------
    bore : didgesynth.bore.Bore
    params : ModelParams
    fs : sample rate of the output
    fmax : highest partial frequency
    """

    def __init__(self, bore, params: ModelParams = ModelParams(), fs=48000, fmax=10000.0,
                 seed=None, inertia=True):
        self.bore = bore
        self.params = params
        self.fs = fs
        self.fmax = min(fmax, 0.45 * fs, bore.freqs[-1])
        self.rng = np.random.default_rng(seed)
        self.inertia = inertia
        self._tau = None

    # ------------------------------------------------------- bore inertia
    def envelope_time_constant(self, f):
        """Time constant (s) with which the envelope of a partial at ``f`` can change.

        The bore stores energy: near a resonance of bandwidth B a partial
        builds up and decays with tau = 1/(pi B) (= Q / (pi f)). This equals
        the group delay -d(arg H)/d(omega) of the bore's flow transfer
        function H with the lips as a (nearly) closed end, which is used here
        (clipped to 2..300 ms).
        """
        if self._tau is None:
            b = self.bore
            phase = np.unwrap(np.angle(b.H))
            tg = -np.gradient(phase, 2 * np.pi * b.freqs)
            # smooth over 2 Hz to remove numerical noise, keep the peaks
            k = max(1, int(round(2.0 / (b.freqs[1] - b.freqs[0]))))
            tg = np.convolve(tg, np.ones(k) / k, mode="same")
            self._tau = np.clip(tg, 0.002, 0.3)
        return np.interp(f, self.bore.freqs, self._tau)

    def _apply_inertia(self, A, freqs, rate):
        """Time-varying one-pole smoothing of each partial's complex amplitude."""
        tau = self.envelope_time_constant(freqs)          # (K, N)
        a = np.exp(-1.0 / (tau * rate))
        out = np.empty_like(A)
        acc = A[0].copy()
        for k in range(A.shape[0]):
            acc = a[k] * acc + (1 - a[k]) * A[k]
            out[k] = acc
        return out

    # ---------------------------------------------------------- vocal tract
    VT_DF = 20.0

    def _prepare_vt(self, c: Controls):
        """Z_vt of every control frame on a 20 Hz grid (unique articulations only)."""
        self._vt_f = np.arange(0.0, self.fmax + 2 * self.VT_DF, self.VT_DF)
        q = np.round(c.vt, 3)
        uniq, inv = np.unique(q, axis=0, return_inverse=True)
        Z = np.empty((len(uniq), len(self._vt_f)), dtype=complex)
        F = np.maximum(self._vt_f, 1.0)
        for s in range(0, len(uniq), 500):
            e = min(len(uniq), s + 500)
            Z[s:e] = vocaltract.impedance(uniq[s:e], np.tile(F, (e - s, 1)))
        self._vt_Z, self._vt_inv = Z, inv.ravel()

    def _zvt(self, freqs, rows=None):
        """Interpolated Z_vt for freqs of shape (K, N) (rows = control frames)."""
        rows = np.arange(len(self._vt_inv)) if rows is None else rows
        Z = self._vt_Z[self._vt_inv[rows]]
        pos = np.clip(freqs / self.VT_DF, 0, len(self._vt_f) - 1.001)
        i = pos.astype(int)
        a = pos - i
        r = np.arange(Z.shape[0])[:, None]
        return Z[r, i] * (1 - a) + Z[r, i + 1] * a

    # ------------------------------------------------------------------ tone
    def _harmonic_amplitudes(self, c: Controls, n_harm):
        n = np.arange(1, n_harm + 1)
        freqs = c.f0[:, None] * n[None, :]
        src = source(self.params, n_harm, oq=c.oq, skew=c.skew)          # (K, N)
        A = filter_gain(self.bore, freqs, None, self.params, zvt=self._zvt(freqs))
        A *= src * c.amp[:, None]
        A[freqs > self.fmax] = 0
        if self.inertia:
            A = self._apply_inertia(A, freqs, c.rate)
        return A

    def _render_bank(self, A, n_mult, k_mult, theta, phi, rate, chunk_frames=96):
        """Additive synthesis  y = sum_p Re(A_p(t) z^n_p w^k_p)

        with z = exp(i theta), w = exp(i phi) (audio-rate phases), integer
        multipliers n_p >= 0, k_p (may be negative) and complex amplitudes
        A (P, K) at control rate, linearly interpolated to audio rate.
        Powers of z and w are built by repeated multiplication (no exp per
        partial).
        """
        n_out = len(theta)
        keep = np.any(np.abs(A) > 0, axis=1)
        A, n_mult, k_mult = A[keep], n_mult[keep], k_mult[keep]
        if len(A) == 0:
            return np.zeros(n_out)
        n_max = int(np.max(n_mult))
        k_max = int(np.max(np.abs(k_mult))) if phi is not None else 0
        hop = self.fs / rate
        if abs(hop - round(hop)) > 1e-9:
            raise ValueError("fs must be an integer multiple of the control rate")
        hop = int(round(hop))
        A = np.concatenate([A, A[:, -1:]], axis=1)        # guard frame
        fr = np.arange(hop) / hop
        y = np.zeros(n_out)
        K = A.shape[1] - 1
        for k0 in range(0, K, chunk_frames):
            k1 = min(K, k0 + chunk_frames)
            s, e = k0 * hop, min(n_out, k1 * hop)
            if s >= e:
                break
            a0, a1 = A[:, k0:k1], A[:, k0 + 1:k1 + 1]
            Aup = (a0[:, :, None] * (1 - fr) + a1[:, :, None] * fr).reshape(len(A), -1)[:, :e - s]
            z = np.exp(1j * theta[s:e])
            zp = np.empty((n_max + 1, e - s), dtype=complex)
            zp[0] = 1.0
            for m in range(1, n_max + 1):
                zp[m] = zp[m - 1] * z
            ph = zp[n_mult]
            if k_max:
                w = np.exp(1j * phi[s:e])
                wp = np.empty((k_max + 1, e - s), dtype=complex)
                wp[0] = 1.0
                for m in range(1, k_max + 1):
                    wp[m] = wp[m - 1] * w
                wk = wp[np.abs(k_mult)]
                ph = ph * np.where((k_mult < 0)[:, None], np.conj(wk), wk)
            y[s:e] = (Aup.real * ph.real - Aup.imag * ph.imag).sum(0)
        return y

    def render_tone(self, c: Controls, n_out):
        fs = self.fs
        f0_a = _upsample(c.f0, n_out, c.rate, fs)
        theta = 2 * np.pi * np.cumsum(f0_a) / fs
        n_harm = int(self.fmax / max(np.min(c.f0), 20.0))
        A = self._harmonic_amplitudes(c, n_harm)
        n_mult = np.arange(1, n_harm + 1)
        y = self._render_bank(A.T, n_mult, np.zeros_like(n_mult), theta, None, c.rate)
        if np.any(c.voice > 0):
            y += self.render_voice(c, n_out, theta)
        return y, theta

    def render_voice(self, c: Controls, n_out, theta, n_lip=16, n_voice=6, depth=1.0):
        """Vocalisation: lip flow u(t) * (1 + depth*v(t)) + v(t).

        With u = Re sum S_n z^n and v = Re sum V_k w^k the product contains
        the partials  1/2 S_n V_k z^n w^k  and  1/2 S_n conj(V_k) z^n w^-k,
        i.e. sum and difference tones n*f0 +- k*fv. Each partial is filtered
        by G at its frequency |n f0 +- k fv| (conjugated for negative
        instantaneous frequency).
        """
        fs = self.fs
        fv_a = _upsample(c.voice_f0, n_out, c.rate, fs)
        phi = 2 * np.pi * np.cumsum(fv_a) / fs
        V = voice_harmonics(n_voice)
        S = source(self.params, n_lip, oq=c.oq, skew=c.skew)       # (K, n_lip)
        amps, nm, km = [], [], []
        for k in range(1, n_voice + 1):
            for n in range(0, n_lip + 1):
                for sgn in ([1, -1] if n > 0 else [1]):
                    f = n * c.f0 + sgn * k * c.voice_f0
                    vk = V[k - 1] if sgn > 0 else np.conj(V[k - 1])
                    coef = depth * 0.5 * S[:, n - 1] * vk if n > 0 else vk
                    fq = np.maximum(np.abs(f), 1.0)[:, None]
                    G = filter_gain(self.bore, fq, None, self.params, zvt=self._zvt(fq))[:, 0]
                    a = coef * c.voice * c.amp * np.where(f >= 0, G, np.conj(G))
                    a[(np.abs(f) > self.fmax) | (np.abs(f) < 20)] = 0
                    amps.append(a)
                    nm.append(n)
                    km.append(sgn * k)
        # render only the stretch of frames where the voice is active
        act = np.where(c.voice > 1e-4)[0]
        k0, k1 = act[0], min(c.n, act[-1] + 2)
        hop = int(round(fs / c.rate))
        s, e = k0 * hop, min(n_out, k1 * hop)
        y = np.zeros(n_out)
        y[s:e] = self._render_bank(np.array(amps)[:, k0:k1], np.array(nm), np.array(km),
                                   theta[s:e], phi[s:e], c.rate)
        return y

    # ----------------------------------------------------------------- noise
    def render_noise(self, c: Controls, n_out, theta, win=2048, hop=512):
        """Breath and frication noise, filtered by G frame by frame (STFT)."""
        fs = self.fs
        breath = _upsample(c.breath, n_out, c.rate, fs) * (1 + 0.8 * np.cos(theta)) / 1.8
        fric = _upsample(c.frication, n_out, c.rate, fs)
        nb = self.rng.standard_normal(n_out) * breath
        nf = self.rng.standard_normal(n_out) * fric
        w = np.hanning(win)
        freqs = np.fft.rfftfreq(win, 1 / fs)
        sub = freqs[:, None] + (np.arange(8) - 3.5)[None, :] * (fs / win / 8)
        out = np.zeros(n_out + win)
        pad_b = np.concatenate([nb, np.zeros(win)])
        pad_f = np.concatenate([nf, np.zeros(win)])
        frames = range(0, n_out, hop)
        # reference level: breath noise relative to the tone uses |G| normalised
        # to its RMS over the tone's harmonic range
        for s in frames:
            k = min(int((s + win / 2) / fs * c.rate), c.n - 1)
            if c.breath[k] <= 0 and c.frication[k] <= 0:
                continue
            fq = np.clip(sub, 1, None).ravel()[None, :]
            g = filter_gain(self.bore, fq, None, self.params, zvt=self._zvt(fq, np.array([k])))
            g = np.sqrt(np.mean(np.abs(g.reshape(sub.shape)) ** 2, axis=1))
            g[freqs > self.fmax] = 0
            fc = c.fric_freq[k]
            color = 1 / (1 + ((freqs - fc) / (0.5 * fc)) ** 2)
            Xb = np.fft.rfft(pad_b[s:s + win] * w)
            Xf = np.fft.rfft(pad_f[s:s + win] * w)
            out[s:s + win] += np.fft.irfft((Xb + Xf * color) * g, win) * w
        return out[:n_out] / 1.5 * self._noise_scale(c)

    def _noise_scale(self, c):
        # breath noise level is defined relative to the RMS of a harmonic
        # with the same filter gain: a unit-variance white noise has power
        # fs/2 spread over the band; scale so that 1.0 ~ one harmonic per 100 Hz
        return np.sqrt(100.0 / (self.fs / 2))

    def render_sniff(self, c: Controls, n_out):
        """Nasal inhalation noise (bypasses the bore), band-limited 1.5-6 kHz."""
        if not np.any(c.sniff > 0):
            return np.zeros(n_out)
        env = _upsample(c.sniff, n_out, c.rate, self.fs)
        X = np.fft.rfft(self.rng.standard_normal(n_out))
        f = np.fft.rfftfreq(n_out, 1 / self.fs)
        X *= np.exp(-0.5 * (np.log2(np.maximum(f, 1) / 3000) / 0.8) ** 2)
        return np.fft.irfft(X, n_out) * env

    # ------------------------------------------------------------------ main
    def render(self, c: Controls, reverb=0.25, normalize=True):
        n_out = int(round(c.duration * self.fs))
        self._prepare_vt(c)
        tone, theta = self.render_tone(c, n_out)
        noise = self.render_noise(c, n_out, theta)
        y = tone + noise
        ref = np.sqrt(np.mean(tone ** 2)) + 1e-12
        y += self.render_sniff(c, n_out) * ref
        if reverb:
            y = add_reverb(y, self.fs, rt60=0.45, wet=reverb, rng=self.rng)
        if normalize:
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.89
        return y


def add_reverb(y, fs, rt60=0.45, wet=0.25, rng=None):
    """Small-room reverb: convolution with exponentially decaying noise."""
    rng = rng or np.random.default_rng(0)
    n = int(rt60 * fs)
    t = np.arange(n) / fs
    ir = rng.standard_normal(n) * 10 ** (-3 * t / rt60)
    ir[: int(0.005 * fs)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    L = len(y) + n
    nfft = 1 << (L - 1).bit_length()
    r = np.fft.irfft(np.fft.rfft(y, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(y)]
    return y + wet * r * np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(r ** 2)) + 1e-12)
