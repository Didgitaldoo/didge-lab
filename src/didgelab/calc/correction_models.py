from abc import ABC, abstractmethod
from joblib import load
import numpy as np
from didgelab.calc.geo import Geo

import didgelab.calc.fft
from didgelab.calc.sim.sim import *
from didgelab.calc.fft import *
from didgelab.app import get_app
from didgelab.initializer import init_console_no_output
from didgelab.calc.geo import Geo
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import json

class CorrectionModel(ABC):

    @abstractmethod
    def correct_frequency(self, freq : np.array):
        pass

class LinearRegressionCorrectionModel(CorrectionModel):

    def __init__(self):
        infile = "../assets/correction_models/2024-03-18-linear-regression.joblib"
        self.model = load(infile)

    def correct_frequency(self, freqs):
        freqs = np.log2(freqs).reshape(len(freqs), 1)
        correction = self.model.predict(freqs).reshape(len(freqs))
        correction = np.power(2, correction)
        return correction


# test method
# python -m didgelab.calc.correction_models
if __name__ == "__main__":

    def minmax(a):
        a -= a.min()
        a /= a.max()
        return a

    da_path = "/Users/jane03/workspaces/music/didge/didge-archive"
    didge_archive = json.load(open(os.path.join(da_path, "didge-archive.json")))
    didge_archive = list(filter(lambda x:x["shape"] == "straight", didge_archive))

    didge = didge_archive[2]
    infile = os.path.join(da_path, didge["audio-samples"]["neutral-sound"])
    fft_freq, fft = do_fft(infile)
    fft = np.log2(fft)
    fft = minmax(fft)
    fft = fft/fft.max()
    fft_smooth = savgol_filter(fft, 20, 3)

    geofile = os.path.join(da_path, didge["geometry"])
    geo = json.load(open(geofile))
    geo = Geo(geo)
    freqs = get_log_simulation_frequencies(1, 1000, 1)
    segments = create_segments(geo)
    impedance = compute_impedance(segments, freqs)
    ground_freqs, imp_ip = interpolate_spectrum(freqs, impedance)
    ground = compute_ground_spektrum(ground_freqs, imp_ip)
    ground /= ground.max()

    model = LinearRegressionCorrectionModel()
    corrected_freqs = model.correct_frequency(ground_freqs)

    plt.plot(fft_freq, fft_smooth, label="measured")
    plt.plot(ground_freqs, ground, label="computed")
    plt.plot(corrected_freqs, ground, label="corrected")
    plt.title(didge["name"])
    plt.xlabel("Frequenz")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
