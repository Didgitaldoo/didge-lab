"""
python -m experiments.tamaki.evolve_tamaki3
"""

from didgelab.calc.conv import note_to_freq

from didgelab.evo.evolution import MultiEvolution
from didgelab.app import get_config, get_app

from didgelab.calc.sim.sim import compute_impedance_iteratively, get_notes, compute_impedance, create_segments, get_log_simulation_frequencies
from didgelab.calc.geo import Geo, geotools

from didgelab.evo.nuevolution import Genome, LossFunction, Nuevolution
from didgelab.evo.nuevolution import GeoGenomeA, NuevolutionWriter, GeoGenome
from didgelab.evo.nuevolution import NuevolutionProgressBar, LinearDecreasingCrossover,LinearDecreasingMutation

import math
import numpy as np
import json
import pandas as pd
import logging

base_freq = 440


class Tamaki3Loss(LossFunction):

    def __init__(self):
        self.reference = '{"impedance": {"0": 1.0, "1": 0.8576649663048393, "2": 0.6066206728881713, "3": 0.8420721962421757, "4": 0.8373902860195547, "5": 0.8084926830620255, "6": 0.7780097302125277, "7": 0.6681873008986587, "8": 0.6864601376736074, "9": 0.8340840526459629, "10": 0.44992931052124463}, "freq": {"0": 91.91400470014798, "1": 184.52432761772133, "2": 276.4383323178693, "3": 368.3523370180173, "4": 440.2663417181653, "5": 572.1803464183132, "6": 644.7906693358866, "7": 736.7046740360346, "8": 827.9223605187572, "9": 921.2290016537559, "10": 999.2166420053966}, "harmonic_series": {"0": 91.91400470014798, "1": 183.82800940029597, "2": 275.74201410044395, "3": 367.65601880059194, "4": 459.5700235007399, "5": 551.4840282008879, "6": 643.3980329010359, "7": 735.3120376011839, "8": 827.2260423013319, "9": 919.1400470014798, "10": 1011.0540517016278}, "diff": {"0": 0.0, "1": -0.6963182174253575, "2": -0.6963182174253575, "3": -0.6963182174253575, "4": 19.303681782574643, "5": -20.6963182174253, "6": -1.392636434850715, "7": -1.392636434850715, "8": -0.6963182174253006, "9": -2.0889546522760156, "10": 11.837409696231248}}'
        self.target_peaks = [4,5]
        self.reference = pd.DataFrame(json.loads(self.reference))        
        self.reference["impedance_normalized"] = self.reference.impedance / self.reference.impedance.max()
        self.reference["logfreq"] = np.log2(self.reference.freq)
        
    def loss(self, genome, context=None):
        freqs = get_log_simulation_frequencies(1, 1000, 5)
        segments = create_segments(genome.genome2geo())
        impedances = compute_impedance(segments, freqs)
        peaks = get_notes(freqs, impedances)

        peaks["logfreq"] = np.log2(peaks.freq)
        peaks["impedance_normalized"] = peaks.impedance / peaks.impedance.max()
        
        tuning_loss = []
        imp_loss = []
        wobble_freq_loss = []
        wobble_vol_loss = []

        # i_harmonic = 1

        # difference to target peaks loss
        base_freq = peaks.freq.iloc[0]
        for ix, peak in peaks.iterrows():
            mini = np.argmin([np.abs(peak.logfreq-f) for f in self.reference.logfreq])
            
            tl = np.abs(peak.logfreq-self.reference.logfreq[mini])
            il = np.abs(peak.impedance_normalized - self.reference.impedance_normalized[mini])
            tuning_loss.append(tl)
            imp_loss.append(il)
        
        # wobble freq losses
        for target_peak in self.target_peaks:
            f_target = self.reference.logfreq[target_peak]
            closest_peak_i = np.argmin(np.abs(f_target-peaks.logfreq))
            
            tuning_diff = np.abs(peaks.logfreq[closest_peak_i]-f_target)
            wobble_freq_loss.append(tuning_diff)

            imp_target = self.reference.impedance[target_peak]
            imp_diff = np.abs(peak.impedance - imp_target)
            wobble_vol_loss.append(imp_diff)

        fundamental_loss = tuning_loss[0]*10
        tuning_loss = np.sum(tuning_loss) / len(tuning_loss)
        imp_loss = np.sum(imp_loss) / len(imp_loss)
        wobble_freq_loss = np.sum(wobble_freq_loss) / len(wobble_freq_loss)
        wobble_vol_loss = np.sum(wobble_vol_loss) / len(wobble_vol_loss)

        losses =  {
            "fundamental_loss": 10*fundamental_loss,
            "tuning_loss": 10*tuning_loss,
            "imp_loss": 3*imp_loss,
            "wobble_freq_loss": 50*wobble_freq_loss,
            "wobble_vol_loss": 20*wobble_vol_loss
        }

        losses["total"] = np.sum(list(losses.values()))
        return losses
    
class BlackEucaShape(GeoGenome):
    
    def add_param(self, name, minval, maxval):
        self.named_params[name] = {
            "index": len(self.named_params),
            "min": minval,
            "max": maxval
        }

    def get_value(self, name):
        p = self.named_params[name]
        v = self.genome[p["index"]]
        v = v*(p["max"]-p["min"]) + p["min"]
        return v

    def __init__(self):
        
        self.named_params = {}

        self.d1=32
        self.n_segments = 15
        
        self.add_param("length", 1520, 1600)
        self.add_param("bellsize", 65, 80)
        self.add_param("power", 1,2)
        
        for i in range(self.n_segments-1):
            self.add_param(f"delta_x{i}", -20, 20)
            self.add_param(f"delta_y{i}", 0.8, 1.2)
        
        GeoGenome.__init__(self, n_genes = len(self.named_params))

    def genome2geo(self):
        length = self.get_value("length")
        bellsize = self.get_value("bellsize")

        x = length*np.arange(self.n_segments+1)/self.n_segments
    
        y= np.arange(self.n_segments+1)/self.n_segments
        p = self.get_value("power")
        y = np.power(y, p)
        y = np.power(y, p)
        y = np.power(y, p)
        y = self.d1 + y*(bellsize - self.d1)
        
        for i in range(1, self.n_segments-1):
            delta_x = self.get_value(f"delta_x{i}")
            delta_y = self.get_value(f"delta_y{i}")
            y[i] *= delta_y
            x[i] += delta_x
            x = sorted(x)
            
        geo = list(zip(x,y))
        
        return Geo(geo)

def evolve():

    get_config()["log_folder_suffix"] = "nuevolution_test"
    loss = Tamaki3Loss()

    writer = NuevolutionWriter(write_population_interval=20)

    n_segments = 10

    evo = Nuevolution(
        loss, 
        BlackEucaShape(),
        # generation_size = 5,
        # num_generations = 5,
        # population_size = 10,
        generation_size = 500,
        num_generations = 600,
        population_size = 1000,
    )

    schedulers = [
        # LinearDecreasingCrossover(),
        LinearDecreasingMutation()
    ]


    def generation_ended(i_generation, population):
        genome = population[0]
        losses = [f"{key}: {value}" for key, value in genome.loss.items()]
        msg = "\n".join(losses)
        
        geo = genome.genome2geo()
        freqs = get_log_simulation_frequencies(1, 1000, 5)
        segments = create_segments(geo)
        impedances = compute_impedance(segments, freqs)
        notes = get_notes(freqs, impedances, base_freq=base_freq).to_string()
        msg += "\n" + notes
        logging.info(msg)

    get_app().subscribe("generation_ended", generation_ended)

    pbar = NuevolutionProgressBar()
    population = evo.evolve() 

if __name__ == "__main__":
    evolve()