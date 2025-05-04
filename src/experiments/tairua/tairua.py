import sys
sys.path.append('../../')

import numpy as np
import logging
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from didgelab.evo.nuevolution import Genome
from didgelab.calc.geo import Geo
from didgelab.calc.conv import cent_diff
from didgelab.evo.nuevolution import LossFunction
from didgelab.calc.sim.sim import get_notes, compute_impedance, create_segments, get_log_simulation_frequencies, quick_analysis
from didgelab.util.didge_visualizer import plot_geo_to_axs

from time import time
from didgelab.calc.conv import cent_diff, note_name, freq_to_note
from didgelab.evo.nuevolution import Nuevolution
from didgelab.app import get_app
import pandas as pd

class PointShape(Genome):

    def __init__(self, geo):
        self.num_segments = len(geo.geo)
        self.x_scaling = geo.geo[-1][0] * 1.1
        self.y_scaling = max([s[1] for s in geo.geo]) * 1.1

        genome = []
        for i in range(self.num_segments):
            segment = geo.geo[i]
            xval = segment[0]/self.x_scaling
            genome.append(xval)
            yval = segment[1]/self.y_scaling
            genome.append(yval)

        genome = np.array(genome)
        Genome.__init__(self, genome=genome)

    def genome2geo(self):
        geo = []
        for i in range(0, len(self.genome), 2):
            x = self.genome[i] * self.x_scaling
            y = self.genome[i+1] * self.y_scaling
            if y<=0:
                y = 1 

            if x not in [segment[0] for segment in geo]:
                geo.append([x,y])

        geo = sorted(geo, key=lambda x:x[0])
        for i in range(1, len(geo)):
            if geo[i-1][0] == geo[i][0]:
                geo[i][0]+=0.01
        geo[0][0] = 0
        return Geo(geo)
    
class TairuaLoss(LossFunction):

    def __init__(self, target_freqs, target_impedances, target_weights, max_error=5):
        self.target_freqs = target_freqs
        self.target_impedances = target_impedances
        self.target_weights = target_weights
        self.max_error = 5
        self.freqs = get_log_simulation_frequencies(1, 1000, max_error)

    def loss(self, shape):
        try:
            geo = shape.genome2geo()
            segments = create_segments(geo)
            impedances = compute_impedance(segments, self.freqs)
            peaks = find_peaks(impedances)

            peak_freqs = np.array([self.freqs[i] for i in peaks[0]])
            peak_impedances = [impedances[i] for i in peaks[0]]
            peak_impedances = np.array(peak_impedances) / impedances.max()

            losses = []
            for target_freq, target_impedance in zip(self.target_freqs, self.target_impedances):
                i = np.argmin(np.abs(peak_freqs - target_freq))
                peak_freq = peak_freqs[i]
                peak_impedance = peak_impedances[i]
                diff_freq = cent_diff(target_freq, peak_freq) / 600
                diff_impedance = peak_impedance - target_impedance
                loss = np.sqrt(diff_freq*diff_freq + diff_impedance*diff_impedance)
                losses.append(loss)

            losses = np.array(losses)
            losses = losses*losses
            losses *= self.target_weights

            return {
                "total": np.sum(losses),
                "individual_losses": losses
            }
        except Exception as e:
            print(e)
            logging.error(e)
            return {
                "total": 100000000,
                "individual_losses": np.array([-1]*len(self.target_freqs))
            }

def visualize_population(population, n_individuals, target_freqs, target_impedances, loss_function):
    fix, axs = plt.subplots(nrows=n_individuals, ncols=3, figsize=(15,n_individuals*3))
    for i in range(n_individuals):

        if n_individuals == 1:
            row = axs
        else:
            row = axs[i]

        ax = row[0]
        geo = population[i].genome2geo()
        plot_geo_to_axs(geo, ax)
        ax.set_title(f"individual {i+1}")

        ax = row[1]
        analysis = quick_analysis(geo)

        ax.scatter(analysis["notes"]["freq"], analysis["notes"]["rel_imp"], label="individual")
        ax.scatter(target_freqs, target_impedances, label="target")
        ax.legend()
        
        ax = row[2]
        ax.text(0, 0.3, analysis["notes"][["note_name", "freq", "cent_diff"]].round(2).to_string(index=False))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

class TairuaMonitor:
    """ Generate log messages to monitor an evolution that aims at target peaks in the resonance spektrum.
        This is an example log message
        generation 1/100, best loss=1.19, time per generation: 47.78s, remaining_time = 78.84 minutes
        target_note   F1  F#2   C3   G3   C4  D#4   F4  A#5
        cent_diff   -5  122  -10   -5  -10  -15   15   45
        losses 0.00 0.65 0.30 0.18 0.03 0.01 0.01 0.01
    """

    def __init__(self, 
        loss : LossFunction, 
        evo : Nuevolution, 
        target_freqs : np.array, 
        target_impedances : np.array,
        log_interval : int = 10,
        log_before_generation : int = 3):
        """_summary_

        Args:
            loss (LossFunction): The loss function of the evolution
            evo (Nuevolution): The nuevolution object
            target_freqs (np.array): The target frequencies
            target_impedances (np.array): The target impedances
            log_interval (int, optional): The system will log every log_interval generations. Defaults to 10.
            log_before_generation (int, optional): For the first log_before_generation generations, the monitor will log every generation. Defaults to 3.
        """

        self.generation_durations = []
        self.start_time : int = time()
        self.loss : LossFunction = loss 
        self.evo : Nuevolution = evo
        self.target_freqs : np.array = target_freqs
        self.target_impedances : np.array = target_impedances
        self.log_interval : int = log_interval
        self.log_before_generation : int = log_before_generation

        # reset duration counters when the first generation started
        def generation_started(i_generation, population):
            if i_generation == 0:
                self.generation_durations = []
                self.start_time = time()
        get_app().subscribe("generation_started", generation_started)

        def generation_ended(i_generation, population):
            if i_generation < 3 or i_generation%10 == 0:

                # acoustic simulation and loss computation of the best individual
                geo = population[0].genome2geo()
                analysis = quick_analysis(geo)
                l = loss.loss(population[0])

                # compute the time per generation and time left
                duration = time() - self.start_time
                self.generation_durations.append(duration)
                mean_duration = np.mean(self.generation_durations)
                remaining_generations = evo.num_generations - i_generation
                remaining_time = remaining_generations * mean_duration / 60
                self.start_time = time()
                
                print(f"generation {i_generation}/{self.evo.num_generations}, best loss={l['total']:.2f}, time per generation: {mean_duration:.2f}s, remaining_time = {remaining_time:.2f} minutes")

                # visualize difference from the target
                notes = analysis["notes"]
                row1 = ["target_note"]
                row2 = ["freq_diff"]
                row3 = ["imp_diff"]

                row4 = ["losses"]
                for i in range(len(self.target_freqs)):
                    target_freq = target_freqs[i]

                    name = note_name(freq_to_note(target_freq)) # compute the name of the note
                    row1.append(name)
                    j = np.argmin(np.abs(np.log2(notes.freq)-np.log2(target_freq))) # find the clostest resonant peak of the acoustic simulation to that target peak
                    individual_freq = notes.freq[j]
                    row2.append(round(cent_diff(target_freq, individual_freq))) # compute the difference in cent between the target peak and the closest peak of the acoustic simulation

                    row3.append(np.abs(self.target_impedances[i]-notes.rel_imp[j]).round(2))
                    individual_imp = notes.rel_imp[i]

                    row4.append(f'{l["individual_losses"][i]:.2f}')
                df = pd.DataFrame([row1, row2, row3, row4])
                print(df.to_string(index=False, header=False))
            
        get_app().subscribe("generation_ended", generation_ended)
