"""
python -m experiments.spline_shape.evolve_splines
"""

from didgelab.calc.conv import note_to_freq
from didgelab.evo.loss import LossFunction

from didgelab.evo.evolution import MultiEvolution
#from didgelab.initializer import init_console
from didgelab.app import get_config, get_app

from didgelab.calc.sim.sim import compute_impedance_iteratively, get_notes, compute_impedance, create_segments, get_log_simulation_frequencies
from didgelab.calc.geo import Geo, geotools

from didgelab.evo.nuevolution import Genome, LossFunction, Nuevolution
from didgelab.evo.nuevolution import GeoGenomeA, NuevolutionWriter, GeoGenome
from didgelab.evo.nuevolution import NuevolutionProgressBar, LinearDecreasingCrossover,LinearDecreasingMutation

import math
import numpy as np

import logging

from scipy.interpolate import CubicSpline


class SplineShape(GeoGenome):

    def __init__(
        self,
        init_resolution = 5, 
        d0=32, 
        d_bell_min=50, 
        d_bell_max=80,
        max_length = 1900, 
        min_length = 1500):

        self.max_length = max_length
        self.min_length = min_length
        self.resolution = init_resolution
        self.init_resolution = init_resolution
        self.d0 = 32
        self.d_bell_min = d_bell_min
        self.d_bell_max = d_bell_max

        self.geo_offset = 3
        GeoGenome.__init__(self, n_genes = 3+2*init_resolution)

    def get_properties(self):
        length = self.genome[0] * (self.max_length-self.min_length) + self.min_length
        bell_size = self.genome[1] * (self.d_bell_max - self.d_bell_min) + self.d_bell_min
        power = self.genome[2]*2

        x_genome = np.array([self.genome[i] for i in range(self.geo_offset, len(self.genome), 2)])
        y_genome = np.array([self.genome[i] for i in range(self.geo_offset+1, len(self.genome), 2)])

        return length, bell_size, power, x_genome, y_genome

    def genome2geo(self):
        length, bell_size, power, x_genome, y_genome = self.get_properties()
        x, y, normalization = self.make_shape(x_genome, y_genome, length, self.d0, bell_size, power)
        x, y = self.fix_didge(x, y, self.d0, bell_size)
        geo = list(zip(x,y))
        return Geo(list(zip(x,y)))
    
    def double_resolution(self):
        length, bell_size, power, x_genome, y_genome = self.get_properties()
        x1, y1, normalization = self.make_shape(x_genome, y_genome, length, self.d0, bell_size, power)
        x2, y2 = self.smooth(x1, y1)
        x3, y3 = self.backward(x2, y2, x_genome, y_genome, normalization, length, self.d0, bell_size, power)

        genome = self.genome[0:self.geo_offset].tolist()
        for i in range(len(x3)):
            genome.append(x3[i])
            genome.append(y3[i])
        self.genome = np.array(genome)

    def smooth(self, x1,y1):
        cs = CubicSpline(x1, y1)

        x2 = [0]
        for i in range(1, len(x1)):
            x2.append(0.5*(x1[i]+x1[i-1]))
            x2.append(x1[i])
        y2 = cs(x2)

        x2 = np.array(x2)
        return x2,y2

    def basic_y_shape(self, n, bellsize, power, d0):
        y= np.arange(n+1)/n
        y = np.power(y, power)
        y = np.power(y, power)
        y = np.power(y, power)
        y = d0 + y*(bellsize - d0)
        return y

    def make_shape(self, x_genome, y_genome, length, d0, bellsize, power):
        x_genome = x_genome.copy()

        x = [0]
        x_genome += 0.3
        for i in range(len(x_genome)):
            x.append(x[-1] + x_genome[i])

        x = np.array(x)    

        normalization = x[-1]
        x /= x[-1]
        x *= length

        y = y_genome.copy()
        y -= 0.5
        y *= 120
        y = np.concatenate(([0], y))
        y_basic = self.basic_y_shape(len(y)-1, bellsize, power, d0)
        y = y_basic + y
        return x,y, normalization

    def fix_didge(self, x,y, d0, bellsize):
        mind = d0*0.9
        x=x.copy()
        y=y.copy()
        y[y<mind] = mind
        maxd = bellsize
        y[y>maxd] = maxd#
        return x,y

    def backward(self, x_smooth, y_smooth, x_genome, y_genome, normalization, length, d0, bellsize, power):
        odd = lambda x: np.array([x[i] for i in range(0, len(x), 2)])
        x_new = x_smooth/length
        x_new *= normalization

        x2 = []
        for i in range(len(x_new)-1):
            x2.append(x_new[i+1]-x_new[i])
        x_new = np.array(x2)
        x_new -= 0.3

        basic_y = self.basic_y_shape(len(y_smooth)-1, bellsize, power, d0)

        y_new = y_smooth - basic_y
        y_new = y_new[1:]
        y_new /= 120
        y_new += 0.5
        return x_new, y_new

base_freq = 425

# a loss that deviates 
def single_note_loss(note, peaks, i_note=0, filter_rel_imp=0.1):
    peaks=peaks[peaks.rel_imp>filter_rel_imp]
    if len(peaks)<=i_note:
        return 1000000
    f_target=note_to_freq(note, base_freq=base_freq)
    f_fundamental=peaks.iloc[i_note]["freq"]
    return np.sqrt(abs(math.log(f_target, 2)-math.log(f_fundamental, 2)))

# add loss if the didge gets smaller
def diameter_loss(geo):

    if type(geo)==Geo:
        shape=geo.geo
    elif type(geo) == list:
        shape=geo
    else:
        raise Exception("unknown type " + str(type(geo)))

    loss=0
    for i in range(1, len(shape)):
        delta_y=shape[i-1][1]-shape[i][1]
        if delta_y < 0:
            loss+=-1*delta_y

    loss*=0.005
    return loss



class MbeyaLoss(LossFunction):

    # fundamental: note number of the fundamental
    # add_octave: the first toot is one octave above the fundamental
    # scale: define the scale of the toots of the didgeridoo as semitones relative from the fundamental
    # target_peaks: define the target peaks as list of math.log(frequency, 2). overrides scale 
    # n_notes: set > 0 to determine the number of impedance peaks (above fundamental and add_octave)
    # weights: override the default weights
    # {
    #     "tuning_loss": 8,
    #     "volume_loss": 0.5,
    #     "octave_loss": 4,
    #     "n_note_loss": 5,
    #     "diameter_loss": 0.1,
    #     "fundamental_loss": 8,
    # }
    def __init__(self, fundamental=-31, add_octave=True, n_notes=-1, scale=[0,2,3,5,7,9,10], target_peaks=None, weights={}):
        LossFunction.__init__(self)

        self.weights={
            "tuning_loss": 8,
            "volume_loss": 12,
            "octave_loss": 4,
            "n_note_loss": 5,
            "diameter_loss": 0.1,
            "fundamental_loss": 16,
        }
        for key, value in weights.items():
            if key not in self.weights:
                raise Exception(f"Unknown weight {key}")
            self.weights[key]=value

        self.scale=scale
        self.fundamental=fundamental
        self.add_octave=add_octave
        self.n_notes=n_notes

        if target_peaks is not None:
            self.target_peaks=target_peaks
        else:
            self.scale_note_numbers=[]
            for i in range(len(self.scale)):
                self.scale_note_numbers.append(self.scale[i]+self.fundamental)

            n_octaves=10
            self.target_peaks=[]
            for note_number in self.scale_note_numbers:
                for i in range(0, n_octaves):
                    transposed_note=note_number+12*i
                    freq=note_to_freq(transposed_note, base_freq=base_freq)
                    freq=math.log(freq, 2)
                    self.target_peaks.append(freq)

    def loss(self, genome, context=None):

        # evolution_nr = get_app().get_service(MultiEvolution).evolution_nr

        geo = genome.genome2geo()

        evo = get_app().get_service(Nuevolution)
        progress = evo.i_generation / evo.num_generations

        max_error = 20 - int(17*progress)

        freqs = get_log_simulation_frequencies(1, 1000, max_error)
        segments = create_segments(geo)
        impedances = compute_impedance(segments, freqs)
        notes = get_notes(freqs, impedances, base_freq=base_freq)

        fundamental=single_note_loss(-31, notes)*self.weights["fundamental_loss"]
        octave=single_note_loss(-19, notes, i_note=1)*self.weights["octave_loss"]

        #notes=geo.get_cadsd().get_notes()
        tuning_loss=0
        volume_loss=0

        start_index=1
        if self.add_octave:
            start_index+=1
        if len(notes)>start_index:
            for ix, note in notes[start_index:].iterrows():
                f1=math.log(note["freq"],2)
                closest_target_index=np.argmin([abs(x-f1) for x in self.target_peaks])
                f2=self.target_peaks[closest_target_index]
                tuning_loss += math.sqrt(abs(f1-f2))
                imp = note["rel_imp"] / notes.impedance.max()
                volume_loss += imp

        tuning_loss*=self.weights["tuning_loss"]
        volume_loss*=self.weights["volume_loss"]
        
        n_notes=self.n_notes+1
        if self.add_octave:
            n_notes+=1
        n_note_loss=max(n_notes-len(notes), 0)*self.weights["n_note_loss"]

        d_loss = diameter_loss(geo)*self.weights["diameter_loss"]

        loss={
            "tuning_loss": tuning_loss,
            "volume_loss": volume_loss,
            "n_note_loss": n_note_loss,
            "diameter_loss": d_loss,
            "fundamental_loss": fundamental,
            "octave_loss": octave,
        }
        loss["total"]=sum(loss.values())
        return loss

errors = [20, 15, 10, 3]
last_max_error = errors[0]


def evolve():

    get_config()["log_folder_suffix"] = "nuevolution_test"
    loss = MbeyaLoss(n_notes=7)

    writer = NuevolutionWriter(write_population_interval=5)

    n_segments = 10

    evo = Nuevolution(
        loss, 
        SplineShape(init_resolution=12),
        #generation_size = 5,
        #num_generations = 5,
        #population_size = 10,
        generation_size = 500,
        num_generations = 1000,
        population_size = 1000,
    )

    schedulers = [
        # LinearDecreasingCrossover(),
        LinearDecreasingMutation()
    ]

    def generation_ended(i_generation, population):
        # update max error if necessary
        global last_max_error, errors

        evo = get_app().get_service(Nuevolution)
        progress = evo.i_generation / evo.num_generations
        max_error = errors[int(progress*len(errors))]

        if last_max_error != max_error:
            evo.recompute_losses = True

            # also double the resolution
            for i in range(len(population)):
                population[i].double_resolution()

        last_max_error = max_error

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