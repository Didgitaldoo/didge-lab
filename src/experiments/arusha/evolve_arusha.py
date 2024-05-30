"""
python -m experiments.arusha.evolve_arusha
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

base_freq = 425



class MbeyaGemome(GeoGenome):

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

    def __init__(self, n_bubbles=1, add_bubble_prob=0.9):

        self.named_params = {}

        self.d1=32
        self.add_bubble_prob=add_bubble_prob
        self.n_bubbles=n_bubbles

        # straight part
        self.add_param("l_gerade", 500, 1500)
        self.add_param("d_gerade", 0.9, 1.2)

        # opening part
        self.add_param("n_opening_segments", 0, 8)
        self.add_param("opening_factor_x", -2, 2)
        self.add_param("opening_factor_y", -2, 2)
        self.add_param("opening_length", 700, 1000)

        # bell
        self.add_param("d_pre_bell", 5, 20)
        self.add_param("l_bell", 20, 50)
        self.add_param("bellsize", 5, 20)

        # bubble
        for i in range(self.n_bubbles):
            self.add_param(f"add_bubble_{i}", 0, 1)
            self.add_param(f"bubble_height_{i}", -0.5, 1)
            self.add_param(f"bubble_pos_{i}", 0, 1)
            self.add_param(f"bubble_width_{i}", 150, 300)

        GeoGenome.__init__(self, n_genes = len(self.named_params))

    def make_bubble(self, shape, pos, width, height):

        i=self.get_index(shape, pos-0.5*width)
        n_segments = 11

        bubbleshape=shape[0:i]

        x=pos-0.5*width
        y=Geo(geo=shape).diameter_at_x(x)

        if shape[i-1][0]<x:
            bubbleshape.append([x,y])

        for j in range(1, n_segments):
            x=pos-0.5*width + j*width/n_segments

            # get diameter at x
            y=Geo(geo=shape).diameter_at_x(x)
            factor=1+math.sin(j*math.pi/(n_segments))*height
            y*=factor

            bubbleshape.append([x,y])

        x=pos+0.5*width
        y=Geo(geo=shape).diameter_at_x(x)
        bubbleshape.append([x,y])

        while shape[i][0]<=bubbleshape[-1][0]+1:
            i+=1
        
        bubbleshape.extend(shape[i:])

        return bubbleshape

    # return last index that is smaller than x
    def get_index(self, shape, x):
        for i in range(len(shape)):
            if shape[i][0]>x:
                return i
        return len(shape)-1

    def genome2geo(self):
        shape=[[0, self.d1]]

        # straight part
        p=[self.get_value("l_gerade"), shape[-1][1]*self.get_value("d_gerade")]
        shape.append(p)

        # opening part
        n_seg=self.get_value("n_opening_segments")
        seg_x=[]
        seg_y=[]
        for i in range(int(n_seg)):
            x=pow(i+1, self.get_value("opening_factor_x"))
            y=pow(i+1, self.get_value("opening_factor_y"))
            seg_x.append(x)
            seg_y.append(y)

        def normalize(arr):
            m=sum(arr)
            return [x/m for x in arr]

        seg_x=normalize(seg_x)
        seg_y=normalize(seg_y)
        seg_x=[x*self.get_value("opening_length") for x in seg_x]
        seg_y=[y*self.get_value("d_pre_bell") for y in seg_y]

        start_x=shape[-1][0]
        start_y=shape[-1][1]
        for i in range(int(n_seg)):
            x=sum(seg_x[0:i+1]) + start_x
            y=sum(seg_y[0:i+1]) + start_y
            shape.append([x,y])

        p=[shape[-1][0] + self.get_value("l_bell"), shape[-1][1]+self.get_value("bellsize")]
        shape.append(p)

        # add bubble
        for i in range(self.n_bubbles):
            if self.get_value(f"add_bubble_{i}")<self.add_bubble_prob:
                pos=shape[-1][0]*self.get_value(f"bubble_pos_{i}")
                width=self.get_value(f"bubble_width_{i}")
                height=self.get_value(f"bubble_height_{i}")
                if pos-width/2<-10:
                    pos=width/2 + 10
                if pos+width/2+10>shape[-1][0]:
                    pos=shape[-1][0]-width/2 - 10
                shape=self.make_bubble(shape, pos, width, height)

        geo=Geo(shape)
        geo=geotools.fix_zero_length_segments(geo)
        return geo

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
        self.max_error = 10

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

        freqs = get_log_simulation_frequencies(1, 1000, self.max_error)
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
best_loss = 9999999999
last_loss_update = -1

def evolve():

    get_config()["log_folder_suffix"] = "nuevolution_test"
    loss = MbeyaLoss(n_notes=7)

    writer = NuevolutionWriter(write_population_interval=5)

    n_segments = 10

    evo = Nuevolution(
        loss, 
        MbeyaGemome(n_bubbles=3, add_bubble_prob=0.7),
        #generation_size = 5,
        #num_generations = 5,
        #population_size = 10,
        generation_size = 3,
        num_generations = 1000,
        population_size = 1000,
    )

    schedulers = [
        # LinearDecreasingCrossover(),
        LinearDecreasingMutation()
    ]

    def generation_ended(i_generation, population):
        global last_max_error, errors

        # update max error if necessary
        evo = get_app().get_service(Nuevolution)
        progress = evo.i_generation / evo.num_generations
        max_error = errors[int(progress*len(errors))]

        if last_max_error != max_error:
            evo.recompute_losses = True
            loss.max_error = max_error
            logging.info(f"set max_error to {max_error}")

        last_max_error = max_error

        # debug output
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

        # stop evolution if there is no progress
        global last_loss_update, best_loss
        thisloss = population[0].loss["total"]
        print(thisloss)
        if thisloss is None or thisloss < best_loss:
            best_loss = thisloss
            last_loss_update = i_generation
        elif i_generation-last_loss_update > 30:
            logging.info("stop evolution because it did not improve")
            evo.continue_evolution = False

    get_app().subscribe("generation_ended", generation_ended)

    pbar = NuevolutionProgressBar()
    population = evo.evolve()

if __name__ == "__main__":
    evolve()