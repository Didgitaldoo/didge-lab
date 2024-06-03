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

from didgelab.evo.nuevolution import *

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

        self.min_d = 24
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

        # enforce minimum diameter
        for i in range(len(shape)):
            shape[i][1] = np.max((self.min_d, shape[i][1]))

        geo=Geo(shape)
        geo=geotools.fix_zero_length_segments(geo)
        return geo


class TagangaLoss(LossFunction):

    def __init__(self, fundamental=-31, add_octave=True, n_notes=-1, scale=[0,2,3,5,7,9,10], target_peaks=None, weights={}):
        LossFunction.__init__(self)

        self.weights={
            "tuning_loss": 8,
            "volume_loss": 36,
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

    def loss(self, genome):

