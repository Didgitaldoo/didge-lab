import json
import sys
import os
import gzip 

from didgelab.calc.geo import Geo, geotools
import matplotlib.pyplot as plt

import didgelab.calc.fft
from didgelab.calc.sim.sim import *
from didgelab.calc.fft import *
from didgelab.calc.conv import *
from didgelab.util.didge_visualizer import vis_didge

# open population.json.gz file and return latest generation
def get_latest_population(infile):
    for line in gzip.open(infile):
        continue
    return json.loads(line)

# print various information about the population
def visualize_individuals(population, n=None, base_freq=440):

    if n is not None and n<len(population):
        population = population[0:n]

    for i in range(len(population)):
        geo = population[i]["representation"]["geo"]
        geo = Geo(geo)
        freqs = get_log_simulation_frequencies(1, 1000, 5)
        segments = create_segments(geo)
        impedance = compute_impedance(segments, freqs)
        notes = get_notes(freqs, impedance, base_freq=base_freq)
        print("************************")
        print(f"Individual {i}")
        print("************************")
        vis_didge(geo)
        plt.show()
        plt.plot(freqs, impedance)
        plt.title("Impedance spektrum")
        plt.show()
        print(notes.round(2))
        print(f"Impedance sum: {notes.impedance.sum():.2f}")
        print(f"Volume: {geo.compute_volume()/1000:.2f} cm3")
        print(f"Length: {geo.geo[-1][0]/10:.2f} cm")
        print(f"Bell diameter: {geo.geo[-1][1]/10:.2f} cm")
        print()
