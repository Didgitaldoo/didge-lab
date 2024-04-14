import os
import csv
from time import time

from didgelab.app import get_app, get_config
from didgelab.evo.shapes import Shape
from didgelab.evo.evolution import MultiEvolution

class LossWriter:
    
    def __init__(self, interval=100, write_loss=False):

        self.interval = interval
        self.writer = None
        self.format = None
        self.write_loss = write_loss

        def write_loss(i_generation, population):
            self.write_loss(i_generation, population)

        get_app().subscribe("generation_ended", write_loss)

        def close(population):
            self.csvfile.close()
            self.writer = None

        get_app().subscribe("evolution_ended", close)

        get_app().register_service(self)
        
    def write_loss(self, i_generation, population : list[Shape]):

        if self.writer is None and self.write_loss:
            outfile = os.path.join(get_app().get_output_folder(), "losses.csv")
            self.csvfile = open(outfile, "a")
            self.writer = csv.writer(self.csvfile)

        if self.format is None:
            self.format = ["i_generation", "i_mutant", "step", "time"]
            for key in population[0].loss.keys():
                self.format.append(key)
            self.writer.writerow(self.format)

        step = 0
        if "is_multi_evolution" in get_config() and get_config()["is_multi_evolution"] is True:
            step = get_app().get_service(MultiEvolution).evolution_nr

        for i in range(len(population)):
            individual = population[i]
            row = [i_generation, i, step, time()]
            for key in self.format[len(row):]:
                row.append(individual.loss[key])
            self.writer.writerow(row)
