"""
Evolution logging: JSONL operations, population snapshots, loss CSV.
"""

import json
import os
import csv
import gzip
import logging
from collections import defaultdict
from datetime import datetime
from time import time
from typing import List, DefaultDict

import numpy as np

from didgelab.app import get_app
from .genome import Genome
from .evolution import Nuevolution


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to Python int/float/list."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NuevolutionWriter:
    """
    Writes evolution logs: per-generation operations (JSONL), population snapshots, and optional loss CSV.

    Subscribes to app events (generation_ended, evolution_ended, log_evolution_operations).
    Output files go to the app output folder (evolution_operations.jsonl.gz, population.json.gz, losses.csv.gz).
    """

    def __init__(self,
                 interval=20,
                 write_loss=False,
                 write_population_interval=100,
                 log_operations=True):

        self.interval = interval
        self.writer = None
        self.format = None
        self._write_loss_csv = write_loss
        self.write_population_interval = write_population_interval
        self.log_operations = log_operations
        self.log_evolutions_file = None
        self.write_population_writer = None
        self.csvfile = None
        self.current_generation = None

        def generation_ended(i_generation, population):
            self.current_generation = i_generation

            if self._write_loss_csv:
                self.write_loss(i_generation, population)
            if self.write_population_interval > 0 and i_generation % self.write_population_interval == 0:
                msg = f"generation {i_generation} ended, writing population to file\n"
                msg += "loss:\n"
                losses = [f"{key}: {value}" for key, value in population[0].loss.items()]
                msg += "\n".join(losses)
                logging.info(msg)
                self.write_population(population, i_generation)

        if self.write_population_interval > 0 or self._write_loss_csv:
            get_app().subscribe("generation_ended", generation_ended)

        def evolution_ended(population):
            self.write_population(population)

            if self.csvfile is not None:
                self.csvfile.close()
            self.log_evolutions_file.close()
            self.writer = None
            self.write_population_writer.close()

        get_app().subscribe("evolution_ended", evolution_ended)
        get_app().subscribe("log_evolution_operations", self.log_evolution_operations)
        get_app().register_service(self)

    def log_evolution_operations(self, i_generation, individuals, mutation_operations, crossover_operations):
        """Append one line (JSON) to evolution_operations.jsonl.gz for this generation."""
        if self.log_evolutions_file is None:
            outfile = os.path.join(get_app().get_output_folder(), "evolution_operations.jsonl.gz")
            self.log_evolutions_file = gzip.open(outfile, "w")

        def flatten_dicts(list_of_dicts):
            flattened = defaultdict(list)
            for di in list_of_dicts:
                for key, value in di.items():
                    flattened[key].append(value)
            return flattened

        nuevo = get_app().get_service(Nuevolution)
        operator_probs = {}
        for i in range(len(nuevo.evolution_operators)):
            op = type(nuevo.evolution_operators[i]).__name__
            p = nuevo.evolution_operator_probs[i]
            operator_probs[op] = p

        data = {
            "genome_ids": [g.id for g in individuals],
            "genomes": [g.genome.tolist() for g in individuals],
            "losses": flatten_dicts([g.loss for g in individuals]),
            "mutations_operations": flatten_dicts(mutation_operations),
            "crossover_operations": flatten_dicts(crossover_operations),
            "date": datetime.now().isoformat(),
            "generation": i_generation,
            "evolution_operator_probs": operator_probs
        }

        self.log_evolutions_file.write(json.dumps(data).encode())
        self.log_evolutions_file.write("\n".encode())

    def write_population(self, population: List[Genome], generation=None):
        """Append top individuals (up to 20) to population.json.gz and update latest_population.json."""
        if generation is None:
            generation = self.current_generation

        if self.write_population_writer is None:
            outfile = os.path.join(get_app().get_output_folder(), "population.json.gz")
            self.write_population_writer = gzip.open(outfile, "w")

        data = []
        max_individuals = 20
        i = 0

        for p in population:
            i += 1
            if i > max_individuals:
                break

            data.append({
                "genome": list(p.genome),
                "loss": p.loss,
                "representation": p.representation()
            })

        data2 = {"generation": generation, "population": data}
        data2 = json.dumps(data2, cls=NumpyEncoder)
        self.write_population_writer.write(data2.encode())
        self.write_population_writer.write("\n".encode())

        outfile = os.path.join(get_app().get_output_folder(), "latest_population.json")
        with open(outfile, "w") as f:
            json.dump(data2, f)

    def write_loss(self, i_generation, population: List[Genome]):
        """Append one row per individual to losses.csv.gz (generation, step, time, genome, loss keys)."""
        if self.writer is None and self._write_loss_csv:
            outfile = os.path.join(get_app().get_output_folder(), "losses.csv.gz")
            self.csvfile = gzip.open(outfile, "a")
            self.writer = csv.writer(self.csvfile)

        if self.format is None:
            self.format = ["i_generation", "step", "time", "genome"]
            for key in population[0].loss.keys():
                self.format.append(key)
            self.writer.writerow(self.format)

        step = 0

        for i in range(len(population)):
            individual = population[i]
            row = [i_generation, step, time(), individual.genome]
            for key in self.format[len(row):]:
                row.append(individual.loss[key])
            self.writer.writerow(row)

def load_latest_evolution(shape, output_folder = "outputs"):
    files = os.listdir(output_folder)
    files = sorted(files)
    infile = os.path.join(output_folder, files[-1])
    population = []
    with gzip.open(infile, "rb") as f:
        x = f.read().decode("utf-8")
        x = json.loads(x)

        for i in range(len(x["population"])):
            ind = shape.clone()
            ind.genome = np.array(x["population"][i])
            ind.loss = {key: np.float32(value) for key, value in x["losses"][i].items()}
            population.append(ind)
        return population
