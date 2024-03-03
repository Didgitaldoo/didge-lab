"""
python -m didgelab.evo.nuevolution
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List
import multiprocessing
from tqdm import tqdm
from didgelab.app import get_app, get_config
import csv
import os
from time import time
import json
from didgelab.calc.geo import Geo
from didgelab.calc.sim.sim import get_log_simulation_frequencies, create_segments, compute_impedance, get_notes
from copy import deepcopy
import sys
import threading


class Genome(ABC):

    genome_id_lock = threading.Lock()
    genome_id_counter = 0

    def __init__(self, n_genes=None, genome=None):
        assert n_genes is not None or genome is not None

        if n_genes is not None:
            self.genome = np.random.sample(size=n_genes)
        elif genome is not None:
            self.genome = genome
        self.loss = None

        self.id = Genome.generate_id()

    def representation(self):
        return self.genome.tolist()
    
    def randomize_genome(self):
        self.genome = np.random.sample(size=len(self.genome))

    # static method to generate a genomes id 
    def generate_id():
        Genome.genome_id_lock.acquire()
        try:
            id = Genome.genome_id_counter
            Genome.genome_id_counter += 1
        finally:
            Genome.genome_id_lock.release()
        return id

class GeoGenome(Genome):

    def representation(self):
        geo = self.genome2geo()
        freqs = get_log_simulation_frequencies(1, 1000, 1)
        segments = create_segments(geo)
        impedance = compute_impedance(segments, freqs)
        notes = get_notes(freqs, impedance).to_string().split("\n")
        return {
            "geo": geo.geo,
            "analysis": notes
        }

    @abstractmethod
    def genome2geo(self) -> Geo:
        pass


class GeoGenomeA(GeoGenome):

    def build(n_segments):
        return GeoGenomeA(n_genes=(n_segments*2)+1)

    def genome2geo(genome : Genome) -> Geo:

        d0 = 32
        x = [0]
        y = [d0]
        min_l = 1000
        max_l = 2000

        d_factor = 75
        min_d = 25

        l = genome.genome[0] * (max_l-min_l) + min_l 
        i=1
        while i+2 < len(genome.genome):
            x.append(genome.genome[i] + x[-1])
            y.append(genome.genome[i+1])
            i += 2

        x = np.array(x)
        x /= x[-1]

        x = x * l
        x[0] = 0
        y = np.array(y) * d_factor + min_d
        y[0] = d0

        geo = list(zip(x,y))

        return Geo(geo)


class LossFunction(ABC):

    @abstractmethod
    def loss(self, shape : Genome):
        pass

class MutationOperator(ABC):

    @abstractmethod
    def apply(self, genome : Genome) -> Genome:
        pass

class CrossoverOperator(ABC):

    @abstractmethod
    def apply(self, genome1 : Genome, genome2 : Genome) -> Genome:
        pass

class SimpleMutation(MutationOperator):

    def __init__(self):

        self.mutation_prob=0.1
        self.mutation_rate=0.1

    def apply(self, genome : Genome):
        mutation = np.random.uniform(low=-self.mutation_rate, high=self.mutation_rate, size=len(genome.genome))
        mutation *= (np.random.sample(size=len(mutation))<self.mutation_prob).astype(int)
        mutation = genome.genome + mutation
        mutation[mutation<0] = 0
        mutation[mutation>0] = 1
        return type(genome)(genome=mutation)
    
class RandomMutation(MutationOperator):

    def apply(self, genome : Genome):
        return type(genome)(n_genes=len(genome.genome))
    
class RandomCrossover(CrossoverOperator):

    def apply(self, parent1 : Genome, parent2 : Genome) -> Genome:
        assert type(parent1) == type(parent2)
        new_genome = list(zip(parent1.genome, parent2.genome))
        new_genome = np.array([np.random.choice(x) for x in new_genome])
        return type(parent1)(genome=new_genome)
    
class AverageCrossover(CrossoverOperator):

    def apply(self, parent1 : Genome, parent2 : Genome) -> Genome:
        assert type(parent1) == type(parent2)
        new_genome = (parent1.genome + parent2.genome) / 2
        return type(parent1)(genome=new_genome)

class NuevolutionWriter:
    
    def __init__(self, interval=100):

        self.interval = interval
        self.writer = None
        self.format = None

        def write_loss(i_generation, population):
            self.write_loss(i_generation, population)

        get_app().subscribe("generation_ended", write_loss)

        def evolution_ended(population):
            self.csvfile.close()
            self.evolution_operations_f.close()
            self.writer = None
            self.evolution_operations_stream = None
            self.write_population(population)

        self.evolution_operations_stream = None
        self.evolution_operations_format = None

        get_app().subscribe("evolution_ended", evolution_ended)
        get_app().subscribe("log_evolution_operations", self.log_evolution_operations)
        get_app().register_service(self)
        
    def log_evolution_operations(self, i_generation, new_genome_ids, mutation_parent, crossover_parent1, crossover_parent2, operations, losses_before, losses_after):

        if self.evolution_operations_stream is None:
            outfile = os.path.join(get_app().get_output_folder(), "evolution_operations.csv")
            self.evolution_operations_f = open(outfile, "w")
            self.evolution_operations_stream = csv.writer(self.evolution_operations_f)
            
            self.evolution_operations_format = ["generation", "new_genome_id", "mutation_parent", "crossover_parent_1", "crossover_parent_2", "mutation", "crossover"]
            for key in losses_before[0].keys():
                self.evolution_operations_format.append("loss_before_" + key)
            for key in losses_after[0].keys():
                self.evolution_operations_format.append("loss_after_" + key)

            self.evolution_operations_stream.writerow(self.evolution_operations_format)

        for i in range(len(operations)):
            if len(operations[i]) == 0:
                continue

            row = [
                i_generation, 
                new_genome_ids[i], 
                mutation_parent[i], 
                crossover_parent1[i], 
                crossover_parent2[i], 
                operations[i][0].__name__]

            if len(operations[i]) > 1:
                row.append(operations[i][1].__name__)
            else:
                row.append(None)

            for key, value in losses_before[i].items():
                row.append(value)
            for key, value in losses_after[i].items():
                row.append(value)

            self.evolution_operations_stream.writerow(row)

    def write_population(self, population : List[Genome]):
        outfile = os.path.join(get_app().get_output_folder(), "population.json")
        f = open(outfile, "w")
        data = []
        max_individuals = 20
        i=0

        for p in population:
            i+=1
            if i>max_individuals:
                break

            data.append({
                "genome": list(p.genome),
                "loss": p.loss,
                "representation": p.representation()
            })

        f.write(json.dumps(data, indent=4))
        f.close()

    def write_loss(self, i_generation, population : list[Genome]):

        if self.writer is None:
            outfile = os.path.join(get_app().get_output_folder(), "losses.csv")
            self.csvfile = open(outfile, "a")
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

class Nuevolution():

    def __init__( self,
        loss : LossFunction,
        father_genome : Genome,
        generation_size = 5,
        num_generations = 10,
        population_size = 10,
        mutation_prob = 0.5,
        crossover_prob = 0.5,
        crossover_operators = [RandomCrossover(), AverageCrossover()],
        mutation_operators = [SimpleMutation(), RandomMutation()]):

        self.loss = loss
        self.father_genome = father_genome
        self.generation_size = generation_size
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.crossover_operators = crossover_operators
        self.mutation_operators = mutation_operators

        self.i_generation = -1
        self.population = None

    def evolve(self):

        # initialize
        pool = multiprocessing.Pool()

        self.population = []
        for i in range(self.generation_size):
            mutant = deepcopy(self.father_genome)
            mutant.randomize_genome()
            self.population.append(mutant)

        losses = pool.map(self.loss.loss, self.population)
        for i in range(len(losses)):
            self.population[i].loss = losses[i]

        self.population = sorted(self.population, key=lambda x:x.loss["total"])

        probs = []

        # evolve
        pbar = tqdm(total=self.num_generations)
        for i_generation in range(self.num_generations):
            
            if len(probs) != len(self.population):
                # compute probabilities that an individual will be selected
                probs = (1+np.arange(len(self.population))) / len(self.population)
                probs = np.exp(probs)
                probs = np.exp(probs)
                probs = np.flip(probs)
                probs /= probs.sum()
                #print(len(probs))

            self.i_generation = i_generation

            losses = np.array([p.loss for p in self.population])

            #print(len(self.population), self.generation_size, len(probs))

            indizes = np.random.choice(np.arange(len(self.population)), size=self.generation_size, replace=False, p=probs)
            generation = [self.population[i] for i in indizes]
            losses_before = [p.loss for p in generation]

            i_mutants = np.arange(self.generation_size)[np.random.sample(self.generation_size)<self.mutation_prob]
            operations = []

            mutation_parent = []
            crossover_parent1 = []
            crossover_parent2 = []

            for i in range(self.generation_size):
                operations.append([])
                mutation_parent.append(None)
                crossover_parent1.append(None)
                crossover_parent2.append(None)

            for i in i_mutants:
                if len(self.mutation_operators) == 1:
                    operator = self.mutation_operators[0]
                else:
                    operator = np.random.choice(self.mutation_operators)

                operations[i].append(type(operator))
                mutation_parent[i] = generation[i].id
                generation[i] = operator.apply(generation[i])

            # crossover
            i_crossover = np.arange(self.generation_size)[np.random.sample(self.generation_size)<self.crossover_prob]
            for parent1 in i_crossover:
                parent2 = parent1
                while parent1 == parent2:
                    parent2 = np.random.choice(np.arange(len(self.population)), p=probs)

                if len(self.crossover_operators) == 1:
                    operator = self.crossover_operators[0]
                else:
                    operator = np.random.choice(self.crossover_operators)

                crossover_parent1[i] = self.population[parent1].id
                crossover_parent2[i] = self.population[parent2].id
                operations[parent1].append(type(operator))
                generation[parent1] = operator.apply(self.population[parent1], self.population[parent2])
                
            # add only changed genes to population
            i_changed = np.arange(self.generation_size)
            i_changed = i_changed[[i in i_mutants or i in i_crossover for i in i_changed]]

            # collect logging data
            generation = [generation[i] for i in i_changed]
            operations = [operations[i] for i in i_changed]
            crossover_parent1 = [crossover_parent1[i] for i in i_changed]
            crossover_parent2 = [crossover_parent2[i] for i in i_changed]
            new_genome_ids = [g.id for g in generation]
            mutation_parent = [mutation_parent[i] for i in i_changed]
            losses_before = [losses_before[i] for i in i_changed]

            # compute loss
            losses = pool.map(self.loss.loss, generation)

            get_app().publish("log_evolution_operations", (self.i_generation, new_genome_ids, mutation_parent, crossover_parent1, crossover_parent2, operations, losses_before, losses))

            for i in range(len(losses)):
                generation[i].loss = losses[i]

            self.population = self.population + generation
            self.population = sorted(self.population, key=lambda x:x.loss["total"])

            if len(self.population) > self.population_size:
                self.population = self.population[0:self.population_size]

            get_app().publish("generation_ended", (self.i_generation, self.population))

            pbar.update(1)

        get_app().publish("evolution_ended", (self.population))
        return self.population

class TestLossFunction(LossFunction):

    def loss(self, genome : Genome):
        l = int(len(genome.genome)/2)
        return {"total": np.sum(genome.genome[0:l]) / np.sum(genome.genome[l:])}
    
# test method
if __name__ == "__main__":

    writer = NuevolutionWriter()
    evo = Nuevolution(TestLossFunction(), GeoGenomeA.build(5))
    evo.evolve()

