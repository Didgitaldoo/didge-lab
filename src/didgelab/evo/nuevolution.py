"""
python -m didgelab.evo.nuevolution
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, DefaultDict, Tuple
import multiprocessing
from tqdm import tqdm
from didgelab.app import get_app, get_config
import csv
import os
from time import time
import json
from json import JSONEncoder
from didgelab.calc.geo import Geo
from didgelab.calc.sim.sim import get_log_simulation_frequencies, create_segments, compute_impedance, get_notes
from copy import deepcopy
import sys
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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

    def clone(self):
        clone = deepcopy(self)
        clone.id = Genome.generate_id()
        return clone

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

    def genome2geo(self) -> Geo:

        d0 = 32
        x = [0]
        y = [d0]
        min_l = 1000
        max_l = 2000

        d_factor = 75
        min_d = 25

        l = self.genome[0] * (max_l-min_l) + min_l 
        i=1
        while i+2 < len(self.genome):
            x.append(self.genome[i] + x[-1])
            y.append(self.genome[i+1])
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
    def apply(self, genome : Genome, evolution_parameters : Dict) -> Tuple[Genome, Dict]:
        pass

    def describe(self, father_genome, child_genome):
        return {
            "operation": type(self).__name__,
            "father_id": father_genome.id,
            "child_id": child_genome.id
        }

class CrossoverOperator(ABC):

    @abstractmethod
    def apply(self, genome1 : Genome, genome2 : Genome, evolution_parameters : Dict) -> Genome:
        pass

    def describe(self, parent1_genome, parent2_genome, child_genome):
        return {
            "operation": type(self).__name__,
            "parent1_genome": parent1_genome.id,
            "parent2_genome": parent2_genome.id,
            "child_id": child_genome.id
        }


class SimpleMutation(MutationOperator):

    def apply(self, genome : Genome, evolution_parameters : Dict) -> Tuple[Genome, Dict]:

        mr = evolution_parameters["mutation_rate"]
        if mr is None:
            mr = 0.5
        mp = evolution_parameters["gene_mutation_prob"]
        if mp is None:
            mp = 0.5

        mutation = np.random.uniform(low=-mr, high=mr, size=len(genome.genome))
        mutation *= (np.random.sample(size=len(mutation))<mp).astype(int)
        mutation = genome.genome + mutation
        mutation[mutation<0] = 0
        mutation[mutation>0] = 1

        new_genome = genome.clone()
        new_genome.genome = mutation
        return new_genome, self.describe(genome, new_genome)
    
class RandomMutation(MutationOperator):

    def apply(self, genome : Genome, evolution_parameters : Dict) -> Tuple[Genome, Dict]:
        new_genome = genome.clone()
        new_genome.genome = np.random.sample(len(genome.genome))
        return new_genome, self.describe(genome, new_genome)
    
class RandomCrossover(CrossoverOperator):

    def apply(self, parent1 : Genome, parent2 : Genome, evolution_parameters : Dict) -> Genome:
        assert type(parent1) == type(parent2)
        new_genome = list(zip(parent1.genome, parent2.genome))
        new_genome = np.array([np.random.choice(x) for x in new_genome])

        offspring = parent1.clone()
        offspring.genome = new_genome
        return offspring, self.describe(parent1, parent2, offspring)
    
class AverageCrossover(CrossoverOperator):

    def apply(self, parent1 : Genome, parent2 : Genome, evolution_parameters : Dict) -> Genome:
        assert type(parent1) == type(parent2)
        new_genome = (parent1.genome + parent2.genome) / 2

        offspring = parent1.clone()
        offspring.genome = new_genome
        return offspring, self.describe(parent1, parent2, offspring)

class NuevolutionWriter:
    
    def __init__(self, 
        interval=20,
        write_loss = True,
        write_population_interval=100, # log best individual at this interval
        log_operations = True):

        self.interval = interval
        self.writer = None
        self.format = None
        self.write_loss = self.write_loss
        self.write_population_interval = write_population_interval
        self.log_operations = log_operations
        self.log_evolutions_file = None

        def generation_ended(i_generation, population):
            if self.write_loss:
                self.write_loss(i_generation, population)
            if self.write_population_interval > 0 and i_generation % self.write_population_interval == 0:
                msg = f"generation {i_generation} ended, writing population to file\n"
                msg += "loss:\n"
                losses = [f"{key}: {value}" for key, value in population[0].loss.items()]
                msg += "\n".join(losses)
                logging.info(msg)                
                self.write_population(population, i_generation)

        if self.write_population_interval>0 or self.write_loss:
            get_app().subscribe("generation_ended", generation_ended)

        def evolution_ended(population):
            self.csvfile.close()
            self.log_evolutions_file.close()
            self.writer = None
            self.write_population(population)

        get_app().subscribe("evolution_ended", evolution_ended)
        get_app().subscribe("log_evolution_operations", self.log_evolution_operations)
        get_app().register_service(self)
        
    def log_evolution_operations(self, i_generation, individuals, mutation_operations, crossover_operations):

        if self.log_evolutions_file is None:
            outfile = os.path.join(get_app().get_output_folder(), "evolution_operations.jsonl")
            self.log_evolutions_file = open(outfile, "w")

        def flatten_dicts(list_of_dicts):
            flattened = DefaultDict(list)
            for di in list_of_dicts:
                for key, value in di.items():
                    flattened[key].append(value)
            return flattened

        mutation = DefaultDict(list)
        data = {
            "genome_ids": [g.id for g in individuals],
            "genomes": [g.genome.tolist() for g in individuals],
            "losses": flatten_dicts([g.loss for g in individuals]),
            "mutations_operations": flatten_dicts(mutation_operations),
            "crossover_operations": flatten_dicts(crossover_operations),
            "date": datetime.now().isoformat(),
        }
        
        self.log_evolutions_file.write(json.dumps(data))
        self.log_evolutions_file.write("\n")

    def log_evolution_operations_old(self, i_generation, new_genome_ids, mutation_parent, crossover_parent1, crossover_parent2, operations, losses):

        if self.evolution_operations_stream is None:
            outfile = os.path.join(get_app().get_output_folder(), "evolution_operations.csv")
            self.evolution_operations_f = open(outfile, "w")
            self.evolution_operations_stream = csv.writer(self.evolution_operations_f)
            
            self.evolution_operations_format = ["generation", "new_genome_id", "mutation_parent", "crossover_parent_1", "crossover_parent_2", "mutation", "crossover"]
            for key in losses[0].keys():
                self.evolution_operations_format.append("loss_" + key)

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
                operations[i]]
            
            if len(operations[i]) > 1:
                row.append(operations[i][1])
            else:
                row.append(None)

            for key, value in losses[i].items():
                row.append(value)

            self.evolution_operations_stream.writerow(row)

    def write_population(self, population : List[Genome], generation=None):

        if generation is None:
            generation = f"_{generation}"
        outfile = os.path.join(get_app().get_output_folder(), f"population{generation}.json")
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

    def write_loss(self, i_generation, population : List[Genome]):

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
        evolution_parameters = {
            "mutation_rate": 0.5,
            "gene_mutation_prob": 0.5,
            "individual_mutation_prob": 0.5,
            "crossover_prob": 0.5
        },
        crossover_operators = [RandomCrossover(), AverageCrossover()],
        mutation_operators = [SimpleMutation(), RandomMutation()]):

        self.loss = loss
        self.father_genome = father_genome
        self.generation_size = generation_size
        self.num_generations = num_generations
        self.population_size = population_size
        self.evolution_parameters = evolution_parameters
        self.crossover_operators = crossover_operators
        self.mutation_operators = mutation_operators

        self.i_generation = -1
        self.population = None
        
        self.recompute_losses = False

        get_app().register_service(self)

        def recompute_loss():
            self.recompute_losses = True

        get_app().subscribe("recompute_loss", recompute_loss)

        logging_infos = {
            "loss": type(loss).__name__,
            "father_genome": type(father_genome).__name__,
            "generation_size": generation_size,
            "num_generations": num_generations,
            "population_size": population_size,
            "evolution_parameters": evolution_parameters,
            "crossover_operators": [type(o).__name__ for o in crossover_operators],
            "mutation_operators": [type(o).__name__ for o in mutation_operators]
        }
        logging_infos = sorted([f"{key}: {value}" for key, value in logging_infos.items()])
        logging_infos = "Initialize Nuevolution\n" + "\n".join(logging_infos)
        logging.info(logging_infos)

    def get_evolution_progress(self):
        return (self.i_generation+1) / self.num_generations 

    def evolve(self):

        # initialize
        num_workers = multiprocessing.cpu_count()
        pool = ThreadPoolExecutor(max_workers=num_workers)

        self.population = []
        for i in range(self.generation_size):
            mutant = self.father_genome.clone()
            mutant.randomize_genome()
            self.population.append(mutant)

        probs = []

        
        logging.info("compute initial generation")
        losses = list(tqdm(pool.map(self.loss.loss, self.population), total=len(self.population)))
        # losses = pool.map(self.loss.loss, self.population)
        for i in range(len(losses)):
            self.population[i].loss = losses[i]
        self.population = sorted(self.population, key=lambda x:x.loss["total"])

        get_app().publish("log_evolution_operations", (self.i_generation, self.population, [], []))

        # evolve
        for i_generation in range(self.num_generations):
            
            get_app().publish("generation_started", (self.i_generation, self.population))

            if len(probs) != len(self.population):
                # compute probabilities that an individual will be selected
                probs = (1+np.arange(len(self.population))) / len(self.population)
                probs = np.exp(probs)
                probs = np.exp(probs)
                probs = np.flip(probs)
                probs /= probs.sum()

            self.i_generation = i_generation

            if self.recompute_losses:
                losses = list(pool.map(self.loss.loss, self.population))
                self.recompute_losses = False

            losses = np.array([p.loss for p in self.population])

            indizes = np.random.choice(np.arange(len(self.population)), size=self.generation_size, replace=False, p=probs)
            generation = [self.population[i] for i in indizes]
            losses_before = [p.loss for p in generation]

            mutation_prob = self.evolution_parameters["individual_mutation_prob"]
            if mutation_prob is None:
                mutation_prob = 0.5

            i_mutants = np.arange(self.generation_size)[np.random.sample(self.generation_size)<mutation_prob]
            mutation_operations = []
            for i in i_mutants:
                if len(self.mutation_operators) == 1:
                    operator = self.mutation_operators[0]
                else:
                    operator = np.random.choice(self.mutation_operators)

                individual, operation = operator.apply(generation[i], self.evolution_parameters)
                generation[i] = individual
                mutation_operations.append(operation)

            # crossover
            crossover_prob = self.evolution_parameters["crossover_prob"]
            if mutation_prob is None:
                crossover_prob = 0.5

            i_crossover = np.arange(self.generation_size)[np.random.sample(self.generation_size)<crossover_prob]
            crossover_operations = []
            for parent1 in i_crossover:
                parent2 = parent1
                while parent1 == parent2:
                    parent2 = np.random.choice(np.arange(len(self.population)), p=probs)

                if len(self.crossover_operators) == 1:
                    operator = self.crossover_operators[0]
                else:
                    operator = np.random.choice(self.crossover_operators)

                # generation[parent1] = operator.apply(generation[parent1], self.population[parent2], self.evolution_parameters)

                individual, operation = operator.apply(generation[parent1], self.population[parent2], self.evolution_parameters)
                generation[parent1] = individual
                crossover_operations.append(operation)
                
            # add only changed genes to population
            i_changed = np.arange(self.generation_size)
            i_changed = i_changed[[i in i_mutants or i in i_crossover for i in i_changed]]

            # collect logging data
            generation = [generation[i] for i in i_changed]
            losses = list(pool.map(self.loss.loss, generation))
            for i in range(len(losses)):
                generation[i].loss = losses[i]

            get_app().publish("log_evolution_operations", (self.i_generation, generation, mutation_operations, crossover_operations))

            self.population = self.population + generation
            self.population = sorted(self.population, key=lambda x:x.loss["total"])

            if len(self.population) > self.population_size:
                self.population = self.population[0:self.population_size]

            get_app().publish("generation_ended", (self.i_generation, self.population))


        get_app().publish("evolution_ended", (self.population))
        return self.population

class NuevolutionProgressBar:

    def __init__(self):

        self.pbar = None
        
        def update(i_generation, population):
            if self.pbar is None:
                num_generations = get_app().get_service(Nuevolution).num_generations
                self.pbar = tqdm(total=num_generations)
            self.pbar.update(1)

        get_app().subscribe("generation_ended", update)

class TestLossFunction(LossFunction):

    def loss(self, genome : Genome):
        l = int(len(genome.genome)/2)
        return {"total": np.sum(genome.genome[0:l]) / np.sum(genome.genome[l:])}

class LinearDecreasingMutation:

    def __init__(self):

        n_steps = 4
        self.schedule = np.arange(1,n_steps+1) / n_steps
        self.rates = 1-(np.arange(n_steps)/n_steps)
        self.i = -1

        def update(i_generation, population):

            if self.i>=len(self.schedule):
                return
            
            nuevolution = get_app().get_service(Nuevolution)
            progress = nuevolution.get_evolution_progress()
            if self.i == -1 or progress > self.schedule[self.i]:
                self.i += 1
                rate = self.rates[self.i]
                nuevolution.evolution_parameters["mutation_rate"] = rate
                nuevolution.evolution_parameters["gene_mutation_prob"] = rate
                get_app().publish("recompute_loss")

        get_app().subscribe("generation_started", update)

class LinearDecreasingCrossover:

    def __init__(self):

        n_steps = 4
        self.schedule = np.arange(1,n_steps+1) / n_steps
        self.rates = 0.5-(np.arange(n_steps)/n_steps)
        self.rates = [np.max(x, 0) for x in self.rates]
        self.i = -1

        def update(i_generation, population):
            if self.i>=len(self.schedule):
                return
            
            nuevolution = get_app().get_service(Nuevolution)
            progress = nuevolution.get_evolution_progress()
            if self.i == -1 or progress > self.schedule[self.i]:
                self.i += 1
                rate = self.rates[self.i]
                nuevolution.evolution_parameters["crossover_prob"] = rate
                get_app().publish("recompute_loss")
        get_app().subscribe("generation_started", update)


# test method
if __name__ == "__main__":
    # np.seterr(invalid='raise')
    writer = NuevolutionWriter()
    evo = Nuevolution(
        TestLossFunction(), 
        GeoGenomeA.build(5),
        num_generations=1000,
        population_size=1000,
        generation_size=200)
    evo.evolve()

