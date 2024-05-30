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
import gzip

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
        self.named_parameters = {}

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

        if x[-1] != 0:
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
        mutation[mutation>1] = 1

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

class PartSwapCrossover(CrossoverOperator):

    def apply(self, parent1 : Genome, parent2 : Genome, evolution_parameters : Dict = {}) -> Genome:
        assert type(parent1) == type(parent2)

        indizes = np.random.choice(np.arange(len(parent1.genome)), size=2, replace=False)
        i1 = np.min(indizes)
        i2 = np.max(indizes)

        offspring = parent1.clone()
        offspring.genome = np.concatenate(
            (parent1.genome[0:i1], 
             parent2.genome[i1:i2], 
             parent1.genome[i2:]))

        assert len(offspring.genome) == len(parent1.genome)
        return offspring, self.describe(parent1, parent2, offspring)

class PartAverageCrossover(CrossoverOperator):

    def apply(self, parent1 : Genome, parent2 : Genome, evolution_parameters : Dict = {}) -> Genome:
        assert type(parent1) == type(parent2)

        indizes = np.random.choice(np.arange(len(parent1.genome)), size=2, replace=False)
        i1 = np.min(indizes)
        i2 = np.max(indizes)

        offspring = parent1.clone()
        offspring.genome = np.concatenate(
            (parent1.genome[0:i1], 
             (parent1.genome[i1:i2] + parent2.genome[i1:i2]) / 2, 
             parent1.genome[i2:]))

        assert len(offspring.genome) == len(parent1.genome)
        return offspring, self.describe(parent1, parent2, offspring)

class SingleMutation(MutationOperator):

    def apply(self, genome : Genome, evolution_parameters : Dict) -> Tuple[Genome, Dict]:
        new_genome = genome.clone()

        i = np.random.randint(0, len(new_genome.genome))
        v = new_genome.genome[i]
        v += np.random.uniform(-1, 1)
        if v<0:
            v=0
        if v>1:
            v=1
        new_genome.genome[i] = v
        return new_genome, self.describe(genome, new_genome)

class NuevolutionWriter:
    
    def __init__(self, 
        interval=20,
        write_loss = False,
        write_population_interval=100, # log best individual at this interval
        log_operations = True):

        self.interval = interval
        self.writer = None
        self.format = None
        self.write_loss = write_loss
        self.write_population_interval = write_population_interval
        self.log_operations = log_operations
        self.log_evolutions_file = None
        self.write_population_writer = None
        self.csvfile = None
        self.current_generation = None

        def generation_ended(i_generation, population):
            self.current_generation = i_generation

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

        if self.log_evolutions_file is None:
            outfile = os.path.join(get_app().get_output_folder(), "evolution_operations.jsonl.gz")
            self.log_evolutions_file = gzip.open(outfile, "w")

        def flatten_dicts(list_of_dicts):
            flattened = DefaultDict(list)
            for di in list_of_dicts:
                for key, value in di.items():
                    flattened[key].append(value)
            return flattened

        mutation = DefaultDict(list)

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

    def write_population(self, population : List[Genome], generation=None):

        if generation is None:
            generation = self.current_generation

        if self.write_population_writer is None:
            outfile = os.path.join(get_app().get_output_folder(), f"population.json.gz")
            self.write_population_writer = gzip.open(outfile, "w")

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

        data2 = {"generation": generation, "population": data}
        self.write_population_writer.write(json.dumps(data2).encode())
        self.write_population_writer.write("\n".encode())

        outfile = os.path.join(get_app().get_output_folder(), "latest_population.json")
        with open(outfile, "w") as f:
            json.dump(data2, f)

    def write_loss(self, i_generation, population : List[Genome]):

        if self.writer is None and self.write_loss:
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
        },
        evolution_operators = [
            RandomCrossover(), 
            AverageCrossover(), 
            SimpleMutation(),
            SingleMutation(),
            PartAverageCrossover(),
            PartSwapCrossover()
        ],
        evolution_operator_probs = None):

        self.loss = loss
        self.father_genome = father_genome
        self.generation_size = generation_size
        self.num_generations = num_generations
        self.population_size = population_size
        self.evolution_parameters = evolution_parameters
        self.evolution_operators = evolution_operators

        if evolution_operator_probs is None:
            evolution_operator_probs = [1/len(evolution_operators)]*len(evolution_operators)
        self.evolution_operator_probs = evolution_operator_probs

        self.i_generation = -1
        self.population = None
        
        self.recompute_losses = False

        self.continue_evolution

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
            "evolution_operators": [type(o).__name__ for o in evolution_operators],
            "evolution_operator_probs": evolution_operator_probs
        }
        logging_infos = sorted([f"{key}: {value}" for key, value in logging_infos.items()])
        logging_infos = "Initialize Nuevolution\n" + "\n".join(logging_infos)
        logging.info(logging_infos)

    def get_evolution_progress(self):
        return (self.i_generation+1) / self.num_generations 

    def evolve(self):

        # initialize
        num_workers = multiprocessing.cpu_count()
        logging.info(f"initialize threadpoolexecutor with {num_workers} workers")
        pool = ThreadPoolExecutor(max_workers=num_workers)

        self.population = []
        for i in range(self.generation_size):
            mutant = self.father_genome.clone()
            mutant.randomize_genome()
            self.population.append(mutant)

        probs = []
        
        logging.info("compute initial generation")
        losses = list(tqdm(pool.map(self.loss.loss, self.population), total=len(self.population)))
        for i in range(len(losses)):
            self.population[i].loss = losses[i]
        self.population = sorted(self.population, key=lambda x:x.loss["total"])

        get_app().publish("log_evolution_operations", (self.i_generation, self.population, [], []))

        # evolve
        for i_generation in range(1, self.num_generations+1):
            
            if not self.continue_evolution:
                break
            
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

            operations = np.random.choice(self.evolution_operators, size=self.generation_size, p=self.evolution_operator_probs)

            new_generation = []
            mutation_operations = []
            crossover_operations = []
            for operator in operations:

                if isinstance(operator, MutationOperator):
                    parent = np.random.choice(self.population, p=probs)
                    individual, operation_description = operator.apply(parent, self.evolution_parameters)
                    mutation_operations.append(operation_description)
                    new_generation.append(individual)
                elif isinstance(operator, CrossoverOperator):
                    parent1 = np.random.choice(self.population, p=probs)
                    parent2 = parent1
                    while parent1.id == parent2.id:
                        parent2 = np.random.choice(self.population, p=probs)
                    individual, operation_description = operator.apply(parent1, parent2, self.evolution_parameters)
                    crossover_operations.append(operation_description)
                    new_generation.append(individual)

            # compute losses                    
            losses = list(pool.map(self.loss.loss, new_generation))
            for i in range(len(losses)):
                new_generation[i].loss = losses[i]

            # collect logging data
            get_app().publish("log_evolution_operations", (self.i_generation, new_generation, mutation_operations, crossover_operations))

            self.population = self.population + new_generation
            self.population = sorted(self.population, key=lambda x:x.loss["total"])

            if len(self.population) > self.population_size:
                self.population = self.population[0:self.population_size]

            get_app().publish("generation_ended", (self.i_generation, self.population))


            # losses = np.array([p.loss for p in self.population])
            # indizes = np.random.choice(np.arange(len(self.population)), size=self.generation_size, replace=False, p=probs)
            # generation = [self.population[i] for i in indizes]
            # losses_before = [p.loss for p in generation]

            # mutation_prob = self.evolution_parameters["individual_mutation_prob"]
            # if mutation_prob is None:
            #     mutation_prob = 0.5

            # i_mutants = np.arange(self.generation_size)[np.random.sample(self.generation_size)<mutation_prob]
            # mutation_operations = []
            # for i in i_mutants:
            #     if len(self.mutation_operators) == 1:
            #         operator = self.mutation_operators[0]
            #     else:
            #         operator = np.random.choice(self.mutation_operators)

            #     individual, operation = operator.apply(generation[i], self.evolution_parameters)
            #     generation[i] = individual
            #     mutation_operations.append(operation)

            # # crossover
            # crossover_prob = self.evolution_parameters["crossover_prob"]
            # if mutation_prob is None:
            #     crossover_prob = 0.5

            # i_crossover = np.arange(self.generation_size)[np.random.sample(self.generation_size)<crossover_prob]
            # crossover_operations = []
            # for parent1 in i_crossover:
            #     parent2 = parent1
            #     while parent1 == parent2:
            #         parent2 = np.random.choice(np.arange(len(self.population)), p=probs)

            #     if len(self.crossover_operators) == 1:
            #         operator = self.crossover_operators[0]
            #     else:
            #         operator = np.random.choice(self.crossover_operators)

            #     # generation[parent1] = operator.apply(generation[parent1], self.population[parent2], self.evolution_parameters)

            #     individual, operation = operator.apply(generation[parent1], self.population[parent2], self.evolution_parameters)
            #     generation[parent1] = individual
            #     crossover_operations.append(operation)
                
            # # add only changed genes to population
            # i_changed = np.arange(self.generation_size)
            # i_changed = i_changed[[i in i_mutants or i in i_crossover for i in i_changed]]

            # # collect logging data
            # generation = [generation[i] for i in i_changed]
            # losses = list(pool.map(self.loss.loss, generation))
            # for i in range(len(losses)):
            #     generation[i].loss = losses[i]

            # get_app().publish("log_evolution_operations", (self.i_generation, generation, mutation_operations, crossover_operations))

            # self.population = self.population + generation
            # self.population = sorted(self.population, key=lambda x:x.loss["total"])

            # if len(self.population) > self.population_size:
            #     self.population = self.population[0:self.population_size]

            # get_app().publish("generation_ended", (self.i_generation, self.population))


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
            best_loss = population[0].loss['total']
            self.pbar.set_description(f"best loss: {best_loss:.2f}")

        get_app().subscribe("generation_ended", update)

class TestLossFunction(LossFunction):

    def loss(self, genome : Genome):
        l = int(len(genome.genome)/2)
        return {"total": np.sum(genome.genome[0:l]) / np.sum(genome.genome[l:]), "test": -5, "test2": 10}

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


class AdaptiveProbabilities:

    def __init__(self):
        get_app().subscribe("log_evolution_operations", self.log_evolution_operations)
        self.loss_index = {}
        self.loss_history = {}
        self.window_size = 100
        self.probabilities = {}

    # compute a dictionary of all operations to a list of differences between child loss and parent loss
    def compute_loss_delta_of_generation(self, mutation_operations, crossover_operations):
        generation = {}
        for mutation in mutation_operations:

            if mutation["father_id"] not in self.loss_index:
                continue
            if mutation["child_id"] not in self.loss_index:
                continue

            father_loss = self.loss_index[mutation["father_id"]]
            child_loss = self.loss_index[mutation["child_id"]]
            
            if mutation["operation"] not in generation:
                generation[mutation["operation"]] = []
            
            generation[mutation["operation"]].append(child_loss - father_loss)
        
        for crossover in crossover_operations:
            if crossover["parent1_genome"] not in self.loss_index:
                continue
            if crossover["parent2_genome"] not in self.loss_index:
                continue
            if crossover["child_id"] not in self.loss_index:
                continue

            p1_loss = self.loss_index[crossover["parent1_genome"]]
            p2_loss = self.loss_index[crossover["parent2_genome"]]
            child_loss = self.loss_index[crossover["child_id"]]

            minloss = np.min((p1_loss, p2_loss))
            if crossover["operation"] not in generation:
                generation[crossover["operation"]] = []
            generation[crossover["operation"]].append(child_loss - minloss)

        return generation

    def log_evolution_operations(self, i_generation, individuals, mutation_operations, crossover_operations):
        
        # maintain an index of node id -° loss
        if len(self.loss_index) == 0:
            pop = get_app().get_service(Nuevolution).population
            for i in pop:
                self.loss_index[i.id] = i.loss["total"]

        for i in range(len(individuals)):
            self.loss_index[individuals[i].id] = individuals[i].loss["total"]

        generation = self.compute_loss_delta_of_generation(mutation_operations, crossover_operations)

        for operation, losses in generation.items():
            if operation not in self.loss_history:
                self.loss_history[operation] = []
            self.loss_history[operation].extend(losses)
            if len(self.loss_history[operation]) > self.window_size:
                self.loss_history[operation] = self.loss_history[operation][-self.window_size:]

            num_successful = (np.array(self.loss_history[operation]) < 0).astype(int).sum()
            num_total = len(self.loss_history[operation])
            self.probabilities[operation] = np.max((num_successful / num_total, 0.05))

        # normalize probabiliies
        prob_sum =  np.sum(list(self.probabilities.values()))
        self.probabilities = {key:value/prob_sum for key, value in self.probabilities.items()}
        
        # update probs in nuevolution
        nuevo = get_app().get_service(Nuevolution)
        probs = []
        for i in range(len(nuevo.evolution_operators)):
            o = nuevo.evolution_operators[i]
            if type(o).__name__ in self.probabilities:
                p = self.probabilities[type(o).__name__]
            else:
                p = nuevo.evolution_operator_probs[i]
            probs.append(p)
        nuevo.evolution_operator_probs = probs

# test method
if __name__ == "__main__":
    # np.seterr(invalid='raise')
    writer = NuevolutionWriter()
    evo = Nuevolution(
        TestLossFunction(), 
        GeoGenomeA.build(5),
        num_generations=10,
        population_size=1000,
        generation_size=20)

    schedulers = [
        LinearDecreasingMutation()
    ]
    AdaptiveProbabilities()
    evo.evolve()

