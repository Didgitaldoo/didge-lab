import json
import sys
import os
import gzip 
import logging

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

# a node in the parent / children hierarchy of evolution_operations.jsonl.gz
class Node:

    def __init__(self, genome_id, losses, generation):
        self.genome_id = genome_id
        self.losses = losses
        self.generation = generation
        self.children = []
        self.parents = []

    def add_child(self, child_node, edge_name, edge_params):
        con = Edge(edge_name, edge_params, self, child_node)
        self.children.append(con)

    def add_parent(self, parent_node, edge_name, edge_params):
        con = Edge(edge_name, edge_params, parent_node, self)
        self.parents.append(con)

# an edge in the parent / children hierarchy of evolution_operations.jsonl.gz
class Edge:

    def __init__(self, name, params, parent, child):
        self.name = name
        self.params = params
        self.parent = parent
        self.child = child

class Nodes:

    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.genome_id] = node

    def has_node_id(self, node_id):
        return node_id in self.nodes.keys()

    def get_node(self, node_id):
        return self.nodes[node_id]
    
    def get_generation(self, i_generation):
        return list(filter(lambda x:x.generation==i_generation, self.nodes.values()))

    def iterate_edges(self):
        for node in self.nodes.values():
            for edge in node.children:
                yield edge

    def connect(self, nodeid_parent, nodeid_child, edge_name, edge_params):
        parent = self.get_node(nodeid_parent)
        child = self.get_node(nodeid_child)
        parent.add_child(child, edge_name, edge_params)
        child.add_parent(parent, edge_name, edge_params)

# build the parent / children hierarchy of evolution_operations.jsonl.gz
def build_graph(infile):
    generation_counter = 0
    nodes = Nodes()
    num_errors = 0
    for line in gzip.open(infile):
        try:
            data = json.loads(line)
        except Exception as e:
            num_errors += 1
            if num_errors >= 3:
                break
            logging.error(e)
            continue
        for i in range(len(data["genome_ids"])):

            losses = {key: data["losses"][key][i] for key in data["losses"].keys()}
            node = Node(
                data["genome_ids"][i], 
                losses,
                generation_counter
            )
            nodes.add_node(node)
        generation_counter += 1

        mo = data["mutations_operations"]
        if len(mo)>0:
            for i in range(len(mo["operation"])):
                father_id = mo["father_id"][i]
                child_id = mo["child_id"][i]

                if nodes.has_node_id(child_id) and nodes.has_node_id(father_id):
                    nodes.connect(
                        father_id,
                        child_id,
                        mo["operation"][i],
                        {}
                    )

        co = data["crossover_operations"]
        if len(co)>0:
            for i in range(len(co["operation"])):
                parent1 = co["parent1_genome"][i]
                parent2 = co["parent2_genome"][i]
                child_id = co["child_id"][i]
                if nodes.has_node_id(child_id) and nodes.has_node_id(parent1) and nodes.has_node_id(parent2):
                        nodes.connect(
                            parent1,
                            child_id,
                            co["operation"][i],
                            {}
                        )
                        nodes.connect(
                            parent2,
                            child_id,
                            co["operation"][i],
                            {}
                        )
    return nodes

# get the loss improvements from  evolution_operations.jsonl.gz
def get_deltas(infile):
    nodes = build_graph(infile)
    deltas = []
    for edge in nodes.iterate_edges():
        for key in edge.parent.losses.keys():
            loss_delta = edge.parent.losses[key] - edge.child.losses[key]
            deltas.append([edge.name, key, np.max((0, loss_delta)), edge.child.losses[key], edge.child.generation])
    deltas = pd.DataFrame(deltas, columns=["operation", "loss_type", "delta", "loss_value", "generation"])
    return deltas

# convert deltas to the probabilities that a mutation operation is succesful
def get_success_probs(deltas):
    success_probs = []
    for loss in deltas.loss_type.unique():
        for operation in deltas.operation.unique():
            a = len(deltas.query("operation==@operation and loss_type==@loss and delta>0"))    
            b = len(deltas.query("operation==@operation and loss_type==@loss"))
            operation_str = operation
            operation_str = operation_str.replace("Mutation", "\nMutation")
            operation_str = operation_str.replace("Crossover", "\nCrossover")
            success_probs.append([operation_str, loss, a/b])

        a = len(deltas.query("delta>0"))    
        b = len(deltas)
        success_probs.append(["Total\nAverage", loss, a/b])

    success_probs = pd.DataFrame(success_probs, columns=["operation", "loss_type", "success_prob"])
    success_probs = success_probs.sort_values("success_prob", ascending=False)
    return success_probs

