import json
import sys
import os
import gzip 
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from didgelab.calc.geo import Geo, geotools
import didgelab.calc.fft
from didgelab.calc.sim.sim import *
from didgelab.calc.fft import *
from didgelab.calc.conv import *
from didgelab.util.didge_visualizer import vis_didge

# open population.json.gz file and return latest generation
def get_latest_population(infile):
    line = None
    try:
        for line in gzip.open(infile):
            continue
    except Exception as e:
        logging.error(e)
    finally:
        return json.loads(line)

# get the newest subfolder of saved_evolutions folder
def get_latest_evolution_folder(basefolder = "../../../saved_evolutions/"):
    dirs = list(filter(lambda f:os.path.isdir(os.path.join(basefolder, f)), os.listdir(basefolder)))
    dirs = sorted(dirs, reverse=True)
    dirs = [os.path.join(basefolder, f) for f in dirs]
    return dirs[0]

# print various information about the population
def visualize_individuals(population, n=None, base_freq=440, max_error=1):

    if n is not None and n<len(population):
        population = population[0:n]

    for i in range(len(population)):
        geo = population[i]["representation"]["geo"]
        geo = Geo(geo)
        freqs = get_log_simulation_frequencies(1, 1000, 1)
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

    def iterate_nodes(self):
        for node in self.nodes.values():
            yield node

    def connect(self, nodeid_parent, nodeid_child, edge_name, edge_params):
        parent = self.get_node(nodeid_parent)
        child = self.get_node(nodeid_child)
        parent.add_child(child, edge_name, edge_params)
        child.add_parent(parent, edge_name, edge_params)

def build_graph(infile):
    generation_counter = 0
    nodes = Nodes()
    num_errors = 0
    try:
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
    except Exception as e:
        logging.error(e)
    finally:
        return nodes

def get_deltas(infile=None, nodes=None):
    assert infile is not None or nodes is not None

    if infile is None:
        nodes = build_graph(infile)
        
    deltas = []
    for edge in nodes.iterate_nodes():
        for key in edge.parent.losses.keys():
            loss_delta = edge.parent.losses[key] - edge.child.losses[key]
            deltas.append([edge.name, key, np.max((0, loss_delta)), edge.child.losses[key], edge.child.generation])
    deltas = pd.DataFrame(deltas, columns=["operation", "loss_type", "delta", "loss_value", "generation"])
    return deltas

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

def plot_success_probs_over_time(deltas, do_not_plot=False):
    deltas = deltas.query("loss_type=='total'").sort_values(by="generation")

    generation = -1
    operations = deltas.operation.unique()
    
    total_operations = None
    success_operations = None
    result = []
    for ix, row in deltas.iterrows():
        if row["generation"] != generation:

            if generation>0:
                for operation in operations:
                    if total_operations[operation] > 0:
                        prob = success_operations[operation] / total_operations[operation]
                        result.append([generation, operation, prob])

            generation = row["generation"]
            total_operations = {o:0 for o in operations}
            success_operations = {o:0 for o in operations}

        total_operations[row["operation"]] += 1
        if row["delta"] > 0:
            success_operations[row["operation"]] += 1

    for operation in operations:
        if total_operations[operation] > 0:
            prob = success_operations[operation] / total_operations[operation]
            result.append([generation, operation, prob])

    result = pd.DataFrame(result, columns=["generation", "operation", "success_prob"])

    if do_not_plot:
        return result
    
    sns.lineplot(data=result, x="generation", y="success_prob", hue="operation")
    plt.title("Success probabilities over time")


def plot_loss_over_time(nodes, do_not_plot=False):
    generations = {}
    for node in nodes.iterate_nodes():
        if node.generation not in generations:
            generations[node.generation] = []
        generations[node.generation].append(node.losses)

    indizes = sorted(generations.keys())
    best_loss = None
    losses = []
    for i in indizes:

        besti = np.argmin([x["total"] for x in generations[i]])
        loss = generations[i][besti]

        if best_loss is None:
            best_loss = loss

        if loss["total"] < best_loss["total"]:
            best_loss = loss
        
        for key, value in best_loss.items():
            losses.append([i, key, value])

    losses = pd.DataFrame(losses, columns=["generation", "loss_type", "loss"])

    if do_not_plot:
        return losses
    sns.lineplot(data=losses, x="generation", y="loss", hue="loss_type")
    plt.title("Loss over time")