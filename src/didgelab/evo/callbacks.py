"""
Evolution callbacks: progress bar, logging, early stopping.
"""

import logging
from time import time
import numpy as np
import os
from tqdm import tqdm
from IPython.display import clear_output
from datetime import datetime
import gzip
import json

from didgelab.app import get_app
from .evolution import Nuevolution
from .schedulers import LinearDecreasingCrossover, LinearDecreasingMutation, AdaptiveProbabilities
from .loss import LossFunction

from didgelab.visualize import plot_geo_impedance_notes
from ..acoustical_simulation import get_log_simulation_frequencies, acoustical_simulation, get_notes

class EvolutionMonitor:

    def __init__(self, target_freqs : np.array, loss : LossFunction):
        self.target_freqs = target_freqs
        self.losses = []
        self.loss = loss

    def evolution_monitor_callback(self, i_generation, population):

        loss_value = self.loss.loss(population[0])
        self.losses.append(loss_value["total"])

        frequencies = get_log_simulation_frequencies(30, 1000, 5)
        geo = population[0].genome2geo()
        impedances = acoustical_simulation(geo, frequencies)

        notes = get_notes(frequencies, impedances, target_freqs=self.target_freqs)
        clear_output()

        print()

        print(f"Generation {i_generation}")

        loss_str = {key: round(value, 2) for key, value in loss_value.items()}
        print(f"losses: {loss_str}")
        print(notes.round(2))

        fig, axes = plot_geo_impedance_notes(geo)

class SaveEvolution:

    def __init__(self, output_dir = "outputs"):
        self.output_dir = output_dir
        current_datetime = datetime.now()
        iso_format_string = current_datetime.isoformat()
        self.outfile = os.path.join(self.output_dir, iso_format_string + ".json.gz")
        self.start_time = time()

    def evolution_monitor_callback(self, i_generation, population):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        def serialize_loss(loss):
            return {key: float(value) for key, value in loss.items()}

        log_message = {
            "population": [p.genome.tolist() for p in population],
            "losses": [serialize_loss(p.loss) for p in population],
            "i_generation": i_generation,
            "loss0": serialize_loss(population[0].loss),
            "time_elapsed": time() - self.start_time
        }

        with gzip.open(self.outfile, "wb") as f:
            log_message = json.dumps(log_message).encode('utf-8')
            f.write(log_message)

def init_standard_evolution(target_freqs, evo):

    schedulers = [
        LinearDecreasingCrossover(),
        LinearDecreasingMutation()
    ]

    AdaptiveProbabilities()

    monitor = EvolutionMonitor(target_freqs, evo.loss)
    se = SaveEvolution()
    def evolution_monitor_callback(i_generation, population):
        monitor.evolution_monitor_callback(i_generation, population)
        se.evolution_monitor_callback(i_generation, population)

    evo.callback_generation_ended = evolution_monitor_callback

def load_latest_evolution(output_folder = "outputs"):
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

class NuevolutionProgressBar:
    """Shows a tqdm progress bar that advances each generation and displays best total loss."""

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


class PrintEvolutionInformation:
    """Logs the best individual's loss and note analysis every interval generations (and generation 1)."""

    def __init__(self, interval=5, base_freq=440):

        self.base_freq = base_freq
        self.interval = interval
        self.last_generation_time = None

        def generation_ended(i_generation, population):

            duration = None
            if self.last_generation_time is not None:
                duration = time() - self.last_generation_time
            self.last_generation_time = time()

            if i_generation > 1 or i_generation % self.interval == 0:
                from didgelab.sim.tlm_cython_lib.sim import (
                    get_log_simulation_frequencies,
                    create_segments,
                    compute_impedance,
                    get_notes,
                )
                genome = population[0]
                losses = [f"{key}: {value}" for key, value in genome.loss.items()]
                msg = "Losses:\n" + "\n".join(losses)

                geo = genome.genome2geo()
                freqs = get_log_simulation_frequencies(1, 1000, 5)
                segments = create_segments(geo)
                impedances = compute_impedance(segments, freqs)
                notes = get_notes(freqs, impedances, base_freq=self.base_freq).round(2).to_string()
                msg += "\n" + notes

                if duration is not None:
                    msg += f"\nTime per generation: {duration:.2f} seconds"

                logging.info(msg)

        get_app().subscribe("generation_ended", generation_ended)


class EarlyStopping:
    """Stops evolution when the best loss has not improved for duration generations (intended use)."""

    def __init__(self, duration=100):
        self.duration = duration
        self.best_loss = None
        self.last_loss_update = None

        def generation_ended(i_generation, population):
            thisloss = population[0].loss["total"]
            if self.best_loss is None or thisloss < self.best_loss:
                self.best_loss = thisloss
                self.last_loss_update = i_generation
            elif i_generation - self.last_loss_update > self.duration:
                logging.info("stop evolution because it did not improve")
                evo = get_app().get_service(Nuevolution)
                evo.continue_evolution = False

        get_app().subscribe("generation_ended", generation_ended)
