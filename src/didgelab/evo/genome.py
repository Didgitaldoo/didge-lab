"""
Evolvable genome types: base Genome, GeoGenome, and GeoGenomeA.

Genomes are vectors of genes in [0, 1]; subclasses define how they are
interpreted (e.g. as bore geometry).
"""

import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
import threading

from didgelab.geo import Geo


class Genome(ABC):
    """
    Abstract base for an evolvable genome (vector of genes in [0, 1]).

    Subclasses define how the genome is interpreted (e.g. as bore geometry).
    Each instance has a unique id, optional loss dict, and optional named_parameters.
    """

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
        """Return a serializable representation of this genome (e.g. for logging)."""
        return self.genome.tolist()

    def randomize_genome(self):
        """Replace the genome with random values in [0, 1], keeping the same length."""
        self.genome = np.random.sample(size=len(self.genome))

    @staticmethod
    def generate_id():
        """Return a new unique integer id for a genome (thread-safe)."""
        Genome.genome_id_lock.acquire()
        try:
            id = Genome.genome_id_counter
            Genome.genome_id_counter += 1
        finally:
            Genome.genome_id_lock.release()
        return id

    def clone(self):
        """Return a deep copy of this genome with a new id and loss cleared."""
        clone = deepcopy(self)
        clone.id = Genome.generate_id()
        clone.loss = None
        return clone


class GeoGenome(Genome):
    """
    Genome that encodes a didgeridoo bore geometry.

    representation() returns geo data and a text analysis (impedance/notes).
    Subclasses must implement genome2geo() to map the gene vector to a Geo.
    """

    def representation(self):
        from didgelab.sim.tlm_cython_lib.sim import (
            get_log_simulation_frequencies,
            create_segments,
            compute_impedance,
            get_notes,
        )
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
        """Convert the genome to a didgeridoo geometry (Geo)."""
        pass


class GeoGenomeA(GeoGenome):
    """
    Genome that encodes a bore as segment lengths and diameters (linear interpolation).

    Gene layout: one length scale plus (length, diameter) pairs per segment.
    Total length is in [1000, 2000] mm; diameters are scaled to [25, 100] mm range,
    with mouth diameter fixed at 32 mm.
    """

    def build(n_segments):
        """Construct a new GeoGenomeA with random genes for n_segments (n_genes = n_segments*2 + 1)."""
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
        i = 1
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

        geo = list(zip(x, y))

        return Geo(geo)
