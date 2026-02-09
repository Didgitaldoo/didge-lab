"""
Entry point for: python -m didgelab.evo
"""

from . import (
    NuevolutionWriter,
    Nuevolution,
    TestLossFunction,
    GeoGenomeA,
    LinearDecreasingMutation,
    AdaptiveProbabilities,
)

if __name__ == "__main__":
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
