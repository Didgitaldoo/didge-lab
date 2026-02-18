"""
DidgeLab: acoustical simulation and computer-aided design of didgeridoos.

Import from the root for the main API, e.g.::

    from didgelab import Nuevolution, Geo, GeoGenome, SimpleMutation
    from didgelab import acoustical_simulation, vis_didge

Submodules (for more specific imports):

- **acoustical_simulation** – run TLM simulation, get_log_simulation_frequencies, get_notes
- **geo** – Geo (bore geometry)
- **conv** – note/frequency conversion (note_to_freq, cent_diff)
- **sim** – simulation backends (tlm_python, tlm_cython)
- **evo** – evolutionary shape search (Nuevolution, genomes, operators, loss)
- **shapes** – parametric shapes (e.g. KigaliShape.WebShape)
- **app** – application shell, config, logging
"""

from .geo import Geo
from .acoustical_simulation import (
    acoustical_simulation,
    get_log_simulation_frequencies,
    get_notes,
    compute_ground_spektrum,
)
from .visualize import (
    vis_didge,
    plot_bore,
    plot_impedance_spectrum,
    plot_notes,
    plot_geo_impedance_notes,
)
from .conv import (
    note_to_freq,
    freq_to_note,
    note_name,
    freq_to_note_and_cent,
    freq_to_wavelength,
    note_name_to_number,
    cent_diff,
)
from .shapes.KigaliShape import KigaliShape

# Evolution: genomes, loss, operators, evolution runner, callbacks
from .evo import (
    Genome,
    GeoGenome,
    GeoGenomeA,
    LossFunction,
    TestLossFunction,
    MutationOperator,
    CrossoverOperator,
    SimpleMutation,
    RandomMutation,
    SingleMutation,
    RandomCrossover,
    AverageCrossover,
    PartSwapCrossover,
    PartAverageCrossover,
    Nuevolution,
    AdaptiveProbabilities,
    NumpyEncoder,
    load_latest_evolution
)
from .evo.callbacks import init_standard_evolution

from .loss import (
    TairuaLoss,
    LossComponent,
    FrequencyTuningLoss,
    QFactorLoss,
    ModalDensityLoss,
    HarmonicSplittingLoss,
    IntegerHarmonicLoss,
    NearIntegerLoss,
    StretchedOddLoss,
    HighInharmonicLoss,
    ScaleTuningLoss,
    PeakQuantityLoss,
    PeakAmplitudeLoss,
    CompositeTairuaLoss,
)

__all__ = [
    "Geo",
    "acoustical_simulation",
    "get_log_simulation_frequencies",
    "get_notes",
    "compute_ground_spektrum",
    "vis_didge",
    "plot_bore",
    "plot_impedance_spectrum",
    "plot_notes",
    "plot_geo_impedance_notes",
    "note_to_freq",
    "freq_to_note",
    "note_name",
    "freq_to_note_and_cent",
    "freq_to_wavelength",
    "note_name_to_number",
    "cent_diff",
    "KigaliShape",
    "init_standard_evolution",
    "Genome",
    "GeoGenome",
    "GeoGenomeA",
    "LossFunction",
    "TestLossFunction",
    "MutationOperator",
    "CrossoverOperator",
    "SimpleMutation",
    "RandomMutation",
    "SingleMutation",
    "RandomCrossover",
    "AverageCrossover",
    "PartSwapCrossover",
    "PartAverageCrossover",
    "Nuevolution",
    "AdaptiveProbabilities",
    "NumpyEncoder",
    "load_latest_evolution",
    "TairuaLoss",
    "LossComponent",
    "FrequencyTuningLoss",
    "QFactorLoss",
    "ModalDensityLoss",
    "HarmonicSplittingLoss",
    "IntegerHarmonicLoss",
    "NearIntegerLoss",
    "StretchedOddLoss",
    "HighInharmonicLoss",
    "ScaleTuningLoss",
    "PeakQuantityLoss",
    "PeakAmplitudeLoss",
    "CompositeTairuaLoss",
]
