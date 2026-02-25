# Documentation Overview

This folder contains documentation for DidgeLab: computer-aided design of didgeridoos. Below is a list of all documentation files with links and brief descriptions. The best place to get started are the tutorials.

| File | Description |
|------|-------------|
| [release.md](release.md) | Instructions for making a DidgeLab release: version bump, tests, git tagging, and PyPI upload. |

## Simulation Methods

| File | Description |
|------|-------------|
| [simulation_methods/acoustical_simulation/transmission_line_model.md](simulation_methods/acoustical_simulation/transmission_line_model.md) | Explains the transmission-line model: transfer matrices, viscothermal losses, conical vs. cylindrical geometry, radiation impedance, and input impedance. |
| [simulation_methods/acoustical_simulation/1d_fem.md](simulation_methods/acoustical_simulation/1d_fem.md) | Describes the 1D finite element method: Helmholtz equation, weak form, discretization, damping, and boundary conditions. |

## Examples

| File | Description |
|------|-------------|
| [examples/acoustical_simulations.ipynb](examples/acoustical_simulations.ipynb) | Compares acoustical simulation methods (tlm_python, tlm_cython, 1d_fem) and plots impedance spectra. |
| [examples/visualize_didgeridoos.ipynb](examples/visualize_didgeridoos.ipynb) | Demonstrates visualization tools: bore geometry, impedance spectrum, notes, and combined plots for single or multiple geometries. |
| [examples/conv_documentation.ipynb](examples/conv_documentation.ipynb) | Documents musical note and frequency conversion utilities (note numbers, names, Hz, wavelength, cent differences). |
| [examples/evolution_callbacks.ipynb](examples/evolution_callbacks.ipynb) | Explains the evolution callback system: generation/loss callbacks, pub/sub events, and built-in monitors (TqdmLossProgressCallback, EvolutionMonitor, SaveEvolution, EarlyStopping). |
| [examples/kigali_shape_parameters.ipynb](examples/kigali_shape_parameters.ipynb) | Describes the Kigali parametric shape: base taper, segment jitter, forced diameters, bell accent, and bubbles, with bore plots. |

## Examples: Tutorials

| File | Description |
|------|-------------|
| [examples/tutorials/tutorial1.ipynb](examples/tutorials/tutorial1.ipynb) | Introduction: didgeridoo geometry, Geo class, visualization, acoustical simulation, and the conversion toolkit. |
| [examples/tutorials/tutorial2.ipynb](examples/tutorials/tutorial2.ipynb) | Parametric shapes: genome, genome2geo(), Kigali shape parameters, and creating geometries from parameter vectors. |
| [examples/tutorials/tutorial3.ipynb](examples/tutorials/tutorial3.ipynb) | Computational evolution: loss functions, mutation, generations, mutant pool, and a practical evolution example. |

## Examples: Loss

| File | Description |
|------|-------------|
| [examples/loss/README.md](examples/loss/README.md) | Overview of the modular loss API and index of all loss notebooks. |
| [examples/loss/00_overview.ipynb](examples/loss/00_overview.ipynb) | Introduction to CompositeTairuaLoss and index of all loss notebooks. |
| [examples/loss/01_frequency_tuning.ipynb](examples/loss/01_frequency_tuning.ipynb) | Frequency Tuning Loss — optimizes target peak frequencies. |
| [examples/loss/02_scale_tuning.ipynb](examples/loss/02_scale_tuning.ipynb) | Scale Tuning Loss — matches resonances to a chosen scale. |
| [examples/loss/03_peak_quantity.ipynb](examples/loss/03_peak_quantity.ipynb) | Peak Quantity Loss — penalizes too many or too few peaks. |
| [examples/loss/04_peak_amplitude.ipynb](examples/loss/04_peak_amplitude.ipynb) | Peak Amplitude Loss — shapes peak heights. |
| [examples/loss/05_q_factor.ipynb](examples/loss/05_q_factor.ipynb) | Q-Factor Loss — controls resonance sharpness (Q factor). |
| [examples/loss/06_modal_density.ipynb](examples/loss/06_modal_density.ipynb) | Modal Density (Shimmer) Loss — encourages uniform spacing of modes. |
| [examples/loss/07_integer_harmonic.ipynb](examples/loss/07_integer_harmonic.ipynb) | Integer Harmonic Loss — favors integer-ratio harmonics. |
| [examples/loss/08_near_integer.ipynb](examples/loss/08_near_integer.ipynb) | Near-Integer (Stretched) Loss — allows slightly stretched harmonics. |
| [examples/loss/09_stretched_odd.ipynb](examples/loss/09_stretched_odd.ipynb) | Stretched Odd Harmonics Loss — targets stretched odd harmonic series. |
| [examples/loss/10_high_inharmonic.ipynb](examples/loss/10_high_inharmonic.ipynb) | High Inharmonic Loss — penalizes inharmonic content in upper range. |
| [examples/loss/11_harmonic_splitting.ipynb](examples/loss/11_harmonic_splitting.ipynb) | Harmonic Splitting Loss — addresses split or doubled peaks. |
| [examples/loss/12_full_example.ipynb](examples/loss/12_full_example.ipynb) | Full evolution example combining multiple loss components. |
