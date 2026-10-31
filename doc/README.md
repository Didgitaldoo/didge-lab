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
| [simulation_methods/acoustical_simulation/2d_fem.md](simulation_methods/acoustical_simulation/2d_fem.md) | Axisymmetric 2D FEM: r-weighted Helmholtz on the meridional half-plane, weak form, bent-centerline sweep, and the inner-wall shortcut correction for bent bores. |
| [simulation_methods/acoustical_simulation/3d_fem.md](simulation_methods/acoustical_simulation/3d_fem.md) | Full 3D FEM: tetrahedral Helmholtz on the swept bore volume, the only backend that predicts bent-bore pitch shifts from first principles without a fitting parameter. |
| [simulation_methods/acoustical_simulation/mesh_construction.md](simulation_methods/acoustical_simulation/mesh_construction.md) | How the 1D, 2D, and 3D FEM meshes are built from a bore profile (and centerline): line, structured triangulation, swept disk template, Bishop frame, prism-to-tet split. |

## Synthesis

| File | Description |
|------|-------------|
| [synthesizer.md](synthesizer.md) | The DidgeLab synthesizer (`didgelab.synth`), from a bore geometry to the sound of a played didgeridoo: transmission-line model and transfer function, beating lips, vocal tract, coupling, rendering, playing techniques, calibration on the Didge Archive, finding techniques in recordings. |
| [song_format.md](song_format.md) | The `.didge` song format: writing grooves and loops for the synthesizer, and the `didgelab-synth play` command. |
| [technique_detection.md](technique_detection.md) | Finding playing techniques (toots, stops, trills, vocalisation, circular breathing, tongue position) in recordings with `didgelab.synth.transcribe`: usage, method, accuracy, retraining the detector. |
| [archive_comparison.md](archive_comparison.md) | Comparing the synthesizer with the Didge Archive recordings: drone and toot pitch, hand slaps, harmonic fingerprint and spectral envelope of the drone. |
| [impedance_from_sound.md](impedance_from_sound.md) | Estimating the impedance spectrum of an unknown didgeridoo from a sound sample (`didgelab.synth.estimate_impedance`): method, accuracy on the Didge Archive, recording tips. |

## Examples: Synthesis

| File | Description |
|------|-------------|
| [examples/geometry_to_sound.ipynb](examples/geometry_to_sound.ipynb) | Geometry to sound: how a bore is played (Z_in and H, lip source, vocal tract, coupling, rendering, the effect of a bell). Introduction, maths, code examples with audio. |
| [examples/impedance_to_sound.ipynb](examples/impedance_to_sound.ipynb) | Impedance to sound: listening to a didgeridoo given only by its impedance peaks (`ModalBore`, `render_song_loop_from_peaks`, modal vs. fitted geometry). Introduction, maths, code examples with audio. |
| [examples/geometry_fit.ipynb](examples/geometry_fit.ipynb) | Geometry fit (`didgelab.synth.fit_geometry`): finding a bore for target impedance peaks. Introduction, maths (peak extraction, bore model, misfit, least squares), code examples (fitting, checking a target, listening), sound accuracy on the Didge Archive and the bell in the TLM. |
| [examples/technique_detection.ipynb](examples/technique_detection.ipynb) | Finding playing techniques in a synthetic and a real recording: lip tracking on the resonances, formant and tongue index, the learned detector, evaluation. |
| [examples/didge_archive_comparison.ipynb](examples/didge_archive_comparison.ipynb) | The synthesizer vs. the Didge Archive recordings: drone pitch, toots, spectra, harmonic fingerprint, third-octave envelope. |

## Examples

| File | Description |
|------|-------------|
| [examples/acoustical_simulations.ipynb](examples/acoustical_simulations.ipynb) | Compares all five acoustical simulation backends (tlm_python, tlm_cython, 1d_fem, 2d_fem, 3d_fem) and plots impedance spectra with execution-time benchmarks. |
| [examples/visualize_didgeridoos.ipynb](examples/visualize_didgeridoos.ipynb) | Demonstrates visualization tools: bore geometry, impedance spectrum, notes, and combined plots for single or multiple geometries. |
| [examples/conv_documentation.ipynb](examples/conv_documentation.ipynb) | Documents musical note and frequency conversion utilities (note numbers, names, Hz, wavelength, cent differences). |
| [examples/optimizer_benchmark.ipynb](examples/optimizer_benchmark.ipynb) | Benchmarks the two shape optimizers, Nuevolution (genetic algorithm) and CMA-ES, on several shapes and targets at equal loss-evaluation budget: final loss, convergence curves, evaluations needed. |
| [examples/least_squares_refinement.ipynb](examples/least_squares_refinement.ipynb) | Least-squares refinement (`refine_least_squares`): tuning the peaks of an evolved bore exactly with a gradient-based method after the evolution; refining a population; a short evolution plus refinement vs. a long evolution. |
| [examples/gradient_refinement.ipynb](examples/gradient_refinement.ipynb) | The experiment behind `refine_least_squares`: smoothness of the loss, finite-difference gradients, kinks and residuals, and a comparison of continued CMA-ES, L-BFGS-B and least squares after the evolution. |
| [examples/evolution_callbacks.ipynb](examples/evolution_callbacks.ipynb) | Explains the evolution callback system: generation/loss callbacks, pub/sub events, and built-in monitors (TqdmLossProgressCallback, EvolutionMonitor, SaveEvolution, EarlyStopping). |
| [examples/kigali_shape_parameters.ipynb](examples/kigali_shape_parameters.ipynb) | Describes the Kigali parametric shape: base taper, segment jitter, forced diameters, bell accent, and bubbles, with bore plots. |
| [examples/temperature_and_humidity.ipynb](examples/temperature_and_humidity.ipynb) | Shows how air temperature and humidity feed into AcousticConstants and how that shifts the impedance spectrum. |
| [examples/bent_shapes/fem2d_bent_didge.ipynb](examples/bent_shapes/fem2d_bent_didge.ipynb) | Bent-bore experiment in the 2D FEM: straight vs. curved centerline, mesh visualisation, and the inner-wall shortcut correction. |
| [examples/bent_shapes/fem3d_bent_didge.ipynb](examples/bent_shapes/fem3d_bent_didge.ipynb) | Bent-bore experiment in the 3D FEM: swept tetrahedral mesh, 3D bore visualisation, and parameter-free pitch-shift prediction. |

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
