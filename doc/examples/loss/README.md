# Loss functions

This folder contains notebooks that document the **modular loss API** for evolutionary optimization of didgeridoo shapes in DidgeLab.

## Structure

Each notebook follows the same format:

- **Title** – name of the loss or topic
- **Setup** – `sys.path` and imports (so each notebook can be run standalone)
- **Purpose** – what the loss optimizes for
- **Formula** – mathematical definition (LaTeX)
- **Symbols** – explanation of variables
- **Code** – minimal example to build and use the component

## Notebooks

| File | Content |
|------|---------|
| **00_overview.ipynb** | Introduction, setup, CompositeTairuaLoss, and index of all notebooks |
| **01_frequency_tuning.ipynb** | Frequency Tuning Loss |
| **02_scale_tuning.ipynb** | Scale Tuning Loss |
| **03_peak_quantity.ipynb** | Peak Quantity Loss |
| **04_peak_amplitude.ipynb** | Peak Amplitude Loss |
| **05_q_factor.ipynb** | Q-Factor Loss |
| **06_modal_density.ipynb** | Modal Density (Shimmer) Loss |
| **07_integer_harmonic.ipynb** | Integer Harmonic Loss |
| **08_near_integer.ipynb** | Near-Integer (Stretched) Loss |
| **09_stretched_odd.ipynb** | Stretched Odd Harmonics Loss |
| **10_high_inharmonic.ipynb** | High Inharmonic Loss |
| **11_harmonic_splitting.ipynb** | Harmonic Splitting Loss |
| **12_full_example.ipynb** | Full evolution example with composite loss |

## Running

From the repository root (`didge-lab`), start Jupyter and open any notebook. The notebooks set `sys.path.insert(0, "../../../src")` so that `didgelab` is importable when the working directory is `doc/examples/loss`.
