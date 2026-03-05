"""Test that Didgelab is installed correctly with the cython acoustical simulation
"""

from didgelab import Geo, acoustical_simulation
import numpy as np

geo = [[0,32], [800,32], [900,38], [970,42], [1050, 40], [1180, 48], [1350, 60], [1390, 68], [1500, 72]]
geo = Geo(geo=geo)
freq_grid = np.linspace(30, 1000, 50)
impedances = acoustical_simulation(geo, freq_grid, simulation_method="tlm_cython")

if impedances is not None:
    print("didgelab is installed correctly")