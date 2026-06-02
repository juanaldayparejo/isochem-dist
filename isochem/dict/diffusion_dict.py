#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Dictionary of diffusion parameters for different gases
"""

import numpy as np
from numba import njit,types
from numba.typed import Dict

#########################################################################
#POLARIZABILITY
#########################################################################

#Creating structured array
polarizability_dtype = np.dtype([
    ('gasid', np.int64),
    ('polarizability', np.float64),
])


polarizability = np.array([

    #O
    (45, 0.79e-24),  #Krasnopolsky book

    #O(1D)
    (133, 0.79e-24),  #Krasnopolsky book

    #N2
    (22, 1.76e-24), #Krasnopolsky book

    #CO2
    (2, 2.6e-24), #Krasnopolsky book

    #H2
    (39, 0.82e-24), #Krasnopolsky book

    #H
    (48, 0.67e-24), #Krasnopolsky book

    #He
    (40, 0.21e-24), #Krasnopolsky book

    #Ar
    (76, 1.64e-24), #Cangi

    #C
    (46, 1.76e-24), #Cangi

    #CO
    (5, 1.953e-24), #Cangi

    #H2O
    (1, 1.50e-24), #Cangi

    #N
    (47, 1.1e-24), #Cangi

    #N(2D)
    (132, 1.1e-24), #Same as N

    #N2O
    (4, 3.0e-24), #Cangi

    #NO
    (8, 1.70e-24), #Cangi

    #NO2
    (10, 2.91e-24), #Cangi

    #O2
    (7, 1.59e-24), #Cangi

    #O3
    (9, 3.08e-24), #Cangi

], dtype=polarizability_dtype)


MAX_GASID = 3000
INVALID_INDEX = MAX_GASID

polarizability_index = np.full((MAX_GASID + 1), INVALID_INDEX, dtype=np.int64)

for i in range(len(polarizability)):
    gasid = polarizability[i]['gasid']
    polarizability_index[gasid] = i


#Useful functions for the dictionary
@njit
def get_polarizability(gasid):
    """
    Return the polarizability of the specified species
    """
    idx = polarizability_index[gasid]

    if idx == INVALID_INDEX:
        return 0.0  #Return default value if the species is not found in the dictionary

    return polarizability[idx]['polarizability']


#########################################################################
# MOLECULAR DIFFUSION COEFFICIENTS
#########################################################################

#Creating structured array
moldiff_dtype = np.dtype([
    ('gasid', np.int64),
    ('A', np.float64),
    ('s', np.float64),
])

moldiff_co2 = np.array([

    #H2
    (39, 2.23e17, 0.75), #Krasnopolsky book

    #H
    (48, 8.4e17, 0.597), #Krasnopolsky book

    #He
    (40, 2.7e17, 0.72),  #Krasnopolsky book

    #O
    (45, 0.92e17, 0.75),  #Krasnopolsky book

    #O(1D)
    (133, 0.92e17, 0.75),  #Same as O

], dtype=moldiff_dtype)


moldiff_index_co2 = np.full((MAX_GASID + 1), MAX_GASID, dtype=np.int64)

for i in range(len(moldiff_co2)):
    gasid = moldiff_co2[i]['gasid']
    moldiff_index_co2[gasid] = i


#Useful functions for the dictionary
@njit
def get_molecular_diffusion_parameters_co2(gasid):
    """
    Return the molecular diffusion parameters of the specified species
    """
    idx = moldiff_index_co2[gasid]

    if idx == INVALID_INDEX:
        return 1.0e17, 0.75  #Return default values if the species is not found in the dictionary

    return moldiff_co2[idx]['A'], moldiff_co2[idx]['s']

#########################################################################
# THERMAL DIFFUSION COEFFICIENTS
#########################################################################

#Creating structured array
thermdiff_dtype = np.dtype([
    ('gasid', np.int64),
    ('alpha', np.float64),
])

thermdiff = np.array([

    #H2
    (39, -0.25), #Krasnopolsky book

    #H
    (48, -0.25), #Krasnopolsky book

    #He
    (40, -0.25),  #Krasnopolsky book

    #H2+
    (1039, -0.25), #Krasnopolsky book

    #H+
    (1048, -0.25), #Krasnopolsky book

    #He+
    (1040, -0.25),  #Krasnopolsky book

], dtype=thermdiff_dtype)


thermdiff_index = np.full((MAX_GASID + 1), INVALID_INDEX, dtype=np.int64)

for i in range(len(thermdiff)):
    gasid = thermdiff[i]['gasid']
    thermdiff_index[gasid] = i


#Useful functions for the dictionary
@njit
def get_thermal_diffusion_coefficient(gasid):
    """
    Return the thermal diffusion coefficient of the specified species
    """
    idx = thermdiff_index[gasid]

    if idx == INVALID_INDEX:
        return 0.0  #Return zero if the species is not found in the dictionary

    return thermdiff[idx]['alpha']
