#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Dictionary with parameters specific for each planet
"""

import numpy as np


#Planetary constants
#################################################################################################

planet_const = {

    'Mars':{
        'Radius': 3397.2e3,              # Mars radius (m)
        'Mass': 6.4169e23,               #Mass (kg)
        'daytosec': 88775.,              # Length of a day (s)
        'rotrate': 2.0*np.pi/88775.,     # Rotation rate (rad s-1)
        'g0': 3.72,                      # Gravity at the surface (m s-2)
        'mugaz': 43.49,                  # Atmosphere molar mars (g mol-1)
        'yeartoday': 668.6,              # Martian year (days)
        'd_perihelion': 1.381436,        # Minimum Sun-Mars distance (AU)
        'd_aphelion': 1.66593,           # Maximum Sun-Mars distance (AU)
        'day_perihelion': 485.,          # Perihelion date (days since Ls=0 - N. Spring)
        'obliquity': 25.2,               # Obliquity (deg)
    }

}


#Diffusion coefficients
###################################################################################################

diffusion_coeff = {

    'Mars':{

        "39": {
            "name": "H2",
            "A": 2.23e17,
            "s": 0.75,
            "Btherm": -0.25,
            },
        "48": {
            "name": "H",
            "A": 8.4e17,
            "s": 0.597,
            "Btherm": -0.25,
            },
        "40": {
            "name": "He",
            "A": 2.7e17,
            "s": 0.72,
            "Btherm": 0.0,  #Don't know the value, some assumed zero
            },
        "45": {
            "name": "O",
            "A": 0.92e17,
            "s": 0.75,
            "Btherm": 0.0,  #Don't know the value, some assumed zero
            },

    },
}

#Polarizabilities
###################################################################################################

polrizabilities = {

    'Mars':{

        "45": {
            "name": "O",
            "beta": 0.79e-24,   #Krasnopolsky book
            },
        "133": {
            "name": "O(1D)",
            "beta": 0.79e-24,   #Same as O
            },
        "22": {
            "name": "N2",
            "beta": 1.76e-24,   #Krasnopolsky book
            },
        "2": {
            "name": "CO2",
            "beta": 2.6e-24,   #Krasnopolsky book
            },
        "39": {
            "name": "H2",
            "beta": 0.82e-24,   #Krasnopolsky book
            },
        "48": {
            "name": "H",
            "beta": 0.67e-24,   #Krasnopolsky book
            },
        "40": {
            "name": "He",
            "beta": 0.21e-24,   #Krasnopolsky book
            },
        "76": {
            "name": "Ar",
            "beta": 1.66e-24,   #Cangi
            },
        "46": {
            "name": "C",
            "beta": 1.76e-24,   #Cangi
            },
        "5": {
            "name": "CO",
            "beta": 1.953e-24,  #Cangi
            },
        "1": {
            "name": "H2O",
            "beta": 1.50e-24,   #Cangi
            },
        "47": {
            "name": "N",
            "beta": 1.1e-24,   #Cangi
            },
        "134": {
            "name": "N(2D)",
            "beta": 1.1e-24,   #Same as N
            },
        "4": {
            "name": "N2O",
            "beta": 3.00e-24,  #Cangi
            },
        "8": {
            "name": "NO",
            "beta": 1.70e-24,   #Cangi
            },
        "10": {
            "name": "NO2",
            "beta": 2.91e-24,   #Krasnopolsky book
            },
        "7": {
            "name": "O2",
            "beta": 1.59e-24,   #Cangi
            },
        "3": {
            "name": "O3",
            "beta": 3.08e-24,   #Cangi
            },
    },
}

#Upper boundary conditions
#######################################################################################################

#Upper boundary conditions (Type 1 - Fixed density (m-3) ; Type 2 - Fixed flux (m-2 s-1) ; Type 3 - Fixed velocity (m s-1) ; Type 4 - Effusion velocity)
#If species is not present it is assumed to be Type 2 with flux = 0.0

upper_bc = {

    'Mars':{

        "39": {
            "name": "H2",
            "type": 3,
            "value": 3.4e1/100., #m s-1
            #"value": 0.0/100., #m s-1
                },
        "45": {
            "name": "O",
            "type": 2,
            "value": 1.2e8*1.0e4, #m-2 s-1
            #"value": 0.0*1.0e4, #m-2 s-1
                },
        "48": {
            "name": "H",
            "type": 3,
            "value": 3.1e3/100., #m s-1
            #"value": 0.0*1.0e4, #m-2 s-1
                },

    }
}

#Lower boundary conditions
#######################################################################################################

#Lower boundary conditions (Type 1 - Fixed density (m-3) ; Type 2 - Fixed flux (m-2 s-1) ; Type 3 - Fixed velocity (m s-1))
#If species is not present it is assumed to be Type 2 with flux = 0.0

lower_bc = {

    'Mars':{
        "1": {
            "name": "H2O",
            "type": 2,
            "value": 0.0,   #m-2 s-1
                },
        "76": {
            "name": "Ar",
            "type": 2,
            "value": 0.0,   #density given by input profile
                },
        "22": {
            "name": "N2",
            "type": 2,
            "value": 0.0,   #density given by input profile
                },
        "2": {
            "name": "CO2",
            "type": 2,
            "value": 0.0,   #density given by input profile
                },
    }
}