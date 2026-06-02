import numpy as np
from isochem.jit import jit

MAX_PRODUCTS = 4
MAX_REACTANTS = 4

reaction_network_dtype = np.dtype([
    #Reaction ID
    ("id", np.int32),

    #Reaction type (isochem)
    ("rtype", np.int32),

    #Reactants information
    ("nreactants", np.int32),
    ("reactant_ids", np.int32, MAX_REACTANTS),
    ("reactant_iso_ids", np.int32, MAX_REACTANTS),
    ("reactant_numbers", np.float64, MAX_REACTANTS),

    #Products information
    ("nproducts", np.int32),
    ("product_ids", np.int32, MAX_PRODUCTS),
    ("product_iso_ids", np.int32, MAX_PRODUCTS),
    ("product_numbers", np.float64, MAX_PRODUCTS),

    #Reaction rates formulation
    ("ratetype", np.int32),   #0 = BIMOLECULAR ; 1 = TERMOLECULAR ; 2 = CHEMICAL ACTIVATION ; 3 = IONIC

    #Bimolecular reaction parameters (or ionic reaction parameters, which have the same formulation as bimolecular reactions)
    ("alpha", np.float64),
    ("n", np.float64),
    ("gamma", np.float64),
    ("branching", np.float64),

    #Termolecular reaction parameters
    ("k0", np.float64),
    #("n", np.float64),
    ("kinf", np.float64),
    ("m", np.float64),

    #Chemical activation reaction parameters
    ("A", np.float64),  # Pre-exponential factor for chemical activation reactions
    ("B", np.float64),  # Temperature exponent for chemical activation reactions

    #Ambient density
    ("ambient_id", np.int32), #None = no ambient ; 0 = all gases ; >0 = specific gas ID (e.g., 7 for O2, 45 for O, etc)

    #Reference for the reaction rate coefficients (e.g., JPL, IUPAC, etc)
    ("ref", 'U50'),

])


#########################################################################################################################

MAX_REACTIONS = 10000
reaction_network = np.empty(MAX_REACTIONS, dtype=reaction_network_dtype)

NONE_ID = np.int32(-999999)
NONE_VALUE = np.nan


# Initialize entire array once at the start
reaction_network["id"].fill(NONE_ID)
reaction_network["rtype"].fill(NONE_ID)
reaction_network["nreactants"].fill(NONE_ID)
reaction_network["nproducts"].fill(NONE_ID)
reaction_network["reactant_ids"].fill(NONE_ID)
reaction_network["reactant_iso_ids"].fill(NONE_ID)
reaction_network["reactant_numbers"].fill(NONE_VALUE)
reaction_network["product_ids"].fill(NONE_ID)
reaction_network["product_iso_ids"].fill(NONE_ID)
reaction_network["product_numbers"].fill(NONE_VALUE)

#########################################################################################################################

#Reaction 1: O + O2 + CO2 -> O3 + CO2

i = 1
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 7, NONE_ID, NONE_ID]  #O + O2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [3, NONE_ID, NONE_ID, NONE_ID] #O3
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0 #Bimolecular reaction

reaction_network[i]["alpha"] = 6.0e-34 * 2.075
reaction_network[i]["n"] = -2.4
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = 2 #The rate of this reaction depends on the total ambient density

reaction_network[i]["ref"] = "sehested et al., j. geophys. res., 100, 1995"


#########################################################################################################################

#Reaction 2: O + O + CO2 -> O2 + CO2

i = 2
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 2

reaction_network[i]["nreactants"] = 1
reaction_network[i]["reactant_ids"] = [45, NONE_ID, NONE_ID, NONE_ID]  #O + O
reaction_network[i]["reactant_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [7, NONE_ID, NONE_ID, NONE_ID] #O2
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0 #Bimolecular reaction

reaction_network[i]["alpha"] = 9.46e-34 * 2.5
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -485.
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = 2 #The rate of this reaction depends on the total ambient density

reaction_network[i]["ref"] = 'NIST kinetics database'

#########################################################################################################################

#Reaction 3: O + O3 -> O2 + O2

i = 3
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 3, NONE_ID, NONE_ID]  # O + O3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [7, NONE_ID, NONE_ID, NONE_ID]  # O2
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 8.0e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 2060.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID  # No ambient dependence

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 4: O(1D) + CO2 -> O + CO2

i = 4
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 2, NONE_ID, NONE_ID]  # O(1D) + CO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [45, 2, NONE_ID, NONE_ID]  # O + CO2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 7.5e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -115.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 5: O(1D) + H2O -> OH + OH

i = 5
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 1, NONE_ID, NONE_ID]  # O(1D) + H2O
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [13, NONE_ID, NONE_ID, NONE_ID]  # 2 OH
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.63e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -60.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 6: O(1D) + H2 -> OH + H

i = 6
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 39, NONE_ID, NONE_ID]  # O(1D) + H2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 48, NONE_ID, NONE_ID]  # OH + H
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.2e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 7: O(1D) + O2 -> O + O2

i = 7
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 7, NONE_ID, NONE_ID]  # O(1D) + O2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [45, 7, NONE_ID, NONE_ID]  # O + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.3e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -55.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 8: O(1D) + O3 -> 2 O2

i = 8
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 3, NONE_ID, NONE_ID]  # O(1D) + O3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [7, NONE_ID, NONE_ID, NONE_ID]  # 2 O2
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.4e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 0.5

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 9: O(1D) + O3 -> O2 + O + O

i = 9
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 3, NONE_ID, NONE_ID]  # O(1D) + O3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [7, 45, NONE_ID, NONE_ID]  # O2 + 2 O
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 2.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.4e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 0.5

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 10: O + HO2 -> OH + O2

i = 10
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 44, NONE_ID, NONE_ID]  # O + HO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 7, NONE_ID, NONE_ID]  # OH + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.0e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -200.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 11: O + OH -> O2 + H

i = 11
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 13, NONE_ID, NONE_ID]  # O + OH
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [7, 48, NONE_ID, NONE_ID]  # O2 + H
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.8e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -180.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 12: H + O3 -> OH + O2

i = 12
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [48, 3, NONE_ID, NONE_ID]  # H + O3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 7, NONE_ID, NONE_ID]  # OH + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.4e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 470.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 13: H + HO2 -> 2 OH

i = 13
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [48, 44, NONE_ID, NONE_ID]  # H + HO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [13, NONE_ID, NONE_ID, NONE_ID]  # 2 OH
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 7.2e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 14: H + HO2 -> H2 + O2

i = 14
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [48, 44, NONE_ID, NONE_ID]  # H + HO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [39, 7, NONE_ID, NONE_ID]  # H2 + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 6.9e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 15: H + HO2 -> H2O + O

i = 15
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [48, 44, NONE_ID, NONE_ID]  # H + HO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1, 45, NONE_ID, NONE_ID]  # H2O + O
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.6e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 16: OH + HO2 -> H2O + O2

i = 16
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 44, NONE_ID, NONE_ID]  # OH + HO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1, 7, NONE_ID, NONE_ID]  # H2O + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 4.8e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -250.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 17: HO2 + HO2 -> H2O2 + O2

i = 17
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 2

reaction_network[i]["nreactants"] = 1
reaction_network[i]["reactant_ids"] = [44, NONE_ID, NONE_ID, NONE_ID]  # 2 HO2
reaction_network[i]["reactant_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [25, 7, NONE_ID, NONE_ID]  # H2O2 + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.0e-13
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -460.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 18: OH + H2O2 -> H2O + HO2

i = 18
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 25, NONE_ID, NONE_ID]  # OH + H2O2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1, 44, NONE_ID, NONE_ID]  # H2O + HO2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.8e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 19: OH + H2 -> H2O + H

i = 19
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 39, NONE_ID, NONE_ID]  # OH + H2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1, 48, NONE_ID, NONE_ID]  # H2O + H
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.8e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 1800.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 20: H + O2 + CO2 -> HO2 + CO2

i = 20
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [48, 7, NONE_ID, NONE_ID]  # OH + H2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [44, NONE_ID, NONE_ID, NONE_ID]  # H2O + H
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_ID, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 1  # Termolecular reaction

reaction_network[i]["k0"] = 2.4 * 5.3e-32
reaction_network[i]["n"] = 1.8
reaction_network[i]["kinf"] = 9.5e-11
reaction_network[i]["m"] = -0.4

reaction_network[i]["ambient_id"] = 2

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 21: O + H2O2 -> OH + HO2

i = 21
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 25, NONE_ID, NONE_ID]  # O + H2O2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 44, NONE_ID, NONE_ID]  # OH + HO2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.4e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 2000.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 22: OH + OH -> H2O + O

i = 22
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 2

reaction_network[i]["nreactants"] = 1
reaction_network[i]["reactant_ids"] = [13, NONE_ID, NONE_ID, NONE_ID]  # 2 OH
reaction_network[i]["reactant_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1, 45, NONE_ID, NONE_ID]  # H2O + O
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.8e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 23: OH + O3 -> HO2 + O2

i = 23
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 3, NONE_ID, NONE_ID]  # OH + O3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [44, 7, NONE_ID, NONE_ID]  # HO2 + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.7e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 940.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 24: HO2 + O3 -> OH + 2 O2

i = 24
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [44, 3, NONE_ID, NONE_ID]  # HO2 + O3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 7, NONE_ID, NONE_ID]  # OH + 2 O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 2.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.0e-14
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 490.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 25: HO2 + HO2 + CO2 -> H2O2 + O2 + CO2

i = 25
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 2

reaction_network[i]["nreactants"] = 1
reaction_network[i]["reactant_ids"] = [44, NONE_ID, NONE_ID, NONE_ID]  # 2 HO2
reaction_network[i]["reactant_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [25, 7, NONE_ID, NONE_ID]  # H2O2 + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.5 * 2.1e-33
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -920.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = 2

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 26: OH + OH + CO2 -> H2O2 + CO2

i = 26
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 2

reaction_network[i]["nreactants"] = 1
reaction_network[i]["reactant_ids"] = [13, NONE_ID, NONE_ID, NONE_ID]  # 2 OH
reaction_network[i]["reactant_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [25, NONE_ID, NONE_ID, NONE_ID]  # H2O2
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 1  # Termolecular reaction

reaction_network[i]["k0"] = 2.5 * 6.9e-31
reaction_network[i]["n"] = 1.0
reaction_network[i]["kinf"] = 2.6e-11
reaction_network[i]["m"] = 0.0

reaction_network[i]["ambient_id"] = 2

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 27: H + H + CO2 -> H2 + CO2

i = 27
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 2

reaction_network[i]["nreactants"] = 1
reaction_network[i]["reactant_ids"] = [48, NONE_ID, NONE_ID, NONE_ID]  # 2 H
reaction_network[i]["reactant_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [39, NONE_ID, NONE_ID, NONE_ID]  # H2
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.5 * 1.8e-30 / 298.  #Factor of 2.5 for CO2 atmosphere; Factor of 1/298 to correct expression from Mars PCM to bimolecular
reaction_network[i]["n"] = -1.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = 2

reaction_network[i]["ref"] = 'Baulch et al., 2005'

#########################################################################################################################

#Reaction 28: O + NO2 + M -> NO + O2 + M

i = 28
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 10, NONE_ID, NONE_ID]  # O + NO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [8, 7, NONE_ID, NONE_ID]  # NO + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 2  # Chemical activation reaction

reaction_network[i]["k0"] = 2.5 * 3.4e-31
reaction_network[i]["n"] = 1.6
reaction_network[i]["kinf"] = 2.3e-11
reaction_network[i]["m"] = 0.2
reaction_network[i]["A"] = 5.3e-12
reaction_network[i]["B"] = -200.0

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 29: NO + O3 -> NO2 + O2

i = 29
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [8, 3, NONE_ID, NONE_ID]  # NO + O3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [10, 7, NONE_ID, NONE_ID]  # NO2 + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.0e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 1500.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2006'


#########################################################################################################################

#Reaction 30: NO + HO2 -> NO2 + OH

i = 30
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [8, 44, NONE_ID, NONE_ID]  # NO + HO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [10, 13, NONE_ID, NONE_ID]  # NO2 + OH
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.44e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -260.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2011'


#########################################################################################################################

#Reaction 31: N + NO -> N2 + O

i = 31
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [47, 8, NONE_ID, NONE_ID]  # N + NO
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [22, 45, NONE_ID, NONE_ID]  # N2 + O
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.1e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -100.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 32: N + O2 -> NO + O

i = 32
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [47, 7, NONE_ID, NONE_ID]  # N + O2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [8, 45, NONE_ID, NONE_ID]  # NO + O
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.3e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 3150.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 33: NO2 + H -> NO + OH

i = 33
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [10, 48, NONE_ID, NONE_ID]  # NO2 + H
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [8, 13, NONE_ID, NONE_ID]  # NO + OH
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.35e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 34: N + O -> NO

i = 34
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [47, 45, NONE_ID, NONE_ID]  # N + O
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [8, NONE_ID, NONE_ID, NONE_ID]  # NO
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.8e-17
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2011'


#########################################################################################################################

#Reaction 35: N + HO2 -> NO + OH

i = 35
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [47, 44, NONE_ID, NONE_ID]  # N + HO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [8, 13, NONE_ID, NONE_ID]  # NO + OH
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.19e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'brune et al., j. chem. phys., 87, 1983'

#########################################################################################################################

#Reaction 36: N + OH -> NO + H

i = 36
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [47, 13, NONE_ID, NONE_ID]  # N + OH
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [8, 48, NONE_ID, NONE_ID]  # NO + H
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.8e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -85.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'atkinson et al., j. phys. chem. ref. data, 18, 881, 1989'


#########################################################################################################################

#Reaction 37: N(2D) + O -> N + O

i = 37
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [134, 45, NONE_ID, NONE_ID]  # N(2D) + O
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [47, 45, NONE_ID, NONE_ID]  # N + O
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.3e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 260.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'herron, j. phys. chem. ref. data, 1999'


#########################################################################################################################

#Reaction 38: N(2D) + N2 -> N + N2

i = 38
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [134, 22, NONE_ID, NONE_ID]  # N(2D) + N2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [47, 22, NONE_ID, NONE_ID]  # N + N2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.7e-14
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'herron, j. phys. chem. ref. data, 1999'


#########################################################################################################################

#Reaction 39: N(2D) + CO2 -> NO + CO

i = 39
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [134, 2, NONE_ID, NONE_ID]  # N(2D) + CO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [8, 5, NONE_ID, NONE_ID]  # NO + CO
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.6e-13
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'herron, j. phys. chem. ref. data, 1999'

#########################################################################################################################

#Reaction 40: OH + CO -> CO2 + H

i = 40
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 5, NONE_ID, NONE_ID]  # OH + CO
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [2, 48, NONE_ID, NONE_ID]  # CO2 + H
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 2  # Chemical activation reaction

reaction_network[i]["k0"] = 2.5 * 6.9e-33
reaction_network[i]["n"] = 2.1
reaction_network[i]["kinf"] = 1.1e-12
reaction_network[i]["m"] = -1.3
reaction_network[i]["A"] = 1.85e-13
reaction_network[i]["B"] = 65.0

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 41: OH + CO + M -> HOCO + M

i = 41
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 5, NONE_ID, NONE_ID]  # OH + CO
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [80, NONE_ID, NONE_ID, NONE_ID]  # HOCO
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 1  # Termolecular reaction

reaction_network[i]["k0"] = 2.5 * 6.9e-33
reaction_network[i]["n"] = 2.1
reaction_network[i]["kinf"] = 1.1e-12
reaction_network[i]["m"] = -1.3

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 42: O + CO + M -> CO2 + M

i = 42
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 5, NONE_ID, NONE_ID]  # O + CO
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [2, NONE_ID, NONE_ID, NONE_ID]  # CO2
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.5 * 6.5e-33
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 2184.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = 'tsang and hampson, 1986'


#########################################################################################################################

#Reaction 43: O(1D) + N2 + M -> N2O + M

i = 43
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 22, NONE_ID, NONE_ID]  # O(1D) + N2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [4, NONE_ID, NONE_ID, NONE_ID]  # N2O
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.8e-36
reaction_network[i]["n"] = -0.9
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 44: O + NO + CO2 -> NO2 + CO2

i = 44 
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 8, NONE_ID, NONE_ID]  # O + NO
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [10, NONE_ID, NONE_ID, NONE_ID]  # NO2
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 1  # Termolecular reaction

reaction_network[i]["k0"] = 2.5 * 9.1e-32
reaction_network[i]["n"] = 1.5
reaction_network[i]["kinf"] = 3.0e-11
reaction_network[i]["m"] = 0.0

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = 'JPL 2020'


#########################################################################################################################

#Reaction 45: O(1D) + N2 -> O + N2

i = 45
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 22, NONE_ID, NONE_ID]  # O(1D) + N2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [45, 22, NONE_ID, NONE_ID]  # O + N2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.15e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 110.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 46: O(1D) + N2O -> N2 + O2

i = 46
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 4, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [22, 7, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 1.19e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -20.0
reaction_network[i]["branching"] = 0.39

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 47: O(1D) + N2O -> NO + NO

i = 47
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [133, 4, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [8, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 1.19e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -20.0
reaction_network[i]["branching"] = 0.61

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 48: O + NO2 + M -> NO3 + M

i = 48
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 10, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [91, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 1

reaction_network[i]["k0"] = 2.5 * 3.4e-31
reaction_network[i]["n"] = 1.6
reaction_network[i]["kinf"] = 2.3e-11
reaction_network[i]["m"] = 0.2

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 49: O + NO3 -> O2 + NO2

i = 49
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 91, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [7, 10, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 1.3e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 50: N + NO2 -> N2O + O

i = 50
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [47, 10, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [4, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 5.8e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -220.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 51: NO + NO3 -> NO2 + NO2

i = 51
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [8, 91, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [10, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 1.7e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -125.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 52: NO2 + O3 -> NO3 + O2

i = 52
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [10, 3, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [91, 7, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 1.2e-13
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 2450.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 53: NO3 + NO3 -> 2NO2 + O2

i = 53
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 2

reaction_network[i]["nreactants"] = 1
reaction_network[i]["reactant_ids"] = [91, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [10, 7, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 8.5e-13
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 2450.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 54: O2 + HOCO -> HO2 + CO2

i = 54
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [7, 80, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [44, 2, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 2.0e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'JPL 2020'

#########################################################################################################################

#Reaction 55: O + H2 -> OH + H

i = 55
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [45, 39, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 1.6e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 4570.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press.'

###############################################################################################################################

# Reaction 56: N + O3 -> NO + O2

i = 56
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [47, 3, NONE_ID, NONE_ID]  # N + O3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [8, 7, NONE_ID, NONE_ID]  # NO + O2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.0e-16
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

###############################################################################################################################

# Reaction 57: N(2D) + NO -> N2 + O

i = 57
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [134, 8, NONE_ID, NONE_ID]  # N(2D) + NO
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [22, 45, NONE_ID, NONE_ID]  # N2 + O
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 6.7e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Fox and Sung, 2001 (based on Fell et al. 1990)"

###############################################################################################################################

# Reaction 58: H + NO3 -> OH + NO2

i = 58
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [48, 91, NONE_ID, NONE_ID]  # H + NO3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 10, NONE_ID, NONE_ID]  # OH + NO2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.1e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

###############################################################################################################################

# Reaction 59: OH + NO + M -> HONO + M

i = 59
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 8, NONE_ID, NONE_ID]  # OH + NO
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [138, NONE_ID, NONE_ID, NONE_ID]  # HONO
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 1  # Termolecular reaction

reaction_network[i]["k0"] = 7.1e-31
reaction_network[i]["n"] = 2.6
reaction_network[i]["kinf"] = 3.6e-11
reaction_network[i]["m"] = 0.1

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = "JPL 2020"

###############################################################################################################################

# Reaction 60: OH + NO2 + M -> HNO3 + M

i = 60
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 10, NONE_ID, NONE_ID]  # OH + NO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [12, NONE_ID, NONE_ID, NONE_ID]  # HNO3
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 1  # Termolecular reaction

# Channel 1: HONO2
reaction_network[i]["k0"] = 1.8e-30
reaction_network[i]["n"] = 3.0
reaction_network[i]["kinf"] = 2.8e-11
reaction_network[i]["m"] = 0.0

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = "JPL 2020"

###############################################################################################################################

# Reaction 61: OH + NO3 -> HO2 + NO2

i = 61
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 91, NONE_ID, NONE_ID]  # OH + NO3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [44, 10, NONE_ID, NONE_ID]  # HO2 + NO2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.0e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "JPL 2020"

###############################################################################################################################

# Reaction 62: OH + HONO -> H2O + NO2

i = 62
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 138, NONE_ID, NONE_ID]  # OH + HONO
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1, 10, NONE_ID, NONE_ID]  # H2O + NO2
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.0e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -250.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "JPL 2020"

###############################################################################################################################

# Reaction 63: OH + HNO3 -> H2O + NO3

i = 63
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 12, NONE_ID, NONE_ID]  # OH + HNO3
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1, 91, NONE_ID, NONE_ID]  # H2O + NO3
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 7.2e-15
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -785.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

###############################################################################################################################

# Reaction 64: OH + HO2NO2 -> H2O + NO2 + O2

i = 64
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [13, 137, NONE_ID, NONE_ID]  # OH + HO2NO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 3
reaction_network[i]["product_ids"] = [1, 10, 7, NONE_ID]  # H2O + NO2 + O2
reaction_network[i]["product_iso_ids"] = [0, 0, 0, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, 1.0, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 4.5e-13
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = -610.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "JPL 2020"

###############################################################################################################################

# Reaction 65: HO2 + NO2 + M -> HO2NO2 + M

i = 65
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [44, 10, NONE_ID, NONE_ID]  # HO2 + NO2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [137, NONE_ID, NONE_ID, NONE_ID]  # HO2NO2
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 1  # Termolecular reaction

reaction_network[i]["k0"] = 1.9e-31
reaction_network[i]["n"] = 3.4
reaction_network[i]["kinf"] = 4.0e-12
reaction_network[i]["m"] = 0.3

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = "JPL 2020"

###############################################################################################################################

# Reaction 66: HO2 + NO3 -> O2 + HNO3

i = 66
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [44, 91, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [7, 12, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 3.5e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 0.3

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mellouki et al. (1993)"

###############################################################################################################################

# Reaction 67: HO2 + NO3 -> OH + NO2 + O2

i = 67
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [44, 91, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 3
reaction_network[i]["product_ids"] = [13, 10, 7, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, 0, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, 1.0, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 3.5e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 0.7

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mellouki et al. (1993)"

###############################################################################################################################

# Reaction 68: NO2 + O3 -> NO3 + O2

i = 68
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [10, 3, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [91, 7, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 1.2e-13
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 2450.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "JPL 2020"

###############################################################################################################################

# Reaction 69: NO2 + NO3 + M -> N2O5 + M

i = 69
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [10, 91, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [139, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 1

reaction_network[i]["k0"] = 2.4e-30
reaction_network[i]["n"] = 3.0
reaction_network[i]["kinf"] = 1.6e-12
reaction_network[i]["m"] = -0.1

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = "JPL 2020"

###############################################################################################################################

# Reaction 70: NO2 + NO3 -> NO + NO2 + O2

i = 70
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [10, 91, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 3
reaction_network[i]["product_ids"] = [8, 10, 7, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, 0, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, 1.0, NONE_VALUE]

reaction_network[i]["ratetype"] = 0

reaction_network[i]["alpha"] = 8.2e-14
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 1480.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

###############################################################################################################################

# Reaction 71: CO2+ + O2 -> O2+ + CO2

i = 71
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1002, 7, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1007, 2, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 5.5e-11
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

###############################################################################################################################

# Reaction 72: CO2+ + O -> O+ + CO2

i = 72
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1002, 45, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1045, 2, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 9.6e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Fox et al. (2021)"

###############################################################################################################################

# Reaction 73: CO2+ + O -> O2+ + CO

i = 73
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1002, 45, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1007, 5, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.64e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Fox et al. (2021)"

###############################################################################################################################

# Reaction 74: O2+ + e- -> O + O

i = 74
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1007, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [45, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.0e-7
reaction_network[i]["n"] = -0.7
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 0.2

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Alge et al. (1983) with branching ratio from Kella et al. (1997)"

###############################################################################################################################

# Reaction 75: O+ + CO2 -> O2+ + CO

i = 75
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1045, 2, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1007, 5, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 9.4e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

#########################################################################################################################

# Reaction 76: CO2+ + e- -> CO + O

i = 76
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1002, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [5, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 3.8e-7
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 77: CO2+ + NO -> NO+ + CO2

i = 77
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1002, 8, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [2, 1008, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.2e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 78: O2+ + NO -> NO+ + O2

i = 78
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1007, 8, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1008, 7, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 4.6e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 79: O2+ + N2 -> NO+ + NO

i = 79
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1007, 22, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1008, 8, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.0e-15
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 80: O2+ + N -> NO+ + O

i = 80
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1007, 47, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1008, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.0e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 81: O+ + N2 -> NO+ + N

i = 81
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1045, 22, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1008, 47, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.2e-12
reaction_network[i]["n"] = -0.45
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 82: NO+ + e- -> N + O

i = 82
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1008, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [47, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 4.3e-7
reaction_network[i]["n"] = -0.37
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 83: CO+ + CO2 -> CO2+ + CO

i = 83
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1005, 2, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1002, 5, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.0e-9
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 84: CO+ + O -> O+ + CO

i = 84
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1005, 45, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1045, 5, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.4e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 85: C+ + CO2 -> CO+ + CO

i = 85
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1046, 2, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1005, 5, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.1e-9
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 86: N2+ + CO2 -> CO2+ + N2

i = 86
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1022, 2, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1002, 22, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 9.0e-10
reaction_network[i]["n"] = -0.23
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 87: N2+ + O -> NO+ + N

i = 87
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1022, 45, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1008, 47, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.33e-10
reaction_network[i]["n"] = -0.44
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 88: N2+ + CO -> CO+ + N2

i = 88
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1022, 5, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1005, 22, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 7.4e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 89: N2+ + e- -> N + N

i = 89
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1022, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [47, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.7e-7
reaction_network[i]["n"] = -0.3
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 90: N2+ + O -> O+ + N2

i = 90
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1022, 45, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1045, 22, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 7.0e-12
reaction_network[i]["n"] = -0.23
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 91: N+ + CO2 -> CO2+ + N

i = 91
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1047, 2, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1002, 47, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 7.5e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 92: CO+ + H -> H+ + CO

i = 92
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1005, 48, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1048, 5, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 4.0e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 93: O+ + H -> H+ + O

i = 93
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1045, 48, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1048, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 5.66e-10
reaction_network[i]["n"] = 0.36
reaction_network[i]["gamma"] = -8.6
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 94: H+ + O -> O+ + H

i = 94
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1048, 45, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1045, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 6.86e-10
reaction_network[i]["n"] = 0.26
reaction_network[i]["gamma"] = 224.3
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 95: CO2+ + H2 -> HCO2+ + H

i = 95
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1002, 39, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1080, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 9.5e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 96: HCO2+ + e- -> H + O + CO

i = 96
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1080, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 3
reaction_network[i]["product_ids"] = [48, 45, 5, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, 0, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, 1.0, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 8.1e-7
reaction_network[i]["n"] = -0.64
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 97: HCO2+ + e- -> OH + CO

i = 97
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1080, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 5, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 3.2e-7
reaction_network[i]["n"] = -0.64
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 98: HCO2+ + e- -> H + CO2

i = 98
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1080, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [48, 2, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 6.0e-8
reaction_network[i]["n"] = -0.64
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 99: HCO2+ + O -> HCO+ + O2

i = 99
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1080, 45, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1081, 7, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.0e-9
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 100: HCO2+ + CO -> HCO+ + CO2

i = 100
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1080, 5, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1081, 2, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 7.8e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 101: H+ + CO2 -> HCO+ + O

i = 101
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1048, 2, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1081, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 3.5e-9
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 102: CO2+ + H -> HCO+ + O

i = 102
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1002, 48, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1081, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 4.5e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 103: CO+ + H2 -> HCO+ + H

i = 103
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1005, 39, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1081, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 7.5e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 104: HCO+ + e- -> CO + H

i = 104
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1081, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [5, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.4e-7
reaction_network[i]["n"] = -0.69
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 105: CO2+ + H2O -> H2O+ + CO2

i = 105
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1002, 1, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1001, 2, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.04e-9
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 106: CO+ + H2O -> H2O+ + CO

i = 106
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1005, 1, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1001, 5, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.72e-9
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 107: O+ + H2O -> H2O+ + O

i = 107
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1045, 1, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1001, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 3.2e-9
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 108: N2+ + H2O -> H2O+ + N2

i = 108
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1022, 1, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1001, 22, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.3e-9
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 109: N+ + H2O -> H2O+ + N

i = 109
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1047, 1, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1001, 47, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.8e-9
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 110: H+ + H2O -> H2O+ + H

i = 110
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1048, 1, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1001, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 6.9e-9
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 111: H2O+ + O2 -> O2+ + H2O

i = 111
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1001, 7, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1007, 1, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 4.6e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 112: H2O+ + CO -> HCO+ + OH

i = 112
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1001, 5, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1081, 13, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 5.0e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 113: H2O+ + O -> O2+ + H2

i = 113
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1001, 45, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1007, 39, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 4.0e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 114: H2O+ + NO -> NO+ + H2O

i = 114
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1001, 8, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1008, 1, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.7e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 115: H2O+ + e- -> H + H + O

i = 115
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1001, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [48, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 3.05e-7
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 116: H2O+ + e- -> H + OH

i = 116
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1001, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [48, 13, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 8.6e-8
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 117: H2O+ + e- -> H2 + O

i = 117
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1001, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [39, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 3.9e-8
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 118: H2O+ + H2O -> H3O+ + OH

i = 118
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1001, 1, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1100, 13, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.1e-9
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 119: H2O+ + H2 -> H3O+ + H

i = 119
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1001, 39, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1100, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 6.4e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 120: HCO+ + H2O -> H3O+ + CO

i = 120
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1081, 1, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1100, 5, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.5e-9
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 121: H3O+ + e- -> OH + H + H

i = 121
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1100, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 2.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 3.05e-7
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 122: H3O+ + e- -> H2O + H

i = 122
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1100, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 7.09e-8
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 123: H3O+ + e- -> OH + H2

i = 123
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1100, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [13, 39, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 5.37e-8
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 124: H3O+ + e- -> O + H2 + H

i = 124
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1100, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 3
reaction_network[i]["product_ids"] = [45, 39, 48, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, 0, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, 1.0, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 5.6e-9
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 125: O+ + H2 -> OH+ + H

i = 125
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1045, 39, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1013, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.7e-9
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 126: OH+ + O -> O2+ + H

i = 126
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1013, 45, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1007, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 7.1e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 127: OH+ + CO2 -> HCO2+ + O

i = 127
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1013, 2, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1080, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.44e-9
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 128: OH+ + CO -> HCO+ + O

i = 128
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1013, 5, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1081, 45, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.05e-9
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 129: OH+ + NO -> NO+ + OH

i = 129
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1013, 8, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1008, 13, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 3.59e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 130: OH+ + H2 -> H2O+ + H

i = 130
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1013, 39, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1001, 48, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 1.01e-9
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

####################################################################################################

# Reaction 131: OH+ + O2 -> O2+ + OH

i = 131
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1013, 7, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [1007, 13, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 5.9e-10
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID
reaction_network[i]["ref"] = "Mars PCM"

#########################################################################################################################

#Reaction 132: N(2D) + CO -> N + CO

i = 132
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [134, 5, NONE_ID, NONE_ID]  # N(2D) + CO
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [47, 5, NONE_ID, NONE_ID]  # N + CO
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.9e-12
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'Herron (1999)'


#########################################################################################################################

#Reaction 133: N + O + M -> NO + M

i = 133
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [47, 45, NONE_ID, NONE_ID]  # N + O
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [8, NONE_ID, NONE_ID, NONE_ID]  # NO
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 2.5 * 1.8e-32
reaction_network[i]["n"] = -0.5
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = 0

reaction_network[i]["ref"] = 'Campbell & Trush (1966)'

#########################################################################################################################

#Reaction 134: N(2D) + O2 -> NO + O(1D)

i = 134
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [134, 7, NONE_ID, NONE_ID]  # N(2D) + O2
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [8, 133, NONE_ID, NONE_ID]  # NO
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 1.2e-11
reaction_network[i]["n"] = 0.0
reaction_network[i]["gamma"] = 2640.
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'NIST'

#########################################################################################################################

#Reaction 135: N(2D) + e -> N + e

i = 135
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [134, 1000, NONE_ID, NONE_ID]  # N(2D) + e
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [47, 1000, NONE_ID, NONE_ID]  # N + e
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 0  # Bimolecular reaction

reaction_network[i]["alpha"] = 3.86e-10
reaction_network[i]["n"] = -0.85
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 1.0

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = 'Fox and Sung (2001)'

###############################################################################################################################

# Reaction 136: O2+ + e- -> O + O(1D)

i = 136
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1007, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 2
reaction_network[i]["product_ids"] = [45, 133, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.0e-7
reaction_network[i]["n"] = -0.7
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 0.44

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Alge et al. (1983) with branching ratio from Kella et al. (1997)"

###############################################################################################################################

# Reaction 137: O2+ + e- -> O(1D) + O(1D)

i = 137
reaction_network[i]["id"] = i

reaction_network[i]["rtype"] = 3

reaction_network[i]["nreactants"] = 2
reaction_network[i]["reactant_ids"] = [1007, 1000, NONE_ID, NONE_ID]
reaction_network[i]["reactant_iso_ids"] = [0, 0, NONE_ID, NONE_ID]
reaction_network[i]["reactant_numbers"] = [1.0, 1.0, NONE_VALUE, NONE_VALUE]

reaction_network[i]["nproducts"] = 1
reaction_network[i]["product_ids"] = [133, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_iso_ids"] = [0, NONE_ID, NONE_ID, NONE_ID]
reaction_network[i]["product_numbers"] = [2.0, NONE_VALUE, NONE_VALUE, NONE_VALUE]

reaction_network[i]["ratetype"] = 3

reaction_network[i]["alpha"] = 2.0e-7
reaction_network[i]["n"] = -0.7
reaction_network[i]["gamma"] = 0.0
reaction_network[i]["branching"] = 0.36

reaction_network[i]["ambient_id"] = NONE_ID

reaction_network[i]["ref"] = "Alge et al. (1983) with branching ratio from Kella et al. (1997)"





####################################################################################################

@jit()
def bimolecular(br,alpha,n,gamma,t):
    """
    Function to calculate the rate of a bimolecular reaction with an Arrhenius expression.
    """
    return br * alpha * ((t / 298.0)**n) * np.exp(-gamma / t)

#####################################################################################################

@jit()
def termolecular(k0, n, kinf, m, t, dens):
    """
    Function to calculate the rate of a termolecular reaction using the Lindemann-Hinshelwood mechanism.
    """
    k0x = k0 * ((298.0 / t)**(n))
    kinfx = kinf * ((298.0 / t)**(m))
    tmp = k0x * dens
    val = (kinfx * tmp) / (kinfx + tmp)
    c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
    ktot = val * (0.6**(c))

    return ktot

#####################################################################################################

@jit()
def chemical_activation(k0, n, kinf, m, A, B, t, dens):
    """
    Function to calculate the rate of a chemically activated reaction using the Lindemann-Hinshelwood mechanism.
    """
    k0x = k0 * ((298.0 / t)**(n))
    kinfx = kinf * ((298.0 / t)**(m))
    tmp = k0x * dens
    val = (kinfx * tmp) / (kinfx + tmp)
    c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
    ktot = val * (0.6**(c))

    kint = A * np.exp(-B / t)
    kca = kint * (1.0 - ktot / kinfx)

    return kca

#####################################################################################################

@jit()
def ion_reaction(br,A,n,gamma,ti):
    """
    Function to calculate the rate of a reaction involving ions
    """
    return br * A * ((ti / 300.0)**n) * np.exp(-gamma / ti)