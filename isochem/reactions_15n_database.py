import numpy as np
from isochem.jit import jit

MAX_BRANCHES = 5   #For a given reaction with neutrals, maximum number of associated isotopic reactions
MAX_PRODUCTS = 4
MAX_REACTANTS = 2

reaction_network_isotope_dtype = np.dtype([
    #Reaction ID
    ("id", np.int32),

    #Number of isotopic branches the reaction is split into
    ("nbranch", np.int32),

    #Reaction type (isochem)
    ("rtype", np.int32, MAX_BRANCHES),

    #Reactants information
    ("nreactants", np.int32, MAX_BRANCHES),
    ("reactant_ids", np.int32, (MAX_REACTANTS, MAX_BRANCHES)),
    ("reactant_iso_ids", np.int32, (MAX_REACTANTS, MAX_BRANCHES)),
    ("reactant_numbers", np.float64, (MAX_REACTANTS, MAX_BRANCHES)),

    #Products information
    ("nproducts", np.int32, MAX_BRANCHES),
    ("product_ids", np.int32, (MAX_PRODUCTS, MAX_BRANCHES)),
    ("product_iso_ids", np.int32, (MAX_PRODUCTS, MAX_BRANCHES)),
    ("product_numbers", np.float64, (MAX_PRODUCTS, MAX_BRANCHES)),

    #Reaction rates formulation
    ("fractionation_type", np.int32),   #0 = MASS-DEPENDENT FRACTIONATION ; 1 = FRACTIONATION FACTOR

    #Branching ratio
    ("branching_factor", np.float64, MAX_BRANCHES),

    #Specifying fractionation factor
    ("fractionation_factor", np.float64, MAX_BRANCHES), 

    #Reference for the reaction rate coefficients (e.g., JPL, IUPAC, etc)
    ("ref", 'U50'),

])


#########################################################################################################################

MAX_REACTIONS = 10000
reaction_network_15n = np.empty(MAX_REACTIONS, dtype=reaction_network_isotope_dtype)

NONE_ID = np.int32(-999999)
NONE_VALUE = np.nan


# Initialize entire array once at the start
reaction_network_15n["id"].fill(NONE_ID)
reaction_network_15n["rtype"].fill(NONE_ID)
reaction_network_15n["nbranch"].fill(NONE_ID)
reaction_network_15n["nreactants"].fill(NONE_ID)
reaction_network_15n["nproducts"].fill(NONE_ID)
reaction_network_15n["reactant_ids"].fill(NONE_ID)
reaction_network_15n["reactant_iso_ids"].fill(NONE_ID)
reaction_network_15n["reactant_numbers"].fill(NONE_VALUE)
reaction_network_15n["product_ids"].fill(NONE_ID)
reaction_network_15n["product_iso_ids"].fill(NONE_ID)
reaction_network_15n["product_numbers"].fill(NONE_VALUE)
reaction_network_15n["fractionation_factor"].fill(NONE_VALUE)
reaction_network_15n["branching_factor"].fill(NONE_VALUE)

#########################################################################################################################

#Reaction 28: O + (15N)O2 + M -> (15N)O + O2 + M

i = 28
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [45, 10]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0 #Mass-dependent fractionation
reaction_network_15n[i]["ref"] = ""

#########################################################################################################################

# Reaction 29: (15N)O + O3 -> (15N)O2 + O2

i = 29
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [8, 3]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 30: (15N)O + HO2 -> (15N)O2 + OH

i = 30
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [8, 44]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [10, 13]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

#Reaction 31: N + NO -> N2 + O

#(15N) + NO -> (15N)N + O
#N + (15N)O -> (15N)N + O

i = 31
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [22, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

ibranch = 1
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [22, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0 #Mass-dependent fractionation
reaction_network_15n[i]["ref"] = ""

#########################################################################################################################

# Reaction 32: (15N) + O2 -> (15N)O + O

i = 32
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 7]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 33: (15N)O2 + H -> (15N)O + OH

i = 33
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 48]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 13]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 34: (15N) + O -> (15N)O

i = 34
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 45]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 35: (15N) + HO2 -> (15N)O + OH

i = 35
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 44]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 13]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 36: (15N) + OH -> (15N)O + H

i = 36
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 13]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 48]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 37: (15N)(2D) + O -> (15N) + O

i = 37
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [134, 45]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [47, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 38: (15N)(2D) + N2 -> (15N) + N2

i = 38
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [134, 22]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [47, 22]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 39: (15N)(2D) + CO2 -> (15N)O + CO

i = 39
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [134, 2]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 5]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 43: O(1D) + N2 + M -> N2O + M
# Two isotopic branches: (15N)NO with iso=3 and iso=2

i = 43
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

# Branch 0: (15N)NO with iso=3
ibranch = 0
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [133, 22]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [4]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [3]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 1: (15N)NO with iso=2
ibranch = 1
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [133, 22]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [4]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 44: O + (15N)O + CO2 -> (15N)O2 + CO2

i = 44
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [45, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [10]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 46: O(1D) + N2O -> N2 + O2
# Two isotopic branches: 15NNO and N15NO

i = 46
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

# Branch 0: O(1D) + 15NNO -> 15NN + O2
ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [133, 4]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 3]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [22, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

# Branch 1: O(1D) + N15NO -> 15NN + O2
ibranch = 1
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [133, 4]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [22, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 47: O(1D) + N2O -> NO + NO
# Two isotopic branches: 15NNO and N15NO

i = 47
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

# Branch 0: O(1D) + 15NNO -> 15NO + NO
ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [133, 4]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 3]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 8]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

# Branch 1: O(1D) + N15NO -> 15NO + NO
ibranch = 1
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [133, 4]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 8]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 48: O + (15N)O2 + M -> (15N)O3 + M

i = 48
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [45, 10]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [91]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 49: O + (15N)O3 -> O2 + (15N)O2

i = 49
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [45, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [7, 10]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 50: N + NO2 -> N2O + O
# Four isotopic branches

i = 50
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 4

# Branch 0: 15N + NO2 -> 15NNO + O
ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 10]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [4, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [3, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 1: 15N + NO2 -> N15NO + O
ibranch = 1
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 10]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [4, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 2: N + 15NO2 -> 15NNO + O
ibranch = 2
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 10]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [4, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [3, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 3: N + 15NO2 -> N15NO + O
ibranch = 3
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 10]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [4, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 51: NO + NO3 -> NO2 + NO2
# Three isotopic branches

i = 51
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 3

# Branch 0: (15N)O + NO3 -> (15N)O2 + NO2
ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [8, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [10, 10]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

# Branch 1: NO + (15N)O3 -> (15N)O2 + NO2
ibranch = 1
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [8, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [10, 10]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

# Branch 2: (15N)O + (15N)O3 -> 2*(15N)O2
ibranch = 2
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [8, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [10]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [2.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 52: (15N)O2 + O3 -> (15N)O3 + O2

i = 52
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 3]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [91, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 53: NO3 + NO3 -> 2NO2 + O2
# Two isotopic branches

i = 53
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

# Branch 0: NO3 + (15N)O3 -> NO2 + (15N)O2 + O2
ibranch = 0
nreactants = 2
nproducts = 3
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [91, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [10, 10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

# Branch 1: (15N)O3 + (15N)O3 -> 2*(15N)O2 + O2
ibranch = 1
nreactants = 1
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 2
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [2.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [2.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 56: N + O3 -> NO + O2

i = 56
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [47, 3]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 57: N + NO -> N2 + O
# Two isotopic branches

i = 57
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

# Branch 0: (15N)(2D) + NO -> (15N)N + O
ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [134, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [22, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

# Branch 1: N(2D) + (15N)O -> (15N)N + O
ibranch = 1
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [134, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [22, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 58: H + (15N)O3 -> OH + (15N)O2

i = 58
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [48, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [13, 10]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 59: OH + (15N)O + M -> HO(15N)O + M

i = 59
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [13, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [138]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 60: OH + (15N)O2 + M -> H(15N)O3 + M

i = 60
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [13, 10]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [12]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 61: OH + (15N)O3 -> HO2 + (15N)O2

i = 61
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [13, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [44, 10]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 62: OH + HO(15N)O -> H2O + (15N)O2

i = 62
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [13, 138]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1, 10]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 63: OH + H(15N)O3 -> H2O + (15N)O3

i = 63
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [13, 12]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1, 91]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 64: OH + HO2(15N)O2 -> H2O + (15N)O2 + O2

i = 64
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 3
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [13, 137]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1, 10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 65: HO2 + (15N)O2 + M -> HO2(15N)O2 + M

i = 65
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [44, 10]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [137]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 66: HO2 + (15N)O3 -> O2 + H(15N)O3

i = 66
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [44, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [7, 12]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 67: HO2 + (15N)O3 -> OH + (15N)O2 + O2

i = 67
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 3
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [44, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [13, 10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 68: (15N)O2 + O3 -> (15N)O3 + O2

i = 68
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 3]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [91, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 69: NO2 + NO3 + M -> N2O5 + M
# Three isotopic branches

i = 69
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

# Branch 0: (15N)O2 + NO3 -> (15N)NO5
ibranch = 0
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [139]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

# Branch 1: NO2 + (15N)O3 -> (15N)NO5
ibranch = 1
nreactants = 2
nproducts = 1
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [139]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

#########################################################################################################################

# Reaction 70: NO2 + NO3 -> NO + NO2 + O2
# Five isotopic branches

i = 70
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 5

# Branch 0: (15N)O2 + NO3 -> (15N)O + NO2 + O2
ibranch = 0
nreactants = 2
nproducts = 3
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 1: (15N)O2 + NO3 -> NO + (15N)O2 + O2
ibranch = 1
nreactants = 2
nproducts = 3
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 2: NO2 + (15N)O3 -> (15N)O + NO2 + O2
ibranch = 2
nreactants = 2
nproducts = 3
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 3: NO2 + (15N)O3 -> NO + (15N)O2 + O2
ibranch = 3
nreactants = 2
nproducts = 3
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 4: (15N)O2 + (15N)O3 -> (15N)O + (15N)O2 + O2
ibranch = 4
nreactants = 2
nproducts = 3
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [10, 91]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [8, 10, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 77: CO2+ + (15N)O -> (15N)O+ + CO2

i = 77
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1002, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [2, 1008]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 78: O2+ + (15N)O -> (15N)O+ + O2

i = 78
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1007, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 7]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 79: O2+ + (15N)N -> products
# Two isotopic branches

i = 79
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

# Branch 0: O2+ + (15N)N -> (15N)O+ + NO
ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1007, 22]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 8]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 1: O2+ + (15N)N -> NO+ + (15N)
ibranch = 1
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1007, 22]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 8]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 80: O2+ + (15N) -> (15N)O+ + O

i = 80
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1007, 47]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 81: O+ + (15N)N -> products
# Two isotopic branches

i = 81
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

# Branch 0: O+ + (15N)N -> (15N)O+ + N
ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1045, 22]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 47]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 1: O+ + (15N)N -> NO+ + (15N)
ibranch = 1
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1045, 22]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 47]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 82: (15N)O+ + e- -> (15N) + O

i = 82
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1008, 1000]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [47, 45]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 86: (15N)N+ + CO2 -> CO2+ + (15N)N

i = 86
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1022, 2]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1002, 22]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 87: (15N)N+ + O -> products
# Two isotopic branches

i = 87
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 2

# Branch 0: (15N)N+ + O -> (15N)O+ + N
ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1022, 45]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 47]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

# Branch 1: (15N)N+ + O -> NO+ + (15N)
ibranch = 1
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1022, 45]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 47]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 0.5

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 88: (15N)N+ + CO -> CO+ + (15N)N

i = 88
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1022, 5]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1005, 22]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 89: (15N)N+ + e- -> (15N) + N

i = 89
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1022, 1000]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [47, 47]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 90: (15N)N+ + O -> O+ + (15N)N

i = 90
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1022, 45]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1045, 22]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 91: (15N)+ + CO2 -> CO2+ + (15N)

i = 91
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1047, 2]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1002, 47]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 108: (15N)N+ + H2O -> H2O+ + (15N)N

i = 108
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1022, 1]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1001, 22]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 109: (15N)+ + H2O -> H2O+ + (15N)

i = 109
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1047, 1]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [2, 0]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1001, 47]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 114: H2O+ + (15N)O -> (15N)O+ + H2O

i = 114
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1001, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 1]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################

# Reaction 129: OH+ + (15N)O -> (15N)O+ + OH

i = 129
reaction_network_15n[i]["id"] = i

reaction_network_15n[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_15n[i]["rtype"][ibranch] = 3
reaction_network_15n[i]["nreactants"][ibranch] = nreactants
reaction_network_15n[i]["reactant_ids"][:nreactants, ibranch] = [1013, 8]
reaction_network_15n[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_15n[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["nproducts"][ibranch] = nproducts
reaction_network_15n[i]["product_ids"][:nproducts, ibranch] = [1008, 13]
reaction_network_15n[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_15n[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_15n[i]["branching_factor"][ibranch] = 1.0

reaction_network_15n[i]["fractionation_type"] = 0  # Mass-dependent fractionation
reaction_network_15n[i]["ref"] = "fractionation factor following reduced mass factor"

#########################################################################################################################
