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
reaction_network_13c = np.empty(MAX_REACTIONS, dtype=reaction_network_isotope_dtype)

NONE_ID = np.int32(-999999)
NONE_VALUE = np.nan


# Initialize entire array once at the start
reaction_network_13c["id"].fill(NONE_ID)
reaction_network_13c["rtype"].fill(NONE_ID)
reaction_network_13c["nbranch"].fill(NONE_ID)
reaction_network_13c["nreactants"].fill(NONE_ID)
reaction_network_13c["nproducts"].fill(NONE_ID)
reaction_network_13c["reactant_ids"].fill(NONE_ID)
reaction_network_13c["reactant_iso_ids"].fill(NONE_ID)
reaction_network_13c["reactant_numbers"].fill(NONE_VALUE)
reaction_network_13c["product_ids"].fill(NONE_ID)
reaction_network_13c["product_iso_ids"].fill(NONE_ID)
reaction_network_13c["product_numbers"].fill(NONE_VALUE)
reaction_network_13c["fractionation_factor"].fill(NONE_VALUE)
reaction_network_13c["branching_factor"].fill(NONE_VALUE)

#########################################################################################################################

#Reaction 39: N(2D) + (13C)O2 -> NO + (13C)O

i = 39
reaction_network_13c[i]["id"] = i

reaction_network_13c[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_13c[i]["rtype"][ibranch] = 3
reaction_network_13c[i]["nreactants"][ibranch] = nreactants
reaction_network_13c[i]["reactant_ids"][:nreactants, ibranch] = [134, 2]
reaction_network_13c[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_13c[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_13c[i]["nproducts"][ibranch] = nproducts
reaction_network_13c[i]["product_ids"][:nproducts, ibranch] = [8, 5]
reaction_network_13c[i]["product_iso_ids"][:nproducts, ibranch] = [0, 2]
reaction_network_13c[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_13c[i]["branching_factor"][ibranch] = 1.0

reaction_network_13c[i]["fractionation_type"] = 0 #Mass-dependent fractionation
reaction_network_13c[i]["ref"] = ""

#########################################################################################################################

# Reaction 40: OH + (13C)O -> (13C)O2 + H

i = 40
reaction_network_13c[i]["id"] = i

reaction_network_13c[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 2
reaction_network_13c[i]["rtype"][ibranch] = 3
reaction_network_13c[i]["nreactants"][ibranch] = nreactants
reaction_network_13c[i]["reactant_ids"][:nreactants, ibranch] = [13, 5]
reaction_network_13c[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_13c[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_13c[i]["nproducts"][ibranch] = nproducts
reaction_network_13c[i]["product_ids"][:nproducts, ibranch] = [2, 48]
reaction_network_13c[i]["product_iso_ids"][:nproducts, ibranch] = [2, 0]
reaction_network_13c[i]["product_numbers"][:nproducts, ibranch] = [1.0, 1.0]
reaction_network_13c[i]["branching_factor"][ibranch] = 1.0

reaction_network_13c[i]["fractionation_type"] = 1  # Explicit fractionation factor
reaction_network_13c[i]["fractionation_factor"] = 1.006 #Valid for Mars
reaction_network_13c[i]["ref"] = "Stevens et al. (1980)"

#########################################################################################################################

#Reaction 41: OH + (13C)O -> HO(13C)O

i = 41
reaction_network_13c[i]["id"] = i

reaction_network_13c[i]["nbranch"] = 1

ibranch = 0
nreactants = 2
nproducts = 1
reaction_network_13c[i]["rtype"][ibranch] = 3
reaction_network_13c[i]["nreactants"][ibranch] = nreactants
reaction_network_13c[i]["reactant_ids"][:nreactants, ibranch] = [13, 5]
reaction_network_13c[i]["reactant_iso_ids"][:nreactants, ibranch] = [0, 2]
reaction_network_13c[i]["reactant_numbers"][:nreactants, ibranch] = [1.0, 1.0]
reaction_network_13c[i]["nproducts"][ibranch] = nproducts
reaction_network_13c[i]["product_ids"][:nproducts, ibranch] = [80]
reaction_network_13c[i]["product_iso_ids"][:nproducts, ibranch] = [3]
reaction_network_13c[i]["product_numbers"][:nproducts, ibranch] = [1.0]
reaction_network_13c[i]["branching_factor"][ibranch] = 1.0

reaction_network_13c[i]["fractionation_type"] = 1  # Explicit fractionation factor
reaction_network_13c[i]["fractionation_factor"] = 1.006 #Valid for Mars
reaction_network_13c[i]["ref"] = "Stevens et al. (1980)"

#########################################################################################################################
