import numpy as np
import inspect,re
import isochem
from isochem.jit import jit

#The units used in the reactions are as follows:

# Reaction rate coefficients: s-1 if rtype=1; cm3 s-1 if rtype=2

cache = True

###############################################################################################################################

@jit()
def reaction0028(nh, p, t, dens):
    """
    O + NO2 + M -> NO + O2 + M
    
    Assumed to be the same as the main isotope
    """
    
    #O + NO2 + M -> NO + O2 + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0028(nh, p, t, dens)
            
    #Apply fractionation factor
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), (15N)O2 (10)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8), O2 (7)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = "fractionation factor assumed to be 1.0"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0029(nh, p, t, dens):
    """
    NO + O3 -> NO2 + O2
    
    Assumed to be the same as the main isotope
    """
   
    #NO + O3 -> NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0029(nh, p, t, dens)
            
    #Apply fractionation factor
    rrates = rrates1
    
    #Reaction type
    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N)O (8), O3 (3)
    sID[0], sISO[0], sf[0] = 8, 2, 1.0
    sID[1], sISO[1], sf[1] = 3, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] = 10, 2, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = "fractionation factor assumed to be 1.0"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0030(nh, p, t, dens):
    """
    NO + HO2 -> NO2 + OH
    
    Assumed to be the same as the main isotope
    """
    
    #NO + HO2 -> NO2 + OH
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0030(nh, p, t, dens)
    
    #Apply fractionation factor
    rrates = rrates1
    
    #Reaction type
    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N)O (8), HO2 (9)
    sID[0], sISO[0], sf[0] = 8, 2, 1.0
    sID[1], sISO[1], sf[1] = 44, 0, 1.0


    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O2 (10), OH (13)
    pID[0], pISO[0], pf[0] = 10, 2, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = "fractionation factor assumed to be 1.0"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0031A(nh, p, t, dens):
    """
    (15N) + NO -> (15N)N + O
    
    Assumed to be the same as the main isotope
    """
    
    #(15N) + NO -> (15N)N + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0031(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N) (47), NO (8)
    sID[0], sISO[0], sf[0] = 47, 2, 1.0
    sID[1], sISO[1], sf[1] = 8, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)N (22), O (45)
    pID[0], pISO[0], pf[0] = 22, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0031B(nh, p, t, dens):
    """
    N + (15N)O -> (15N)N + O
    
    Assumed to be the same as the main isotope
    """
    
    #N + (15N)O -> (15N)N + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0031(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N (47), (15N)O (8)
    sID[0], sISO[0], sf[0] = 47, 0, 1.0
    sID[1], sISO[1], sf[1] = 8, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)N (22), O (45)
    pID[0], pISO[0], pf[0] = 22, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0032(nh, p, t, dens):
    """
    N + O2 -> NO + O
    
    Assumed to be the same as the main isotope
    """
    
    #N + O2 -> NO + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0032(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N) (47), O2 (7)
    sID[0], sISO[0], sf[0] = 47, 2, 1.0
    sID[1], sISO[1], sf[1] = 7, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8), O (45)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0033(nh, p, t, dens):
    """
    NO2 + H -> NO + OH
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + H -> NO + OH
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0033(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N)O2 (10), H (48)
    sID[0], sISO[0], sf[0] = 10, 2, 1.0
    sID[1], sISO[1], sf[1] = 48, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8), OH (13)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0034(nh, p, t, dens):
    """
    N + O -> NO
    
    Assumed to be the same as the main isotope
    """
    
    #N + O -> NO
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0034(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N) (47), O (45)
    sID[0], sISO[0], sf[0] = 47, 2, 1.0
    sID[1], sISO[1], sf[1] = 45, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0035(nh, p, t, dens):
    """
    N + HO2 -> NO + OH
    
    Assumed to be the same as the main isotope
    """
    
    #N + HO2 -> NO + OH
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0035(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N) (47), HO2 (44)
    sID[0], sISO[0], sf[0] = 47, 2, 1.0
    sID[1], sISO[1], sf[1] = 44, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8), OH (13)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0036(nh, p, t, dens):
    """
    N + OH -> NO + H
    
    Assumed to be the same as the main isotope
    """
    
    #N + HO2 -> NO + OH
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0036(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N) (47), OH (13)
    sID[0], sISO[0], sf[0] = 47, 2, 1.0
    sID[1], sISO[1], sf[1] = 13, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8), H (48)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0037(nh, p, t, dens):
    """
    N(2D) + O -> N + O
    
    Assumed to be the same as the main isotope
    """
    
    #N(2D) + O -> N + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0037(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N)(2D) (134) , O (45)
    sID[0], sISO[0], sf[0] = 134, 2, 1.0
    sID[1], sISO[1], sf[1] = 45, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N) (47), O (45)
    pID[0], pISO[0], pf[0] = 47, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0038(nh, p, t, dens):
    """
    N(2D) + N2 -> N + N2
    
    Assumed to be the same as the main isotope
    """
    
    #N(2D) + N2 -> N + N2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0038(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N)(2D) (134) , N2 (22)
    sID[0], sISO[0], sf[0] = 134, 2, 1.0
    sID[1], sISO[1], sf[1] = 22, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N) (47), N2 (22)
    pID[0], pISO[0], pf[0] = 47, 2, 1.0
    pID[1], pISO[1], pf[1] = 22, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0039(nh, p, t, dens):
    """
    N(2D) + CO2 -> NO + CO
    
    Assumed to be the same as the main isotope
    """
    
    #N(2D) + CO2 -> NO + CO
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0039(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # (15N)(2D) (134) , CO2 (2)
    sID[0], sISO[0], sf[0] = 134, 2, 1.0
    sID[1], sISO[1], sf[1] = 2, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8), CO (5)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref


###############################################################################################################################

@jit()
def reaction0043A(nh, p, t, dens):
    """
    O(1D) + 15NN + M -> 15NNO + M
    
    Assumed to be the same as the main isotope
    """
    
    #O(1D) + N2 + M -> N2O + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0043(nh, p, t, dens)
    
    rrates = rrates1 / 2.0

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133) , (15N)N (22)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 22, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)NO (4)
    pID[0], pISO[0], pf[0] = 4, 3, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0043B(nh, p, t, dens):
    """
    O(1D) + 15NN + M -> N(15N)O + M
    
    Assumed to be the same as the main isotope
    """
    
    #O(1D) + N2 + M -> N2O + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0043(nh, p, t, dens)
    
    rrates = rrates1 / 2.0

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133) , (15N)N (22)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 22, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)NO (4)
    pID[0], pISO[0], pf[0] = 4, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0044(nh, p, t, dens):
    """
    O + NO + CO2 -> NO2 + CO2
    
    Assumed to be the same as the main isotope
    """
    
    #O + NO + CO2 -> NO2 + CO2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0044(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45) , (15N)O (8)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 8, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O2 (10)
    pID[0], pISO[0], pf[0] = 10, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0046A(nh, p, t, dens):
    """
    O(1D) + 15NNO -> 15NN + O2
    
    Assumed to be the same as the main isotope
    """
    
    #O(1D) + N2O -> N2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0046(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133) , (15N)NO (4)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 4, 3, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O2 (10)
    pID[0], pISO[0], pf[0] = 22, 2, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0046B(nh, p, t, dens):
    """
    O(1D) + N15NO -> 15NN + O2
    
    Assumed to be the same as the main isotope
    """
    
    #O(1D) + N2O -> N2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0046(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133) , N(15N)O (4)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 4, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O2 (10)
    pID[0], pISO[0], pf[0] = 22, 2, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0047A(nh, p, t, dens):
    """
    O(1D) + 15NNO -> 15NO + NO
    
    Assumed to be the same as the main isotope
    """
    
    #O(1D) + N2O -> NO + NO
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0047(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133) , (15N)NO (4)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 4, 3, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8), NO (8)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0
    pID[1], pISO[1], pf[1] = 8, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0047B(nh, p, t, dens):
    """
    O(1D) + N15NO -> 15NO + NO
    
    Assumed to be the same as the main isotope
    """
    
    #O(1D) + N2O -> NO + NO
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0047(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133) , N(15N)O (4)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 4, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8), NO (8)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0
    pID[1], pISO[1], pf[1] = 8, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0048(nh, p, t, dens):
    """
    O + (15N)O2 + M -> (15N)O3 + M
    
    Assumed to be the same as the main isotope
    """
    
    #O + NO2 + M -> NO3 + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0048(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), NO2 (10)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO3 (91)
    pID[0], pISO[0], pf[0] = 91, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0049(nh, p, t, dens):
    """
    O + (15N)O3 -> O2 + (15N)O2
    
    Assumed to be the same as the main isotope
    """
    
    #O + NO3 -> O2 + NO2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0049(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), NO3 (91)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 91, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # O2 (7), NO2 (10)
    pID[0], pISO[0], pf[0] = 7,  0, 1.0
    pID[1], pISO[1], pf[1] = 10, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0050A(nh, p, t, dens):
    """
    15N + NO2 -> 15NNO + O
    
    Assumed to be the same as the main isotope
    """
    
    #N + NO2 -> N2O + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0050(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # 15N (47) , NO2 (10)
    sID[0], sISO[0], sf[0] = 47, 2, 1.0
    sID[1], sISO[1], sf[1] = 10, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)NO (4), O (45)
    pID[0], pISO[0], pf[0] = 4, 3, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0050B(nh, p, t, dens):
    """
    15N + NO2 -> N15NO + O
    
    Assumed to be the same as the main isotope
    """
    
    #N + NO2 -> N2O + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0050(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # 15N (47) , NO2 (10)
    sID[0], sISO[0], sf[0] = 47, 2, 1.0
    sID[1], sISO[1], sf[1] = 10, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N(15N)O (4), O (45)
    pID[0], pISO[0], pf[0] = 4, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0050C(nh, p, t, dens):
    """
    N + 15NO2 -> 15NNO + O
    
    Assumed to be the same as the main isotope
    """
    
    #N + NO2 -> N2O + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0050(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N (47) , (15N)O2 (10)
    sID[0], sISO[0], sf[0] = 47, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)NO (4), O (45)
    pID[0], pISO[0], pf[0] = 4, 3, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0050D(nh, p, t, dens):
    """
    N + 15NO2 -> N15NO + O
    
    Assumed to be the same as the main isotope
    """
    
    #N + NO2 -> N2O + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0050(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N (47) , (15N)O2 (10)
    sID[0], sISO[0], sf[0] = 47, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)NO (4), O (45)
    pID[0], pISO[0], pf[0] = 4, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0051A(nh, p, t, dens):
    """
    (15N)O + NO3 -> (15N)O2 + NO2
    
    Assumed to be the same as the main isotope
    """
    
    #NO + NO3 -> NO2 + NO2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0051(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO (8), NO3 (91)
    sID[0], sISO[0], sf[0] = 8,  2, 1.0
    sID[1], sISO[1], sf[1] = 91, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 2 NO2 (10)
    pID[0], pISO[0], pf[0] = 10, 2, 1.0
    pID[1], pISO[1], pf[1] = 10, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0051B(nh, p, t, dens):
    """
    NO + (15N)O3 -> (15N)O2 + NO2
    
    Assumed to be the same as the main isotope
    """
    
    #NO + NO3 -> NO2 + NO2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0051(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO (8), NO3 (91)
    sID[0], sISO[0], sf[0] = 8,  0, 1.0
    sID[1], sISO[1], sf[1] = 91, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 2 NO2 (10)
    pID[0], pISO[0], pf[0] = 10, 2, 1.0
    pID[1], pISO[1], pf[1] = 10, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0051C(nh, p, t, dens):
    """
    (15N)O + (15N)O3 -> (15N)O2 + (15N)O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO + NO3 -> NO2 + NO2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0051(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO (8), NO3 (91)
    sID[0], sISO[0], sf[0] = 8,  2, 1.0
    sID[1], sISO[1], sf[1] = 91, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 2 NO2 (10)
    pID[0], pISO[0], pf[0] = 10, 2, 2.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0052(nh, p, t, dens):
    """
    (15N)O2 + O3 -> (15N)O3 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + O3 -> NO3 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0052(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), O3 (3)
    sID[0], sISO[0], sf[0] = 10, 2, 1.0
    sID[1], sISO[1], sf[1] = 3,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO3 (91), O2 (7)
    pID[0], pISO[0], pf[0] = 91, 2, 1.0
    pID[1], pISO[1], pf[1] = 7,  0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0053A(nh, p, t, dens):
    """
    NO3 + NO3 -> NO2 + NO2 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO3 + NO3 -> 2NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0053(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # 2 NO3 (91)
    sID[0], sISO[0], sf[0] = 91, 0, 1.0
    sID[1], sISO[1], sf[1] = 91, 2, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 2 NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] = 10, 0, 1.0
    pID[1], pISO[1], pf[1] = 10, 2, 1.0
    pID[2], pISO[2], pf[2] = 7,  0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0053B(nh, p, t, dens):
    """
    (15N)O3 + (15N)O3 -> (15N)O2 + (15N)O2 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO3 + NO3 -> 2NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0053(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # 2 NO3 (91)
    sID[0], sISO[0], sf[0] = 91, 2, 2.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 2 NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] = 10, 2, 2.0
    pID[1], pISO[1], pf[1] = 7,  0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0056(nh, p, t, dens):
    """
    N + O3 -> NO + O2
    
    Assumed to be the same as the main isotope
    """
    
    #N + O3 -> NO + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0056(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # 15N (47) , O3 (3)
    sID[0], sISO[0], sf[0] = 47, 2, 1.0
    sID[1], sISO[1], sf[1] = 3, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (15N)O (8), O2 (7)
    pID[0], pISO[0], pf[0] = 8, 2, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0057A(nh, p, t, dens):
    """
    (15N)(2D) + NO -> (15N)N + O
    
    Assumed to be the same as the main isotope
    """
    
    #N(2D) + NO -> N2 + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0057(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # 15N(2D) , NO (8)
    sID[0], sISO[0], sf[0] = 134, 2, 1.0
    sID[1], sISO[1], sf[1] = 8, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 15NN (22), O (45)
    pID[0], pISO[0], pf[0] = 22, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0057B(nh, p, t, dens):
    """
    N(2D) + (15N)O -> (15N)N + O
    
    Assumed to be the same as the main isotope
    """
    
    #N(2D) + NO -> N2 + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0057(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N(2D) , (15N)O (8)
    sID[0], sISO[0], sf[0] = 134, 0, 1.0
    sID[1], sISO[1], sf[1] = 8, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 15NN (22), O (45)
    pID[0], pISO[0], pf[0] = 22, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0058(nh, p, t, dens):
    """
    H + (15N)O3 -> OH + (15N)O2
    
    Assumed to be the same as the main isotope
    """
    
    #H + NO3 -> OH + NO2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0058(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # H (48) , NO3 (91)
    sID[0], sISO[0], sf[0] = 48, 0, 1.0
    sID[1], sISO[1], sf[1] = 91, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # OH (13), NO2 (10)
    pID[0], pISO[0], pf[0] = 13, 0, 1.0
    pID[1], pISO[1], pf[1] = 10, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0059(nh, p, t, dens):
    """
    OH + (15N)O + M -> HO(15N)O + M 
    
    Assumed to be the same as the main isotope
    """
    
    #OH + NO + M -> HONO + M 
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0059(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13) , NO (8)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 8, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HONO
    pID[0], pISO[0], pf[0] = 138, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0060(nh, p, t, dens):
    """
    OH + (15N)O2 + M-> H(15N)O3 + M
    
    Assumed to be the same as the main isotope
    """
    
    #OH + NO2 + M-> HNO3 + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0060(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13) , NO2 (10)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HNO3
    pID[0], pISO[0], pf[0] = 12, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0061(nh, p, t, dens):
    """
    OH + (15N)O3 -> HO2 + (15N)O2
    
    Assumed to be the same as the main isotope
    """
    
    #OH + NO3 -> HO2 + NO2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0061(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13) , NO3 (91)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 91, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HO2 (44) , NO2 (10)
    pID[0], pISO[0], pf[0] = 44, 0, 1.0
    pID[1], pISO[1], pf[1] = 10, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0062(nh, p, t, dens):
    """
    OH + HO(15N)O -> H2O + (15N)O2
    
    Assumed to be the same as the main isotope
    """
    
    #OH + HONO -> H2O + NO2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0062(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13) , HONO (138)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 138, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HO2 (44) , NO2 (10)
    pID[0], pISO[0], pf[0] = 1, 0, 1.0
    pID[1], pISO[1], pf[1] = 10, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0063(nh, p, t, dens):
    """
    OH + H(15N)O3 -> H2O + (15N)O3
    
    Assumed to be the same as the main isotope
    """
    
    #OH + HNO3 -> H2O + NO3
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0063(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13) , HNO3 (12)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 12, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HO2 (44) , NO2 (10)
    pID[0], pISO[0], pf[0] = 1, 0, 1.0
    pID[1], pISO[1], pf[1] = 91, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0064(nh, p, t, dens):
    """
    OH + HO2(15N)O2 -> H2O + (15N)O2 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #OH + HO2NO2 -> H2O + NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0064(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13) , HO2NO2 (137)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 137, 2, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HO2 (44) , NO2 (10)
    pID[0], pISO[0], pf[0] = 1, 0, 1.0
    pID[1], pISO[1], pf[1] = 10, 2, 1.0
    pID[2], pISO[2], pf[2] = 7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0065(nh, p, t, dens):
    """
    HO2 + (15N)O2 + M -> HO2(15N)O2 + M
    
    Assumed to be the same as the main isotope
    """
    
    #HO2 + NO2 + M -> HO2NO2 + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0065(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # HO2 (44) , NO2 (10)
    sID[0], sISO[0], sf[0] = 44, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HO2NO2 (137)
    pID[0], pISO[0], pf[0] = 137, 2, 1.0


    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0066(nh, p, t, dens):
    """
    HO2 + NO3 -> O2 + HNO3
    
    Assumed to be the same as the main isotope
    """
    
    #HO2 + NO3 -> O2 + HNO3
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0066(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # HO2 (44) , NO3 (91)
    sID[0], sISO[0], sf[0] = 44, 0, 1.0
    sID[1], sISO[1], sf[1] = 91, 2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # O2 (7) , HNO3 (12)
    pID[0], pISO[0], pf[0] = 7, 0, 1.0
    pID[1], pISO[1], pf[1] = 12, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0067(nh, p, t, dens):
    """
    HO2 + (15N)O3 -> OH + (15N)O2 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #HO2 + NO3 -> OH + NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0067(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # HO2 (44), NO3 (91)
    sID[0], sISO[0], sf[0] = 44, 0, 1.0
    sID[1], sISO[1], sf[1] = 91, 2, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # OH (13), NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] =  13, 0, 1.0
    pID[1], pISO[1], pf[1] =  10, 2, 1.0
    pID[2], pISO[2], pf[2] =  7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0068(nh, p, t, dens):
    """
    (15N)O2 + O3 -> (15N)O3 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + O3 -> NO3 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0068(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), O3 (3)
    sID[0], sISO[0], sf[0] =   10, 2, 1.0
    sID[1], sISO[1], sf[1] =    3, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO3 (91), O2 (7)
    pID[0], pISO[0], pf[0] =  91, 2, 1.0
    pID[1], pISO[1], pf[1] =   7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0069A(nh, p, t, dens):
    """
    (15N)O2 + NO3 + M -> (15N)NO5 + M
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + NO3 + M -> N2O5 + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0069(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 2, 1.0
    sID[1], sISO[1], sf[1] =   91, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N2O5 (139)
    pID[0], pISO[0], pf[0] =   139, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0069B(nh, p, t, dens):
    """
    NO2 + (15N)O3 + M -> (15N)NO5 + M
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + NO3 + M -> N2O5 + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0069(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 0, 1.0
    sID[1], sISO[1], sf[1] =   91, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N2O5 (139)
    pID[0], pISO[0], pf[0] =   139, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0069C(nh, p, t, dens):
    """
    (15N)O2 + (15N)O3 + M -> (15N)2O5 + M
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + NO3 + M -> N2O5 + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0069(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 2, 1.0
    sID[1], sISO[1], sf[1] =   91, 2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N2O5 (139)
    pID[0], pISO[0], pf[0] =   139, 3, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0070A(nh, p, t, dens):
    """
    (15N)O2 + NO3  -> (15N)O + NO2 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + NO3  -> NO + NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0070(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 2, 1.0
    sID[1], sISO[1], sf[1] =   91, 0, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] =   8, 2, 1.0
    pID[1], pISO[1], pf[1] =  10, 0, 1.0
    pID[2], pISO[2], pf[2] =   7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0070B(nh, p, t, dens):
    """
    (15N)O2 + NO3  -> NO + (15N)O2 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + NO3  -> NO + NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0070(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 2, 1.0
    sID[1], sISO[1], sf[1] =   91, 0, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] =   8, 0, 1.0
    pID[1], pISO[1], pf[1] =  10, 2, 1.0
    pID[2], pISO[2], pf[2] =   7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0070C(nh, p, t, dens):
    """
    NO2 + (15N)O3  -> (15N)O + NO2 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + NO3  -> NO + NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0070(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 0, 1.0
    sID[1], sISO[1], sf[1] =   91, 2, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] =   8, 2, 1.0
    pID[1], pISO[1], pf[1] =  10, 0, 1.0
    pID[2], pISO[2], pf[2] =   7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0070D(nh, p, t, dens):
    """
    NO2 + (15N)O3  -> NO + (15N)O2 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + NO3  -> NO + NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0070(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 0, 1.0
    sID[1], sISO[1], sf[1] =   91, 2, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] =   8, 0, 1.0
    pID[1], pISO[1], pf[1] =  10, 2, 1.0
    pID[2], pISO[2], pf[2] =   7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0070E(nh, p, t, dens):
    """
    (15N)O2 + (15N)O3  -> (15N)O + (15N)O2 + O2
    
    Assumed to be the same as the main isotope
    """
    
    #NO2 + NO3  -> NO + NO2 + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0070(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 2, 1.0
    sID[1], sISO[1], sf[1] =   91, 2, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] =   8, 2, 1.0
    pID[1], pISO[1], pf[1] =  10, 2, 1.0
    pID[2], pISO[2], pf[2] =   7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0077(nh, p, t, dens):
    """
    CO2+ + (15N)O -> (15N)O+ + CO2
    
    Assumed to be the same as the main isotope
    """
    
    #CO2+ + NO -> NO+ + CO2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0077(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1002, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 2, 0, 1.0
    pID[1], pISO[1], pf[1] = 1008, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0078(nh, p, t, dens):
    """
    O2+ + (15N)O -> (15N)O+ + O2
    
    Assumed to be the same as the main isotope
    """
    
    #O2+ + NO -> NO+ + O2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0078(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1007, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 2, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0079A(nh, p, t, dens):
    """
    O2+ + (15N)N -> (15N)O+ + NO
    
    Assumed to be the same as the main isotope
    """
    
    #O2+ + N2 -> NO+ + NO
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0079(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1007, 0, 1.0
    sID[1], sISO[1], sf[1] = 22,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 2, 1.0
    pID[1], pISO[1], pf[1] = 8, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0079B(nh, p, t, dens):
    """
    O2+ + (15N)N -> NO+ + (15N)O
    
    Assumed to be the same as the main isotope
    """
    
    #O2+ + N2 -> NO+ + NO
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0079(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1007, 0, 1.0
    sID[1], sISO[1], sf[1] = 22,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 8, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0080(nh, p, t, dens):
    """
    O2+ + (15N) -> (15N)O+ + O
    
    Assumed to be the same as the main isotope
    """
    
    #O2+ + N -> NO+ + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0080(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1007, 0, 1.0
    sID[1], sISO[1], sf[1] = 47,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0081A(nh, p, t, dens):
    """
    O+ + (15N)N -> (15N)O+ + N
    
    Assumed to be the same as the main isotope
    """
    
    #O+ + N2 -> NO+ + N
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0081(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1045, 0, 1.0
    sID[1], sISO[1], sf[1] = 22,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 2, 1.0
    pID[1], pISO[1], pf[1] = 47, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0081B(nh, p, t, dens):
    """
    O+ + (15N)N -> NO+ + (15N)
    
    Assumed to be the same as the main isotope
    """
    
    #O+ + N2 -> NO+ + N
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0081(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1045, 0, 1.0
    sID[1], sISO[1], sf[1] = 22,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 47, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0082(nh, p, t, dens):
    """
    (15N)O+ + e- -> (15N) + O
    
    Assumed to be the same as the main isotope
    """
    
    #NO+ + e- -> N + O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0082(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1008, 2, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 47, 2, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0086(nh, p, t, dens):
    """
    (15N)N+ + CO2 -> CO2+ + (15N)N
    
    Assumed to be the same as the main isotope
    """
    
    #N2+ + CO2 -> CO2+ + N2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0086(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 2, 1.0
    sID[1], sISO[1], sf[1] = 2,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1002, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0087A(nh, p, t, dens):
    """
    (15N)N+ + O -> (15N)O+ + N
    
    Assumed to be the same as the main isotope
    """
    
    #N2+ + O -> NO+ + N
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0087(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 2, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 2, 1.0
    pID[1], pISO[1], pf[1] = 47, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0087B(nh, p, t, dens):
    """
    (15N)N+ + O -> NO+ + (15N)
    
    Assumed to be the same as the main isotope
    """
    
    #N2+ + O -> NO+ + N
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0087(nh, p, t, dens)
    
    rrates = rrates1 / 2.

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 2, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 47, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0088(nh, p, t, dens):
    """
    (15N)N+ + CO -> CO+ + (15N)N
    
    Assumed to be the same as the main isotope
    """
    
    #N2+ + CO -> CO+ + N2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0088(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 2, 1.0
    sID[1], sISO[1], sf[1] = 5,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1005, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0089(nh, p, t, dens):
    """
    (15N)N+ + e– -> (15N) + N
    
    Assumed to be the same as the main isotope
    """
    
    #N2+ + e– -> N + N
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0089(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 2, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 47, 0, 1.0
    pID[1], pISO[1], pf[1] = 47, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0090(nh, p, t, dens):
    """
    (15N)N+ + O -> O+ + (15N)N
    
    Assumed to be the same as the main isotope
    """
    
    #N2+ + O -> O+ + N2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0090(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 2, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1045, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0091(nh, p, t, dens):
    """
    (15N)+ + CO2 -> CO2+ + (15N)
    
    Assumed to be the same as the main isotope
    """
    
    #N+ + CO2 -> CO2+ + N
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0091(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1047, 2, 1.0
    sID[1], sISO[1], sf[1] = 2,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1002, 0, 1.0
    pID[1], pISO[1], pf[1] = 47, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0108(nh, p, t, dens):
    """
    (15N)N+ + H2O -> H2O+ + (15N)N
    
    Assumed to be the same as the main isotope
    """
    
    #N2+ + H2O -> H2O+ + N2
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0108(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 2, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1001, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0109(nh, p, t, dens):
    """
    (15N)+ + H2O -> H2O+ + (15N)
    
    Assumed to be the same as the main isotope
    """
    
    #N+ + H2O -> H2O+ + N
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0109(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1047, 2, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1001, 0, 1.0
    pID[1], pISO[1], pf[1] = 47, 2, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0114(nh, p, t, dens):
    """
    H2O+ + (15N)O -> (15N)O+ + H2O
    
    Assumed to be the same as the main isotope
    """
    
    #H2O+ + NO -> NO+ + H2O
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0114(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 2, 1.0
    pID[1], pISO[1], pf[1] = 1, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0129(nh, p, t, dens):
    """
    OH+ + (15N)O -> (15N)O+ + OH
    
    Assumed to be the same as the main isotope
    """
    
    #OH+ + NO -> NO+ + OH
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0129(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1013, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 2, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = "assumed to be the same as the main isotope"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################



