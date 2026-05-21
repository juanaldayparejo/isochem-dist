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
