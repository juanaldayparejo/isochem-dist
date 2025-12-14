import numpy as np
import inspect,re
import isochem
from isochem.jit import jit

#The units used in the reactions are as follows:

# Reaction rate coefficients: s-1 if rtype=1; cm3 s-1 if rtype=2

###############################################################################################################################

@jit(nopython=True)
def reaction0039(nh, p, t, dens):
    """
    N(2D) + CO2 -> NO + CO
    
    Assumed to be the same as the main isotope
    """
    
    #N(2D) + CO2 -> NO + CO
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0039(nh, p, t, dens)
            
    #Apply fractionation factor
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N(2D) (134), (13)CO2 (2)
    sID[0], sISO[0], sf[0] = 134, 0, 1.0
    sID[1], sISO[1], sf[1] = 2,   2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), (13)CO (5)
    pID[0], pISO[0], pf[0] = 8, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 2, 1.0

    ref = "herron, j. phys. chem. ref. data, 1999"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit(nopython=True)
def reaction0040(nh, p, t, dens):
    """
    OH + CO -> CO2 + H
    
    Assumed to be the same as the main isotope, but with a 
    fractionation factor from Stevens et al. (1980)

    A polynomial function is fit to capture the pressure-
    dependence of the fractionation

    k13 / k12 = 1.00638 - 1.693e-5*press(hPa) + 4.6968e-9 * press(hPa)**2.
    k13 / k12 = 1.00638 - 1.693e-7*press(Pa) + 4.6968e-13 * press(Pa)**2.
    
    """
   
    #OH + CO -> CO2 + H
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0040(nh, p, t, dens)
            
    #Apply fractionation factor
    rrates = rrates1*(1.00638 - 1.693e-7*p + 4.6968e-13*p**2)
    
    #Reaction type
    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), (13C)O (5)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  2, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (13C)O2 (2), H (48)
    pID[0], pISO[0], pf[0] = 2,   2, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = "JPL 2020 + Stevens et al. (1980)"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit(nopython=True)
def reaction0041(nh, p, t, dens):
    """
    OH + CO -> HOCO
    
    Assumed to be the same as the main isotope, but with a 
    fractionation factor from Stevens et al. (1980)

    A polynomial function is fit to capture the pressure-
    dependence of the fractionation

    k13 / k12 = 1.00638 - 1.693e-5*press(hPa) + 4.6968e-9 * press(hPa)**2.
    k13 / k12 = 1.00638 - 1.693e-7*press(Pa) + 4.6968e-13 * press(Pa)**2.
    """
    
    #OH + CO -> HOCO
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0041(nh, p, t, dens)
    
    #Apply fractionation factor
    rrates = rrates1*(1.00638 - 1.693e-7*p + 4.6968e-13*p**2)
    
    #Reaction type
    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), (13C)O (5)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  2, 1.0


    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HOCO (80)
    pID[0], pISO[0], pf[0] = 80, 2, 1.0

    ref = "JPL 2020 + Stevens et al. (1980)"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit(nopython=True)
def reaction0042(nh, p, t, dens):
    """
    O + CO + M -> CO2 + M
    
    Assumed to be the same as the main isotope
    """
    
    #O + CO + M -> CO2 + M
    rrates1, rtype1, ns1, sID1, sISO1, sf1, npr1, pID1, pISO1, pf1, ref1 = isochem.reactions.reaction0042(nh, p, t, dens)
    
    rrates = rrates1

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), (13)CO (5)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  2, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # (13)CO2 (2)
    pID[0], pISO[0], pf[0] = 2, 2, 1.0

    ref = "tsang and hampson, 1986"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################


###############################################################################################################################