import numpy as np
from isochem.jit import jit
import inspect,re

#The units used in the reactions are as follows:

# Reaction rate coefficients: s-1 if rtype=1; cm3 s-1 if rtype=2

cache = True

###############################################################################################################################

@jit()
def reaction0001(nh, p, t, dens):
    """
    O + O2 + CO2 -> O3 + CO2
    """

    #Reaction constants
    br = 1.0
    A = 6.0e-34
    n = -2.4
    gamma = 0.0

    #Calculating reaction rates
    rrates = bimolecular(br, A, n, gamma, t) * dens
    rrates *= 2.075  # Scaling factor applied in Mars PCM
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 7,  0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 3, 0, 1.0

    ref = 'sehested et al., j. geophys. res., 100, 1995'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0002(nh, p, t, dens):
    """
    O + O + CO2 -> O2 + CO2
    """

    #Reaction constants
    br = 1.0
    A = 9.46e-34
    n = 0.0
    gamma = -485.0

    #Calculating reaction rates
    rrates = bimolecular(br, A, n, gamma, t) * dens
    rrates *= 2.5  # Scaling factor applied in Mars PCM
 
    # NIST expression: 2.5 * 9.46e-34 * exp(485./t) * dens
    #rrates = 2.5 * 9.46e-34 * np.exp(485.0 / t) * dens
    
    rtype = 2

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 45, 0, 2.0  # O + O

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 7, 0, 1.0  # O2

    ref = 'NIST kinetics database'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0003(nh, p, t, dens):
    """
    O + O3 -> O2 + O2
    """
    A = 8.0e-12
    n = 0.0
    gamma = 2060.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 45, 0, 1.0  # O
    sID[1], sISO[1], sf[1] = 3,  0, 1.0  # O3

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 7, 0, 2.0  # 2 O2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0004(nh, p, t, dens):
    """
    O(1D) + CO2 -> O + CO2
    """
    A = 7.5e-11
    n = 0.0
    gamma = -115.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 133, 0, 1.0  # O(1D)
    sID[1], sISO[1], sf[1] = 2, 0, 1.0  # CO2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 45, 0, 1.0  # O
    pID[1], pISO[1], pf[1] = 2, 0, 1.0   # CO2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0005(nh, p, t, dens):
    """
    O(1D) + H2O -> OH + OH
    """
    A = 1.63e-10
    n = 0.0
    gamma = -60.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 133, 0, 1.0   # O(1D)
    sID[1], sISO[1], sf[1] = 1,   0, 1.0   # H2O

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 13, 0, 2.0   # 2 OH

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0006(nh, p, t, dens):
    """
    O(1D) + H2 -> OH + H
    """
    A = 1.2e-10
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 133, 0, 1.0  # O(1D)
    sID[1], sISO[1], sf[1] = 39,  0, 1.0  # H2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 13, 0, 1.0   # OH
    pID[1], pISO[1], pf[1] = 48, 0, 1.0   # H

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0007(nh, p, t, dens):
    """
    O(1D) + O2 -> O + O2
    """
    A = 3.3e-11
    n = 0.0
    gamma = -55.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 133, 0, 1.0  # O(1D)
    sID[1], sISO[1], sf[1] = 7,   0, 1.0  # O2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 45, 0, 1.0   # O
    pID[1], pISO[1], pf[1] = 7,  0, 1.0   # O2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0008(nh, p, t, dens):
    """
    O(1D) + O3 -> O2 + O2  (branching ratio = 0.5)
    """
    A = 2.4e-10
    n = 0.0
    gamma = 0.0
    br = 0.5

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 133, 0, 1.0  # O(1D)
    sID[1], sISO[1], sf[1] = 3,   0, 1.0  # O3

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 7, 0, 2.0  # 2 O2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0009(nh, p, t, dens):
    """
    O(1D) + O3 -> O2 + O + O   (branching ratio = 0.5)
    """
    A = 2.4e-10
    n = 0.0
    gamma = 0.0
    br = 0.5

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 3,   0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 7,  0, 1.0  # O2
    pID[1], pISO[1], pf[1] = 45, 0, 2.0  # 2 O

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0010(nh, p, t, dens):
    """
    O + HO2 -> OH + O2
    """
    A = 3.0e-11
    n = 0.0
    gamma = -200.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 45, 0, 1.0  # O
    sID[1], sISO[1], sf[1] = 44, 0, 1.0  # HO2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 7,  0, 1.0  # O2
    pID[1], pISO[1], pf[1] = 13, 0, 1.0  # OH

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0011(nh, p, t, dens):
    """
    O + OH -> O2 + H
    """
    A = 1.8e-11
    n = 0.0
    gamma = -180.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 45, 0, 1.0  # O
    sID[1], sISO[1], sf[1] = 13, 0, 1.0  # OH

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 7,  0, 1.0  # O2
    pID[1], pISO[1], pf[1] = 48, 0, 1.0  # H

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0012(nh, p, t, dens):
    """
    H + O3 -> OH + O2
    """
    A = 1.4e-10
    n = 0.0
    gamma = 470.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 48, 0, 1.0  # H
    sID[1], sISO[1], sf[1] = 3,  0, 1.0  # O3

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 13, 0, 1.0  # OH
    pID[1], pISO[1], pf[1] = 7,  0, 1.0  # O2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0013(nh, p, t, dens):
    """
    H + HO2 -> OH + OH
    """
    A = 7.2e-11
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 48, 0, 1.0  # H
    sID[1], sISO[1], sf[1] = 44, 0, 1.0  # HO2

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 13, 0, 2.0  # 2 OH

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0014(nh, p, t, dens):
    """
    H + HO2 -> H2 + O2
    """
    A = 6.9e-12
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 48, 0, 1.0  # H
    sID[1], sISO[1], sf[1] = 44, 0, 1.0  # HO2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 39, 0, 1.0  # H2
    pID[1], pISO[1], pf[1] = 7,  0, 1.0  # O2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0015(nh, p, t, dens):
    """
    H + HO2 -> H2O + O
    """
    A = 1.6e-12
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 48, 0, 1.0  # H
    sID[1], sISO[1], sf[1] = 44, 0, 1.0  # HO2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1,  0, 1.0  # H2O
    pID[1], pISO[1], pf[1] = 45, 0, 1.0  # O

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0016(nh, p, t, dens):
    """
    OH + HO2 -> H2O + O2
    """
    A = 4.8e-11
    n = 0.0
    gamma = -250.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 13, 0, 1.0  # OH
    sID[1], sISO[1], sf[1] = 44, 0, 1.0  # HO2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1, 0, 1.0   # H2O
    pID[1], pISO[1], pf[1] = 7, 0, 1.0   # O2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0017(nh, p, t, dens):
    """
    HO2 + HO2 -> H2O2 + O2
    """
    A = 3.0e-13
    n = 0.0
    gamma = -460.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 2

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 44, 0, 2.0  # 2 HO2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # H2O2, O2
    pID[0], pISO[0], pf[0] = 25, 0, 1.0
    pID[1], pISO[1], pf[1] = 7,  0, 1.0

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0018(nh, p, t, dens):
    """
    OH + H2O2 -> H2O + HO2
    """
    A = 1.8e-12
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 13, 0, 1.0  # OH
    sID[1], sISO[1], sf[1] = 25, 0, 1.0  # H2O2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1,  0, 1.0  # H2O
    pID[1], pISO[1], pf[1] = 44, 0, 1.0  # HO2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0019(nh, p, t, dens):
    """
    OH + H2 -> H2O + H
    """
    A = 2.8e-12
    n = 0.0
    gamma = 1800.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 13, 0, 1.0  # OH
    sID[1], sISO[1], sf[1] = 39, 0, 1.0  # H2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1,  0, 1.0  # H2O
    pID[1], pISO[1], pf[1] = 48, 0, 1.0  # H

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0020(nh, p, t, dens):
    """
    H + O2 + CO2 -> HO2 + CO2
    """

    k0 = 2.4 * 5.3e-32 # factor 2.4 in front in the code from Mars PCM
    n = 1.8
    kinf = 9.5e-11
    m = -0.4

    rrates = termolecular(k0, n, kinf, m, t, dens)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 48, 0, 1.0  # H
    sID[1], sISO[1], sf[1] = 7,  0, 1.0  # O2

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 44, 0, 1.0  # HO2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0021(nh, p, t, dens):
    """
    O + H2O2 -> OH + HO2
    """
    A = 1.4e-12
    n = 0.0
    gamma = 2000.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 45, 0, 1.0  # O
    sID[1], sISO[1], sf[1] = 25, 0, 1.0  # H2O2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # OH, HO2
    pID[0], pISO[0], pf[0] = 13, 0, 1.0
    pID[1], pISO[1], pf[1] = 44, 0, 1.0

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0022(nh, p, t, dens):
    """
    OH + OH -> H2O + O
    """
    A = 1.8e-12
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 2

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 13, 0, 2.0  # 2 OH

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # H2O, O
    pID[0], pISO[0], pf[0] = 1,  0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0023(nh, p, t, dens):
    """
    OH + O3 -> HO2 + O2
    """
    A = 1.7e-12
    n = 0.0
    gamma = 940.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 13, 0, 1.0  # OH
    sID[1], sISO[1], sf[1] = 3,  0, 1.0  # O3

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HO2, O2
    pID[0], pISO[0], pf[0] = 44, 0, 1.0
    pID[1], pISO[1], pf[1] = 7,  0, 1.0

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0024(nh, p, t, dens):
    """
    HO2 + O3 -> OH + O2 + O2
    """
    A = 1.0e-14
    n = 0.0
    gamma = 490.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 44, 0, 1.0  # HO2
    sID[1], sISO[1], sf[1] = 3,  0, 1.0  # O3

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # OH, 2 O2
    pID[0], pISO[0], pf[0] = 13, 0, 1.0
    pID[1], pISO[1], pf[1] = 7,  0, 2.0

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0025(nh, p, t, dens):
    """
    HO2 + HO2 + CO2 -> H2O2 + O2 + CO2
    """
    A = 2.1e-33
    n = 0.0
    gamma = -920.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t) * dens
    rrates *= 2.5  # factor 2.5 in front in the code from Mars PCM

    rtype = 2

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 44, 0, 2.0  # 2 HO2

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # H2O2, O2
    pID[0], pISO[0], pf[0] = 25, 0, 1.0
    pID[1], pISO[1], pf[1] = 7,  0, 1.0

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0026(nh, p, t, dens):
    """
    OH + OH + CO2 -> H2O2 + CO2
    """
    k0 = 2.5 * 6.9e-31
    n  = 1.0
    kinf = 2.6e-11
    m  = 0.0

    rrates = termolecular(k0, n, kinf, m, t, dens)

    rtype = 2

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)
    
    sID[0], sISO[0], sf[0] = 13, 0, 2.0  # 2 OH

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 25, 0, 1.0  # H2O2

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0027(nh, p, t, dens):
    """
    H + H + CO2 -> H2 + CO2
    """

    rrates = 2.5 * 1.8e-30 * (t**(-1.0)) * dens

    rtype = 2

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # 2 H
    sID[0], sISO[0], sf[0] = 48, 0, 2.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 39, 0, 1.0  # H2

    ref = 'Baulch et al., 2005'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0028(nh, p, t, dens):
    """
    O + NO2 + M -> NO + O2 + M
    [the code uses a 'chemical activation' approach with partial falloff]
    """
    k0 = 2.5 * 3.4e-31
    n = 1.6
    kinf = 2.3e-11
    m = 0.2
    A = 5.3e-12
    B = -200.0

    rrates = chemical_activation(k0, n, kinf, m, A, B, t, dens)

    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O, NO2
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO, O2
    pID[0], pISO[0], pf[0] = 8, 0, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0029(nh, p, t, dens):
    """
    NO + O3 -> NO2 + O2
    """
    A = 3.0e-12
    n = 0.0
    gamma = 1500.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO, O3
    sID[0], sISO[0], sf[0] = 8, 0, 1.0
    sID[1], sISO[1], sf[1] = 3, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO2, O2
    pID[0], pISO[0], pf[0] = 10, 0, 1.0
    pID[1], pISO[1], pf[1] = 7,  0, 1.0

    ref = 'JPL 2006'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0030(nh, p, t, dens):
    """
    NO + HO2 -> NO2 + OH
    """
    A = 3.44e-12
    n = 0.0
    gamma = -260.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO, HO2
    sID[0], sISO[0], sf[0] = 8, 0, 1.0
    sID[1], sISO[1], sf[1] = 44, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO2, OH
    pID[0], pISO[0], pf[0] = 10, 0, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = 'JPL 2011'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0031(nh, p, t, dens):
    """
    N + NO -> N2 + O
    """
    A = 2.1e-11
    n = 0.0
    gamma = -100.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N, NO
    sID[0], sISO[0], sf[0] = 47, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N2, O
    pID[0], pISO[0], pf[0] = 22, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###########################################################################################################

@jit()
def reaction0032(nh, p, t, dens):
    """
    N + O2 -> NO + O
    """
    A = 3.3e-12
    n = 0.0
    gamma = 3150.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N (47), O2 (7)
    sID[0], sISO[0], sf[0] = 47, 0, 1.0
    sID[1], sISO[1], sf[1] = 7,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), O (45)
    pID[0], pISO[0], pf[0] = 8,  0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0033(nh, p, t, dens):
    """
    NO2 + H -> NO + OH
    """
    A = 1.35e-10
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), H (48)
    sID[0], sISO[0], sf[0] = 10, 0, 1.0
    sID[1], sISO[1], sf[1] = 48, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), OH (13)
    pID[0], pISO[0], pf[0] = 8,  0, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0034(nh, p, t, dens):
    """
    N + O -> NO
    """

    A = 2.8e-17
    n = -0.5
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N (47), O (45)
    sID[0], sISO[0], sf[0] = 47, 0, 1.0
    sID[1], sISO[1], sf[1] = 45, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8)
    pID[0], pISO[0], pf[0] = 8, 0, 1.0

    ref = "JPL 2011"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0035(nh, p, t, dens):
    """
    N + HO2 -> NO + OH
    """
    
    A = 2.19e-11
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N (47), HO2 (44)
    sID[0], sISO[0], sf[0] = 47, 0, 1.0
    sID[1], sISO[1], sf[1] = 44, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), OH (13)
    pID[0], pISO[0], pf[0] = 8,  0, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = "brune et al., j. chem. phys., 87, 1983"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0036(nh, p, t, dens):
    """
    N + OH -> NO + H
    """

    A = 3.8e-11
    n = 0.0
    gamma = -85.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N (47), OH (13)
    sID[0], sISO[0], sf[0] = 47, 0, 1.0
    sID[1], sISO[1], sf[1] = 13, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), H (48)
    pID[0], pISO[0], pf[0] = 8,  0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = "atkinson et al., j. phys. chem. ref. data, 18, 881, 1989"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0037(nh, p, t, dens):
    """
    N(2D) + O -> N + O
    (Here 'o' is an array representing O-atom concentration vs altitude)
    """

    A = 3.3e-12
    n = 0.0
    gamma = 260.
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N(2D) (134) , O (45)
    sID[0], sISO[0], sf[0] = 134, 0, 1.0
    sID[1], sISO[1], sf[1] = 45, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N (47) , O (45)
    pID[0], pISO[0], pf[0] = 47, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "herron, j. phys. chem. ref. data, 1999"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0038(nh, p, t, dens):
    """
    N(2D) + N2 -> N + N2
    """
    A = 1.7e-14
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N(2D) (134) , N2 (22)
    sID[0], sISO[0], sf[0] = 134, 0, 1.0
    sID[1], sISO[1], sf[1] = 22,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N (47) , N2 (22)
    pID[0], pISO[0], pf[0] = 47, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 0, 1.0


    ref = "herron, j. phys. chem. ref. data, 1999"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0039(nh, p, t, dens):
    """
    N(2D) + CO2 -> NO + CO
    """
    # Constant rate
    A = 3.6e-13
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N(2D) (134), CO2 (2)
    sID[0], sISO[0], sf[0] = 134, 0, 1.0
    sID[1], sISO[1], sf[1] = 2,   0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), CO (5)
    pID[0], pISO[0], pf[0] = 8, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = "herron, j. phys. chem. ref. data, 1999"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0040(nh, p, t, dens):
    """
    OH + CO -> CO2 + H
    """
    k0 = 2.5 * 6.9e-33
    n = 2.1
    kinf = 1.1e-12
    m = -1.3
    A = 1.85e-13
    B = 65.0

    rrates = chemical_activation(k0, n, kinf, m, A, B, t, dens)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), CO (5)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # CO2 (2), H (48)
    pID[0], pISO[0], pf[0] = 2, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0041(nh, p, t, dens):
    """
    OH + CO -> HOCO
    """
    k0 = 2.5 * 6.9e-33
    n = 2.1
    kinf = 1.1e-12
    m = -1.3
    A = 1.85e-13
    B = 65.0

    rrates = termolecular(k0, n, kinf, m, t, dens)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), CO (5)
    sID[0], sISO[0], sf[0] = 13, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HOCO (80)
    pID[0], pISO[0], pf[0] = 80, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0042(nh, p, t, dens):
    """
    O + CO + M -> CO2 + M
    """
    A = 6.5e-33
    n = 0.0
    gamma = 2184.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t) * dens
    rrates *= 2.5  # factor 2.5 in front in the code from Mars PCM

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), CO (5)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # CO2 (2)
    pID[0], pISO[0], pf[0] = 2, 0, 1.0

    ref = "tsang and hampson, 1986"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0043(nh, p, t, dens):
    """
    O(1D) + N2 + M -> N2O + M
    """

    A = 2.8e-36
    n = -0.9
    br = 1.0
    gamma = 0.0

    rrates = bimolecular(br, A, n, gamma, t) * dens

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133), N2 (22)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 22,  0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N2O (4)
    pID[0], pISO[0], pf[0] = 4, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0044(nh, p, t, dens):
    """
    O + NO + CO2 -> NO2 + CO2
    """
    k0 = 2.5 * 9.1e-32  # factor 2.5 in front in the code from Mars PCM
    n = 1.5
    kinf = 3.0e-11
    m = 0.0

    rrates = termolecular(k0, n, kinf, m, t, dens)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), NO (8)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO2 (10)
    pID[0], pISO[0], pf[0] = 10, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0045(nh, p, t, dens):
    """
    O(1D) + N2 -> O + N2
    """
    A = 2.15e-11
    n = 0.0
    gamma = 110.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133), N2 (22)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 22,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # O (45), N2 (22)
    pID[0], pISO[0], pf[0] = 45, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0046(nh, p, t, dens):
    """
    O(1D) + N2O -> N2 + O2
    """
    A = 1.19e-10
    n = 0.0
    gamma = -20.0
    br = 0.39

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133), N2O (4)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 4,   0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N2 (22), O2 (7)
    pID[0], pISO[0], pf[0] = 22, 0, 1.0
    pID[1], pISO[1], pf[1] = 7,  0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0047(nh, p, t, dens):
    """
    O(1D) + N2O -> NO + NO
    """
    A = 1.19e-10
    n = 0.0
    gamma = -20.0
    br = 0.61

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133), N2O (4)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0
    sID[1], sISO[1], sf[1] = 4,   0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 2 NO (8)
    pID[0], pISO[0], pf[0] = 8, 0, 2.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0048(nh, p, t, dens):
    """
    O + NO2 + M -> NO3 + M
    """
    k0 = 2.5 * 3.4e-31
    n = 1.6
    kinf = 2.3e-11
    m = 0.2
    A = 5.3e-12
    B = -200.0

    rrates = termolecular(k0, n, kinf, m, t, dens)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), NO2 (10)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO3 (91)
    pID[0], pISO[0], pf[0] = 91, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0049(nh, p, t, dens):
    """
    O + NO3 -> O2 + NO2
    """
    A = 1.3e-11
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), NO3 (91)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 91, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # O2 (7), NO2 (10)
    pID[0], pISO[0], pf[0] = 7,  0, 1.0
    pID[1], pISO[1], pf[1] = 10, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0050(nh, p, t, dens):
    """
    N + NO2 -> N2O + O
    """
    A = 5.8e-12
    n = 0.0
    gamma = -220.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N (47), NO2 (10)
    sID[0], sISO[0], sf[0] = 47, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N2O (4), O (45)
    pID[0], pISO[0], pf[0] = 4,  0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0051(nh, p, t, dens):
    """
    NO + NO3 -> NO2 + NO2
    """
    A = 1.7e-11
    n = 0.0
    gamma = -125.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO (8), NO3 (91)
    sID[0], sISO[0], sf[0] = 8,  0, 1.0
    sID[1], sISO[1], sf[1] = 91, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 2 NO2 (10)
    pID[0], pISO[0], pf[0] = 10, 0, 2.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0052(nh, p, t, dens):
    """
    NO2 + O3 -> NO3 + O2
    """
    A = 1.2e-13
    n = 0.0
    gamma = 2450.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), O3 (3)
    sID[0], sISO[0], sf[0] = 10, 0, 1.0
    sID[1], sISO[1], sf[1] = 3,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO3 (91), O2 (7)
    pID[0], pISO[0], pf[0] = 91, 0, 1.0
    pID[1], pISO[1], pf[1] = 7,  0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0053(nh, p, t, dens):
    """
    NO3 + NO3 -> 2NO2 + O2
    """
    A = 8.5e-13
    n = 0.0
    gamma = 2450.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 2

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # 2 NO3 (91)
    sID[0], sISO[0], sf[0] = 91, 0, 2.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # 2 NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] = 10, 0, 2.0
    pID[1], pISO[1], pf[1] = 7,  0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0054(nh, p, t, dens):
    """
    O2 + HOCO -> HO2 + CO2
    """
    A = 2.0e-12
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O2 (7), HOCO (80)
    sID[0], sISO[0], sf[0] = 7,   0, 1.0
    sID[1], sISO[1], sf[1] = 80,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HO2 (44), CO2 (2)
    pID[0], pISO[0], pf[0] = 44, 0, 1.0
    pID[1], pISO[1], pf[1] = 2,  0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0055(nh, p, t, dens):
    """
    O + H2 -> OH + H
    """

    A = 1.6e-11
    n = 0.0
    gamma = 4570.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns   = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), H2 (39)
    sID[0], sISO[0], sf[0] = 45,  0, 1.0
    sID[1], sISO[1], sf[1] = 39,  0, 1.0

    npr  = 2
    pID  = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf   = np.zeros(4, dtype=np.float64)

    # OH (13), H (48)
    pID[0], pISO[0], pf[0] = 13, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0056(nh, p, t, dens):
    """
    N + O3 -> NO + O2
    """

    A = 1.0e-16
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns   = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N (47), O3 (3)
    sID[0], sISO[0], sf[0] = 47,  0, 1.0
    sID[1], sISO[1], sf[1] =  3,  0, 1.0

    npr  = 2
    pID  = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf   = np.zeros(4, dtype=np.float64)

    # NO (8), O2 (7)
    pID[0], pISO[0], pf[0] =  8, 0, 1.0
    pID[1], pISO[1], pf[1] =  7, 0, 1.0

    ref = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0057(nh, p, t, dens):
    """
    N(2D) + NO -> N2 + O
    """
    # Constant rate
    A = 6.9e-11
    n = 0.0
    gamma = 0.0
    br = 1.0

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N(2D) (134), NO (8)
    sID[0], sISO[0], sf[0] = 134, 0, 1.0
    sID[1], sISO[1], sf[1] =   8,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N2 (22), O (45)
    pID[0], pISO[0], pf[0] = 22, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0058(nh, p, t, dens):
    """
    H + NO3 -> OH + NO2
    """
    # Constant rate
    A = 1.1e-10 # A_Factor
    n  = 0.0
    gamma = 0.0     # E/R
    br = 1.0        # Braching Ratio

    rrates = bimolecular(br, A, n, gamma, t)
    
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # H (48), NO3 (91)
    sID[0], sISO[0], sf[0] =  48, 0, 1.0
    sID[1], sISO[1], sf[1] =  91, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # OH (13), NO2 (10)
    pID[0], pISO[0], pf[0] =  13, 0, 1.0
    pID[1], pISO[1], pf[1] =  10, 0, 1.0

    ref = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0059(nh, p, t, dens):
    """
    OH + NO + M -> HONO + M 
    """
    # Constant rate
    k0 = 7.1e-31
    n =  2.6
    kinf = 3.6e-11
    m = 0.1

    rrates = termolecular(k0, n, kinf, m, t, dens)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), NO (8)
    sID[0], sISO[0], sf[0] =  13, 0, 1.0
    sID[1], sISO[1], sf[1] =   8, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HONO (138)
    pID[0], pISO[0], pf[0] = 138, 0, 1.0


    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0060(nh, p, t, dens):
    """
    OH + NO2 + M-> HNO3 + M
    """
    # Constant rate
    
    #First chanel OH + NO2 -> HONO2
    k01 = 1.8e-30
    n1 =  3.0
    kinf1 = 2.8e-11
    m1 = 0

    rrates1 = termolecular(k01, n1, kinf1, m1, t, dens)
    
    #Second chanel OH + NO2 -> HOONO
    k02 = 9.3e-32
    n2 =  3.9
    kinf2 = 4.2e-11
    m2 = 0.5 

    rrates2 = termolecular(k02, n2, kinf2, m2, t, dens)

    rrates = rrates1 + rrates2

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), NO2 (10)
    sID[0], sISO[0], sf[0] =  13, 0, 1.0
    sID[1], sISO[1], sf[1] =  10, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HNO3 (12)
    pID[0], pISO[0], pf[0] = 12, 0, 1.0


    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################


@jit()
def reaction0061(nh, p, t, dens):
    """
    OH + NO3 -> HO2 + NO2
    """
    # Constant rate
    A = 2.0e-11 # A_Factor
    n  = 0.0
    gamma = 0.0     # E/R
    br = 1.0        # Braching Ratio

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), NO3 (91)
    sID[0], sISO[0], sf[0] =  13, 0, 1.0
    sID[1], sISO[1], sf[1] =  91, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HO2 (44), NO2 (10)
    pID[0], pISO[0], pf[0] = 44, 0, 1.0
    pID[1], pISO[1], pf[1] = 10, 0, 1.0


    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0062(nh, p, t, dens):
    """
    OH + HONO -> H2O + NO2
    """
    # Constant rate
    alpha = 3.0e-12 # A_Factor
    beta  = 0.0
    gamma = -250     # E/R
    br = 1.0        # Braching Ratio 

    rrates = bimolecular(br, alpha, beta, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), HONO (138)
    sID[0], sISO[0], sf[0] =  13, 0, 1.0
    sID[1], sISO[1], sf[1] = 138, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # H2O (1), NO2 (10)
    pID[0], pISO[0], pf[0] =  1, 0, 1.0
    pID[1], pISO[1], pf[1] = 10, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0063(nh, p, t, dens):
    """
    OH + HNO3 -> H2O + NO3
    """

    A = 7.2e-15 # A_Factor
    n  = 0.0
    gamma = -785.0     # E/R
    br = 1.0        # Braching Ratio

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), HNO3 (12)
    sID[0], sISO[0], sf[0] =  13, 0, 1.0
    sID[1], sISO[1], sf[1] =  12, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # H2O (1), NO3 (91)
    pID[0], pISO[0], pf[0] =  1, 0, 1.0
    pID[1], pISO[1], pf[1] = 91, 0, 1.0

    ref = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0064(nh, p, t, dens):
    """
    OH + HO2NO2 -> H2O + NO2 + O2
    """
    # Constant rate
    alpha = 4.5e-13 # A_Factor
    beta  = 0.0
    gamma = -610     # E/R
    br = 1.0        # Braching Ratio 

    rrates = bimolecular(br, alpha, beta, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # OH (13), HO2NO2 (137)
    sID[0], sISO[0], sf[0] =   13, 0, 1.0
    sID[1], sISO[1], sf[1] =  137, 0, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # H2O (1), NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] =  1, 0, 1.0
    pID[1], pISO[1], pf[1] = 10, 0, 1.0
    pID[2], pISO[2], pf[2] =  7, 0, 1.0

    ref = "JPL2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0065(nh, p, t, dens):
    """
    HO2 + NO2 + M -> HO2NO2 + M
    """
    # Constant rate
    k0 = 1.9e-31
    n =  3.4
    kinf = 4.0e-12
    m = 0.3

    rrates = termolecular(k0, n, kinf, m, t, dens)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # HO2 (44), NO2 (10)
    sID[0], sISO[0], sf[0] =   44, 0, 1.0
    sID[1], sISO[1], sf[1] =   10, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # HO2NO2 (1)
    pID[0], pISO[0], pf[0] = 137, 0, 1.0

    ref = "JPL2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0066(nh, p, t, dens):
    """
    HO2 + NO3 -> O2 + HNO3
    """
    # Constant rate
    A = 3.5e-12 # A_Factor
    n = 0.0
    gamma = 0.0     # E/R
    br = 0.3  #Mellouki et al. (1993) - they determined the branching ratio

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # HO2 (44), NO3 (91)
    sID[0], sISO[0], sf[0] =   44, 0, 1.0
    sID[1], sISO[1], sf[1] =   91, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # O2 (7), HNO3 (12)
    pID[0], pISO[0], pf[0] =   7, 0, 1.0
    pID[1], pISO[1], pf[1] =  12, 0, 1.0

    ref = "Mellouki et al. (1993)"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref


###############################################################################################################################

@jit()
def reaction0067(nh, p, t, dens):
    """
    HO2 + NO3 -> OH + NO2 + O2
    """
    # Constant rate
    A = 3.5e-12 # A_Factor
    n = 0.0
    gamma = 0.0     # E/R
    br = 0.7  #Mellouki et al. (1993) - they determined the branching ratio

    rrates = bimolecular(br, A, n, gamma, t)


    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # HO2 (44), NO3 (91)
    sID[0], sISO[0], sf[0] =   44, 0, 1.0
    sID[1], sISO[1], sf[1] =   91, 0, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # OH (13), NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] =  13, 0, 1.0
    pID[1], pISO[1], pf[1] =  10, 0, 1.0
    pID[2], pISO[2], pf[2] =  7, 0, 1.0

    ref = "Mellouki et al. (1993)"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0068(nh, p, t, dens):
    """
    NO2 + O3 -> NO3 + O2
    """
    # Constant rate
    alpha = 1.2e-13 # A_Factor
    beta  = 0.0
    gamma = 2450    # E/R
    br = 1.0        # Braching Ratio 

    rrates = bimolecular(br, alpha, beta, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), O3 (3)
    sID[0], sISO[0], sf[0] =   10, 0, 1.0
    sID[1], sISO[1], sf[1] =    3, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO3 (91), O2 (7)
    pID[0], pISO[0], pf[0] =  91, 0, 1.0
    pID[1], pISO[1], pf[1] =   7, 0, 1.0

    ref = "JPL2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0069(nh, p, t, dens):
    """
    NO2 + NO3 + M -> N2O5 + M
    """
    # Constant rate
    k0 = 2.4e-30
    n =  3.0
    kinf = 1.6e-12
    m = -0.1

    rrates = termolecular(k0, n, kinf, m, t, dens)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 0, 1.0
    sID[1], sISO[1], sf[1] =   91, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N2O5 (139)
    pID[0], pISO[0], pf[0] =   139, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0070(nh, p, t, dens):
    """
    NO2 + NO3  -> NO + NO2 + O2
    """
    # Constant rate
    A = 8.2e-14 # A_Factor
    n = 0.0
    gamma = 1480  # E/R
    br = 1.0        # Braching Ratio 

    rrates = bimolecular(br, A, n, gamma, t)

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # NO2 (10), NO3 (91)
    sID[0], sISO[0], sf[0] =   10, 0, 1.0
    sID[1], sISO[1], sf[1] =   91, 0, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), NO2 (10), O2 (7)
    pID[0], pISO[0], pf[0] =   8, 0, 1.0
    pID[1], pISO[1], pf[1] =  10, 0, 1.0
    pID[2], pISO[2], pf[2] =   7, 0, 1.0

    ref = "Yung, Y. L., & DeMore, W. B. (1999). Photochemistry of planetary atmospheres. Oxford University Press."

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref





###############################################################################################################################

@jit()
def reaction0071(nh, p, ti, dens):
    """
    CO2+ + O2 -> O2+ + CO2
    """

    #Reaction constants
    br = 1.0
    A = 5.5e-11
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1002, 0, 1.0
    sID[1], sISO[1], sf[1] = 7,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1007, 0, 1.0
    pID[1], pISO[1], pf[1] = 2, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref


###############################################################################################################################

@jit()
def reaction0072(nh, p, ti, dens):
    """
    CO2+ + O -> O+ + CO2
    """

    #Reaction constants
    br = 1.0
    A = 9.6e-11
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1002, 0, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1045, 0, 1.0
    pID[1], pISO[1], pf[1] = 2, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref


###############################################################################################################################

@jit()
def reaction0073(nh, p, ti, dens):
    """
    CO2+ + O -> O2+ + CO
    """

    #Reaction constants
    br = 1.0
    A = 1.64e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1002, 0, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1007, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0074(nh, p, ti, dens):
    """
    O2+ + e- -> O + O
    """

    #Reaction constants
    br = 1.0
    A = 2.0e-7
    n = -0.7
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1007, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 45, 0, 2.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0075(nh, p, ti, dens):
    """
    O+ + CO2 -> O2+ + CO
    """

    #Reaction constants
    br = 1.0
    A = 9.4e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1045, 0, 1.0
    sID[1], sISO[1], sf[1] = 2,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1007, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0076(nh, p, ti, dens):
    """
    CO2+ + e- -> CO + O
    """

    #Reaction constants
    br = 1.0
    A = 3.8e-7
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1002, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 5, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0077(nh, p, ti, dens):
    """
    CO2+ + NO -> NO+ + CO2
    """

    #Reaction constants
    br = 1.0
    A = 1.2e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1002, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 2, 0, 1.0
    pID[1], pISO[1], pf[1] = 1008, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0078(nh, p, ti, dens):
    """
    O2+ + NO -> NO+ + O2
    """

    #Reaction constants
    br = 1.0
    A = 4.6e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1007, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0079(nh, p, ti, dens):
    """
    O2+ + N2 -> NO+ + NO
    """

    #Reaction constants
    br = 1.0
    A = 1.0e-15
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1007, 0, 1.0
    sID[1], sISO[1], sf[1] = 22,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 8, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0080(nh, p, ti, dens):
    """
    O2+ + N -> NO+ + O
    """

    #Reaction constants
    br = 1.0
    A = 1.0e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1007, 0, 1.0
    sID[1], sISO[1], sf[1] = 47,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0081(nh, p, ti, dens):
    """
    O+ + N2 -> NO+ + N
    """

    #Reaction constants
    br = 1.0
    A = 1.2e-12
    n = -0.45
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1045, 0, 1.0
    sID[1], sISO[1], sf[1] = 22,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 47, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0082(nh, p, ti, dens):
    """
    NO+ + e- -> N + O
    """

    #Reaction constants
    br = 1.0
    A = 4.3e-7
    n = -0.37
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1008, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 47, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0083(nh, p, ti, dens):
    """
    CO+ + CO2 -> CO2+ + CO
    """

    #Reaction constants
    br = 1.0
    A = 1.0e-9
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1005, 0, 1.0
    sID[1], sISO[1], sf[1] = 2,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1002, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0084(nh, p, ti, dens):
    """
    CO+ + O -> O+ + CO
    """

    #Reaction constants
    br = 1.0
    A = 1.4e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1005, 0, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1045, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0085(nh, p, ti, dens):
    """
    C+ + CO2 -> CO+ + CO
    """

    #Reaction constants
    br = 1.0
    A = 1.1e-9
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1046, 0, 1.0
    sID[1], sISO[1], sf[1] = 2,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1005, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0086(nh, p, ti, dens):
    """
    N2+ + CO2 -> CO2+ + N2
    """

    #Reaction constants
    br = 1.0
    A = 9.0e-10
    n = -0.23
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 0, 1.0
    sID[1], sISO[1], sf[1] = 2,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1002, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0087(nh, p, ti, dens):
    """
    N2+ + O -> NO+ + N
    """

    #Reaction constants
    br = 1.0
    A = 1.33e-10
    n = -0.44
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 0, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 47, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0088(nh, p, ti, dens):
    """
    N2+ + CO -> CO+ + N2
    """

    #Reaction constants
    br = 1.0
    A = 7.4e-11
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1005, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0089(nh, p, ti, dens):
    """
    N2+ + e– -> N + N
    """

    #Reaction constants
    br = 1.0
    A = 1.7e-7
    n = -0.3
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 47, 0, 2.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0090(nh, p, ti, dens):
    """
    N2+ + O -> O+ + N2
    """

    #Reaction constants
    br = 1.0
    A = 7.0e-12
    n = -0.23
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 0, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1045, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0091(nh, p, ti, dens):
    """
    N+ + CO2 -> CO2+ + N
    """

    #Reaction constants
    br = 1.0
    A = 7.5e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1047, 0, 1.0
    sID[1], sISO[1], sf[1] = 2,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1002, 0, 1.0
    pID[1], pISO[1], pf[1] = 47, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0092(nh, p, ti, dens):
    """
    CO+ + H -> H+ + CO
    """

    #Reaction constants
    br = 1.0
    A = 4.0e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1005, 0, 1.0
    sID[1], sISO[1], sf[1] = 48,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1048, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0093(nh, p, ti, dens):
    """
    O+ + H -> H+ + O
    """

    #Reaction constants
    br = 1.0
    A = 5.66e-10
    n = 0.36
    gamma = -8.6

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1045, 0, 1.0
    sID[1], sISO[1], sf[1] = 48,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1048, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0094(nh, p, ti, dens):
    """
    H+ + O -> O+ + H
    """

    #Reaction constants
    br = 1.0
    A = 6.86e-10
    n = 0.26
    gamma = 224.3

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1045, 0, 1.0
    sID[1], sISO[1], sf[1] = 48,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1048, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0095(nh, p, ti, dens):
    """
    CO2+ + H2 -> HCO2+ + H
    """

    #Reaction constants
    br = 1.0
    A = 9.5e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1002, 0, 1.0
    sID[1], sISO[1], sf[1] = 39,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1080, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0096(nh, p, ti, dens):
    """
    HCO2+ + e– -> H + O + CO
    """

    #Reaction constants
    br = 1.0
    A = 8.1e-7
    n = -0.64
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1080, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000, 0, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 48, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0
    pID[2], pISO[2], pf[2] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0097(nh, p, ti, dens):
    """
    HCO2+ + e- -> OH + CO
    """

    #Reaction constants
    br = 1.0
    A = 3.2e-7
    n = -0.64
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1080, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 13, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0098(nh, p, ti, dens):
    """
    HCO2+ + e- -> H + CO2
    """

    #Reaction constants
    br = 1.0
    A = 6.0e-8
    n = -0.64
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1080, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 48, 0, 1.0
    pID[1], pISO[1], pf[1] = 2, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0099(nh, p, ti, dens):
    """
    HCO2+ + O -> HCO+ + O2
    """

    #Reaction constants
    br = 1.0
    A = 1.0e-9
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1080, 0, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1081, 0, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0100(nh, p, ti, dens):
    """
    HCO2+ + CO -> HCO+ + CO2
    """

    #Reaction constants
    br = 1.0
    A = 7.8e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1080, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1081, 0, 1.0
    pID[1], pISO[1], pf[1] = 2, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0101(nh, p, ti, dens):
    """
    H+ + CO2 -> HCO+ + O
    """

    #Reaction constants
    br = 1.0
    A = 3.5e-9
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1048, 0, 1.0
    sID[1], sISO[1], sf[1] = 2,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1081, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0102(nh, p, ti, dens):
    """
    CO2+ + H -> HCO+ + O
    """

    #Reaction constants
    br = 1.0
    A = 4.5e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1002, 0, 1.0
    sID[1], sISO[1], sf[1] = 48,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1081, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0103(nh, p, ti, dens):
    """
    CO+ + H2 -> HCO+ + H
    """

    #Reaction constants
    br = 1.0
    A = 7.5e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1005, 0, 1.0
    sID[1], sISO[1], sf[1] = 39,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1081, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0104(nh, p, ti, dens):
    """
    HCO+ + e- -> CO + H
    """

    #Reaction constants
    br = 1.0
    A = 2.4e-7
    n = -0.69
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1081, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 5, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0105(nh, p, ti, dens):
    """
    CO2+ + H2O -> H2O+ + CO2
    """

    #Reaction constants
    br = 1.0
    A = 2.04e-9
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1002, 0, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1001, 0, 1.0
    pID[1], pISO[1], pf[1] = 2, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0106(nh, p, ti, dens):
    """
    CO+ + H2O -> H2O+ + CO
    """

    #Reaction constants
    br = 1.0
    A = 1.72e-9
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1005, 0, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1001, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0107(nh, p, ti, dens):
    """
    O+ + H2O → H2O+ + O
    """

    #Reaction constants
    br = 1.0
    A = 3.2e-9
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1045, 0, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1001, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0108(nh, p, ti, dens):
    """
    N2+ + H2O -> H2O+ + N2
    """

    #Reaction constants
    br = 1.0
    A = 2.3e-9
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1022, 0, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1001, 0, 1.0
    pID[1], pISO[1], pf[1] = 22, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0109(nh, p, ti, dens):
    """
    N+ + H2O -> H2O+ + N
    """

    #Reaction constants
    br = 1.0
    A = 2.8e-9
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1047, 0, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1001, 0, 1.0
    pID[1], pISO[1], pf[1] = 47, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0110(nh, p, ti, dens):
    """
    H+ + H2O -> H2O+ + H
    """

    #Reaction constants
    br = 1.0
    A = 6.9e-9
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1048, 0, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1001, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0111(nh, p, ti, dens):
    """
    H2O+ + O2 -> O2+ + H2O
    """

    #Reaction constants
    br = 1.0
    A = 4.6e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 7,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1007, 0, 1.0
    pID[1], pISO[1], pf[1] = 1, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0112(nh, p, ti, dens):
    """
    H2O+ + CO -> HCO+ + OH
    """

    #Reaction constants
    br = 1.0
    A = 5.0e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1081, 0, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0113(nh, p, ti, dens):
    """
    H2O+ + O -> O2+ + H2
    """

    #Reaction constants
    br = 1.0
    A = 4.0e-11
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1007, 0, 1.0
    pID[1], pISO[1], pf[1] = 39, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0114(nh, p, ti, dens):
    """
    H2O+ + NO -> NO+ + H2O
    """

    #Reaction constants
    br = 1.0
    A = 2.7e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 1, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0115(nh, p, ti, dens):
    """
    H2O+ + e- -> H + H + O
    """

    #Reaction constants
    br = 1.0
    A = 3.05e-7
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 48, 0, 2.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0116(nh, p, ti, dens):
    """
    H2O+ + e- -> H + OH
    """

    #Reaction constants
    br = 1.0
    A = 8.6e-8
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 48, 0, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0117(nh, p, ti, dens):
    """
    H2O+ + e- -> H2 + O
    """

    #Reaction constants
    br = 1.0
    A = 3.9e-8
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 39, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0118(nh, p, ti, dens):
    """
    H2O+ + H2O -> H3O+ + OH
    """

    #Reaction constants
    br = 1.0
    A = 2.1e-9
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1100, 0, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0119(nh, p, ti, dens):
    """
    H2O+ + H2 -> H3O+ + H
    """

    #Reaction constants
    br = 1.0
    A = 6.4e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1001, 0, 1.0
    sID[1], sISO[1], sf[1] = 39,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1100, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0120(nh, p, ti, dens):
    """
    HCO+ + H2O -> H3O+ + CO
    """

    #Reaction constants
    br = 1.0
    A = 2.5e-9
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1081, 0, 1.0
    sID[1], sISO[1], sf[1] = 1,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1100, 0, 1.0
    pID[1], pISO[1], pf[1] = 5, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0121(nh, p, ti, dens):
    """
    H3O+ + e- -> OH + H + H
    """

    #Reaction constants
    br = 1.0
    A = 3.05e-7
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1100, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 13, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 2.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0122(nh, p, ti, dens):
    """
    H3O+ + e- -> H2O + H
    """

    #Reaction constants
    br = 1.0
    A = 7.09e-8
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1100, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0123(nh, p, ti, dens):
    """
    H3O+ + e- -> OH + H2
    """

    #Reaction constants
    br = 1.0
    A = 5.37e-8
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1100, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 13, 0, 1.0
    pID[1], pISO[1], pf[1] = 39, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0124(nh, p, ti, dens):
    """
    H3O+ + e- -> O + H2 + H
    """

    #Reaction constants
    br = 1.0
    A = 5.6e-9
    n = -0.5
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1100, 0, 1.0
    sID[1], sISO[1], sf[1] = 1000,  0, 1.0

    npr = 3
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 45, 0, 1.0
    pID[1], pISO[1], pf[1] = 39, 0, 1.0
    pID[2], pISO[2], pf[2] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0125(nh, p, ti, dens):
    """
    O+ + H2 -> OH+ + H
    """

    #Reaction constants
    br = 1.0
    A = 1.7e-9
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1045, 0, 1.0
    sID[1], sISO[1], sf[1] = 39,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1013, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0126(nh, p, ti, dens):
    """
    OH+ + O -> O2+ + H
    """

    #Reaction constants
    br = 1.0
    A = 7.1e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1013, 0, 1.0
    sID[1], sISO[1], sf[1] = 45,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1007, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0127(nh, p, ti, dens):
    """
    OH+ + CO2 -> HCO2+ + O
    """

    #Reaction constants
    br = 1.0
    A = 1.44e-9
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1013, 0, 1.0
    sID[1], sISO[1], sf[1] = 2,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1080, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0128(nh, p, ti, dens):
    """
    OH+ + CO -> HCO+ + O
    """

    #Reaction constants
    br = 1.0
    A = 1.05e-9
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1013, 0, 1.0
    sID[1], sISO[1], sf[1] = 5,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1081, 0, 1.0
    pID[1], pISO[1], pf[1] = 45, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0129(nh, p, ti, dens):
    """
    OH+ + NO -> NO+ + OH
    """

    #Reaction constants
    br = 1.0
    A = 3.59e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1013, 0, 1.0
    sID[1], sISO[1], sf[1] = 8,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1008, 0, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0130(nh, p, ti, dens):
    """
    OH+ + H2 -> H2O+ + H
    """

    #Reaction constants
    br = 1.0
    A = 1.01e-09
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1013, 0, 1.0
    sID[1], sISO[1], sf[1] = 39,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1001, 0, 1.0
    pID[1], pISO[1], pf[1] = 48, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0131(nh, p, ti, dens):
    """
    OH+ + O2 -> O2+ + OH
    """

    #Reaction constants
    br = 1.0
    A = 5.9e-10
    n = 0.0
    gamma = 0.0

    #Calculating reaction rates
    rrates = ion_reaction(br, A, n, gamma, ti)
    
    # Metadata
    rtype = 3
    
    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 1013, 0, 1.0
    sID[1], sISO[1], sf[1] = 7,  0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 1007, 0, 1.0
    pID[1], pISO[1], pf[1] = 13, 0, 1.0

    ref = 'Mars PCM'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref


####################################################################################################

@jit()
def bimolecular(br,A,n,gamma,t):
    """
    Function to calculate the rate of a bimolecular reaction with an Arrhenius expression.
    """
    return br * A * ((t / 298.0)**n) * np.exp(-gamma / t)

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