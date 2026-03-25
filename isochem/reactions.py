import numpy as np
from isochem.jit import jit
import inspect,re

#The units used in the reactions are as follows:

# Reaction rate coefficients: s-1 if rtype=1; cm3 s-1 if rtype=2

cache = False

###############################################################################################################################

@jit()
def reaction0001(nh, p, t, dens):
    """
    O + O2 + CO2 -> O3 + CO2
    """
    # Calculate reaction rates
    rrates = 2.075 * 6.0e-34 * ((t / 300.0) ** (-2.4)) * dens
    
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
    # NIST expression: 2.5 * 9.46e-34 * exp(485./t) * dens
    rrates = 2.5 * 9.46e-34 * np.exp(485.0 / t) * dens
    
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
    alpha = 8.0e-12
    beta = 0.0
    gamma = 2060.0
    br = 1.0

    rrates = alpha * br * ( (t/300.0) ** beta ) * np.exp(-gamma / t)

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
def reaction0004(nh, p, t, co2):
    """
    O(1D) + CO2 -> O + CO2
    """
    alpha = 7.5e-11
    beta = 0.0
    gamma = -115.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma / t) * co2

    rtype = 1

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 133, 0, 1.0  # O(1D)

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 45, 0, 1.0  # O

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0005(nh, p, t, dens):
    """
    O(1D) + H2O -> OH + OH
    """
    alpha = 1.63e-10
    beta = 0.0
    gamma = -60.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 1.2e-10
    beta = 0.0
    gamma = 0.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
def reaction0007(nh, p, t, o2):
    """
    O(1D) + O2 -> O + O2
    """
    alpha = 3.3e-11
    beta = 0.0
    gamma = -55.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t) * o2

    rtype = 1

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    sID[0], sISO[0], sf[0] = 133, 0, 1.0  # O(1D)

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    pID[0], pISO[0], pf[0] = 45, 0, 1.0   # O

    ref = 'JPL 2020'

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0008(nh, p, t, dens):
    """
    O(1D) + O3 -> O2 + O2  (branching ratio = 0.5)
    """
    alpha = 2.4e-10
    beta = 0.0
    gamma = 0.0
    br = 0.5

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 2.4e-10
    beta = 0.0
    gamma = 0.0
    br = 0.5

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 3.0e-11
    beta = 0.0
    gamma = -200.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 1.8e-11
    beta = 0.0
    gamma = -180.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 1.4e-10
    beta = 0.0
    gamma = 470.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 7.2e-11
    beta = 0.0
    gamma = 0.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 6.9e-12
    beta = 0.0
    gamma = 0.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 1.6e-12
    beta = 0.0
    gamma = 0.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 4.8e-11
    beta = 0.0
    gamma = -250.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 3.0e-13
    beta = 0.0
    gamma = -460.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 1.8e-12
    beta = 0.0
    gamma = 0.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 2.8e-12
    beta = 0.0
    gamma = 1800.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    # k0 = 5.3e-32, n = 1.8, kinf = 9.5e-11, m = -0.4
    # factor 2.4 in front in the code
    k0 = 5.3e-32
    n = 1.8
    kinf = 9.5e-11
    m = -0.4

    rrates = np.zeros(nh, dtype=np.float64)
    for ih in range(nh):
        k0x   = 2.4 * k0 * (298.0 / t[ih])**(n)
        kinfx = kinf * (298.0 / t[ih])**(m)

        tmp = (k0x * dens[ih])
        val = (kinfx * tmp) / (kinfx + tmp)
        # falloff
        c = (1.0 + (np.log10(tmp/kinfx))**2.0)**(-1.0)
        kf = val * 0.6**(c)
        rrates[ih] = kf

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
    alpha = 1.4e-12
    beta = 0.0
    gamma = 2000.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 1.8e-12
    beta = 0.0
    gamma = 0.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 1.7e-12
    beta = 0.0
    gamma = 940.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 1.0e-14
    beta = 0.0
    gamma = 490.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 2.1e-33
    beta = 0.0
    gamma = -920.0
    br = 1.0

    # factor 2.5 in front
    rrates = 2.5 * alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t) * dens

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
    k0 = 6.9e-31
    n  = 1.0
    kinf = 2.6e-11
    m  = 0.0

    rrates = np.zeros(nh, dtype=np.float64)
    for ih in range(nh):
        k0x = 2.5 * k0 * (298.0 / t[ih])**(n)
        kinfx = kinf * (298.0 / t[ih])**(m)
        tmp = k0x * dens[ih]
        val = (kinfx * tmp)/(kinfx + tmp)
        c = (1.0 + (np.log10(tmp/kinfx))**2.0)**(-1.0)
        kf = val * 0.6**(c)
        rrates[ih] = kf

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
    k0 = 3.4e-31
    n = 1.6
    kinf = 2.3e-11
    m_ = 0.2
    A = 5.3e-12
    B = -200.0

    rrates = np.zeros(nh, dtype=np.float64)
    for ih in range(nh):
        k0x   = 2.5 * k0 * (298.0 / t[ih])**(n)
        kinfx = kinf * (298.0 / t[ih])**(m_)
        tmp   = k0x * dens[ih]
        val   = (kinfx * tmp)/(kinfx + tmp)
        c     = (1.0 + (np.log10(tmp/kinfx))**2.0)**(-1.0)
        fall  = val * 0.6**(c)

        kint = A*np.exp(-B/t[ih])
        kca  = kint*(1.0 - fall/kinf)

        rrates[ih] = kca

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
    alpha = 3.0e-12
    beta = 0.0
    gamma = 1500.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 3.44e-12
    beta = 0.0
    gamma = -260.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 2.1e-11
    beta = 0.0
    gamma = -100.0
    br = 1.0

    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma/t)

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
    alpha = 3.3e-12
    beta = 0.0
    gamma = 3150.0
    br = 1.0

    rrates = alpha * br * ((t / 300.0) ** beta) * np.exp(-gamma / t)

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
    alpha = 1.35e-10
    beta = 0.0
    gamma = 0.0
    br = 1.0

    rrates = alpha * br * ((t / 300.0) ** beta) * np.exp(-gamma / t)
    # The Fortran code had a comment about 4.0e-10*exp(-340./t), you could swap if needed.

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
    rrates = 2.8e-17 * (300.0 / t) ** 0.5

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
    # Constant rate
    rrates = np.ones(nh) * 2.19e-11

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
    rrates = 3.8e-11 * np.exp(85.0 / t)

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
def reaction0037(nh, p, t, o):
    """
    N(2D) + O -> N + O
    (Here 'o' is an array representing O-atom concentration vs altitude)
    """
    rrates = 3.3e-12 * np.exp(-260.0 / t) * o

    rtype = 1

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N(2D) (134)
    sID[0], sISO[0], sf[0] = 134, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N (47)
    pID[0], pISO[0], pf[0] = 47, 0, 1.0

    ref = "herron, j. phys. chem. ref. data, 1999"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0038(nh, p, t, n2):
    """
    N(2D) + N2 -> N + N2
    """
    rrates = 1.7e-14 * n2

    rtype = 1

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # N(2D) (134)
    sID[0], sISO[0], sf[0] = 134, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # N (47)
    pID[0], pISO[0], pf[0] = 47, 0, 1.0

    ref = "herron, j. phys. chem. ref. data, 1999"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0039(nh, p, t, dens):
    """
    N(2D) + CO2 -> NO + CO
    """
    # Constant rate
    rrates = np.ones(nh) * 3.6e-13

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
    k0 = 6.9e-33
    n = 2.1
    kinf = 1.1e-12
    m = -1.3
    A = 1.85e-13
    B = 65.0

    rrates = np.zeros(nh, dtype=np.float64)
    for ih in range(nh):
        k0x = 2.5 * k0 * (298.0 / t[ih])**(n)
        kinfx = kinf * (298.0 / t[ih])**(m)
        tmp = k0x * dens[ih]
        val = (kinfx * tmp) / (kinfx + tmp)
        c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
        falloff = val * 0.6**(c)

        # chemical activation
        kint = A * np.exp(-B / t[ih])
        kca = kint * (1.0 - falloff / kinf)

        rrates[ih] = kca

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
    k0 = 6.9e-33
    n = 2.1
    kinf = 1.1e-12
    m = -1.3
    A = 1.85e-13
    B = 65.0

    rrates = np.zeros(nh, dtype=np.float64)
    for ih in range(nh):
        k0x = 2.5 * k0 * (298.0 / t[ih])**(n)
        kinfx = kinf * (298.0 / t[ih])**(m)
        tmp = k0x * dens[ih]
        val = (kinfx * tmp) / (kinfx + tmp)
        c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
        falloff = val * 0.6**(c)

        # chemical activation
        kint = A * np.exp(-B / t[ih])
        kca = kint * (1.0 - falloff / kinf)

        # The Fortran sets rrates = kf for the HOCO channel
        rrates[ih] = falloff

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
    rrates = 2.5 * 6.5e-33 * np.exp(-2184.0 / t) * dens

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
    O(1D) + N2 + CO2 -> N2O + CO2
    (the Fortran calls it O(1D) + N2 + M -> N2O + M)
    """
    # k0 = 2.8e-36, n = 0.9
    k0 = 2.8e-36
    n = 0.9

    rrates = 2.5 * k0 * (t / 300.0)**(-n) * dens

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
    k0 = 9.1e-32
    n = 1.5
    kinf = 3.0e-11
    m = 0.0

    rrates = np.zeros(nh, dtype=np.float64)
    for ih in range(nh):
        k0x = 2.4 * k0 * (298.0 / t[ih])**(n)
        kinfx = kinf * (298.0 / t[ih])**(m)
        tmp = k0x * dens[ih]
        val = (kinfx * tmp) / (kinfx + tmp)
        c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
        kf = val * 0.6**(c)

        rrates[ih] = kf

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
def reaction0045(nh, p, t, n2):
    """
    O(1D) + N2 -> O + N2
    """
    alpha = 2.5e-11
    beta = 0.0
    gamma = -110.0
    br = 1.0

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t) * n2

    rtype = 1

    ns = 1
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O(1D) (133)
    sID[0], sISO[0], sf[0] = 133, 0, 1.0

    npr = 1
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # O (45)
    pID[0], pISO[0], pf[0] = 45, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0046(nh, p, t, dens):
    """
    O(1D) + N2O -> N2 + O2
    """
    alpha = 1.19e-10
    beta = 0.0
    gamma = -20.0
    br = 0.39

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t)

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
    alpha = 1.19e-10
    beta = 0.0
    gamma = -20.0
    br = 0.61

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t)

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
    O + NO2 + M -> NO + O2 + M
    (similar 'chemical activation' logic as reaction0028)
    """
    k0 = 3.4e-31
    n = 1.6
    kinf = 2.3e-11
    m_ = 0.2
    A = 5.3e-12
    B = -200.0

    rrates = np.zeros(nh, dtype=np.float64)
    for ih in range(nh):
        k0x = 2.5 * k0 * (298.0 / t[ih])**(n)
        kinfx = kinf * (298.0 / t[ih])**(m_)
        tmp = k0x * dens[ih]
        val = (kinfx * tmp) / (kinfx + tmp)
        c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
        fall = val * 0.6**(c)

        kint = A * np.exp(-B / t[ih])
        kca = kint * (1.0 - fall / kinf)

        rrates[ih] = kca

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # O (45), NO2 (10)
    sID[0], sISO[0], sf[0] = 45, 0, 1.0
    sID[1], sISO[1], sf[1] = 10, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # NO (8), O2 (7)
    pID[0], pISO[0], pf[0] = 8, 0, 1.0
    pID[1], pISO[1], pf[1] = 7, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0049(nh, p, t, dens):
    """
    O + NO2 + M -> NO3 + M
    """
    k0 = 3.4e-31
    n = 1.6
    kinf = 2.3e-11
    m_ = 0.2
    A = 5.3e-12
    B = -200.0

    rrates = np.zeros(nh, dtype=np.float64)
    for ih in range(nh):
        k0x = 2.5 * k0 * (298.0 / t[ih])**(n)
        kinfx = kinf * (298.0 / t[ih])**(m_)
        tmp = k0x * dens[ih]
        val = (kinfx * tmp) / (kinfx + tmp)
        c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
        fall = val * 0.6**(c)

        kint = A * np.exp(-B / t[ih])
        kca = kint * (1.0 - fall / kinf)

        # For NO3 channel, the code sets rrates(ih) = fall
        rrates[ih] = fall

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
def reaction0050(nh, p, t, dens):
    """
    O + NO3 -> O2 + NO2
    """
    alpha = 1.3e-11
    beta = 0.0
    gamma = 0.0
    br = 1.0

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t)

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
def reaction0051(nh, p, t, dens):
    """
    N + NO2 -> N2O + O
    """
    alpha = 5.8e-12
    beta = 0.0
    gamma = -220.0
    br = 1.0

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t)

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
def reaction0052(nh, p, t, dens):
    """
    NO + NO3 -> NO2 + NO2
    """
    alpha = 1.7e-11
    beta = 0.0
    gamma = -125.0
    br = 1.0

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t)

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
def reaction0053(nh, p, t, dens):
    """
    NO2 + O3 -> NO3 + O2
    """
    alpha = 1.2e-13
    beta = 0.0
    gamma = 2450.0
    br = 1.0

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t)

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
def reaction0054(nh, p, t, dens):
    """
    NO3 + NO3 -> 2NO2 + O2
    """
    alpha = 8.5e-13
    beta = 0.0
    gamma = 2450.0
    br = 1.0

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t)

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
def reaction0055(nh, p, t, dens):
    """
    O2 + HOCO -> HO2 + CO2
    """
    alpha = 2.0e-12
    beta = 0.0
    gamma = 0.0
    br = 1.0

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t)

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
def reaction0056(nh, p, t, dens):
    """
    O + H2 -> OH + H
    """

    rrates = 1.6e-11*(np.exp(-4570/t))

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
def reaction0057(nh, p, t, dens):
    """
    N + O3 -> NO + O2
    """

    rrates = 1.0e-16

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
def reaction0058(nh, p, t, dens):
    """
    N(2D) + NO -> N2 + O
    """
    # Constant rate
    rrates = 6.9e-11

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
def reaction0059(nh, p, t, dens):
    """
    H + NO2 -> OH + NO
    """
    # Constant rate
    alpha = 1.35e-10 # A_Factor
    beta  = 0.0
    gamma = 0.0     # E/R
    br = 1.0        # Braching Ratio 

    rrates = alpha * br * ((t / 300.0)**beta) * np.exp(-gamma / t)

    #rrates = 2.2e-10 * np.exp(-182/t) ------ Yung, Y. L., & DeMore, W. B. (1999).

    rtype = 3

    ns = 2
    sID = np.zeros(2, dtype=np.int32)
    sISO = np.zeros(2, dtype=np.int32)
    sf = np.zeros(2, dtype=np.float64)

    # H (48), NO2 (10)
    sID[0], sISO[0], sf[0] =  48, 0, 1.0
    sID[1], sISO[1], sf[1] =  10, 0, 1.0

    npr = 2
    pID = np.zeros(4, dtype=np.int32)
    pISO = np.zeros(4, dtype=np.int32)
    pf = np.zeros(4, dtype=np.float64)

    # OH (13), NO (8)
    pID[0], pISO[0], pf[0] = 13, 0, 1.0
    pID[1], pISO[1], pf[1] =  8, 0, 1.0

    ref = "JPL 2020"

    return rrates, rtype, ns, sID, sISO, sf, npr, pID, pISO, pf, ref

###############################################################################################################################

@jit()
def reaction0060(nh, p, t, dens):
    """
    H + NO3 -> OH + NO2
    """
    # Constant rate
    rrates = 1.1e-10
    
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
def reaction0061(nh, p, t, dens):
    """
    OH + NO + M -> HONO + M 
    """
    # Constant rate
    k0 = 7.1e-31
    n =  2.6
    kinf = 3.6e-11
    m = 0.1
    rrates = np.zeros(nh, dtype=np.float64)

    #Pag 430; JPL2020
    for ih in range(nh):  
        k0x = k0 * ((298.0 / t[ih])**(n))
        kinfx = kinf * ((298.0 / t[ih])**(m))
        tmp = k0x * dens[ih]
        val = (kinfx * tmp) / (kinfx + tmp)
        c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
        ktot = val * (0.6**(c))

        rrates[ih] = ktot


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
def reaction0062(nh, p, t, dens):
    """
    OH + NO2 + M-> HNO3 + M
    """
    # Constant rate
    
    #First chanel OH + NO2 -> HONO2
    k01 = 1.8e-30
    n1 =  3.0
    kinf1 = 2.8e-11
    m1 = 0
    
    #Second chanel OH + NO2 -> HOONO
    k02 = 9.3e-32
    n2 =  3.9
    kinf2 = 4.2e-11
    m2 = 0.5 

    rrates = np.zeros(nh, dtype=np.float64)
    #Pag 430; JPL2020
    for ih in range(nh):  
        k0x1 = k01 * ((298.0 / t[ih])**(n1))
        kinfx1 = kinf1 * ((298.0 / t[ih])**(m1))
        tmp1 = k0x1 * dens[ih]
        val1 = (kinfx1 * tmp1) / (kinfx1 + tmp1)
        c1 = (1.0 + (np.log10(tmp1 / kinfx1))**2.0)**(-1.0)
        k_1 = val1 * (0.6**(c1))
        
        k0x2 = k02 * ((298.0 / t[ih])**(n2))
        kinfx2 = kinf2 * ((298.0 / t[ih])**(m2))
        tmp2 = k0x2 * dens[ih]
        val2 = (kinfx2 * tmp2) / (kinfx2 + tmp2)
        c2 = (1.0 + (np.log10(tmp2 / kinfx2))**2.0)**(-1.0)
        k_2 = val2 * (0.6**(c2))

        rrates[ih] = k_1 + k_2

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
def reaction0063(nh, p, t, dens):
    """
    OH + NO3 -> HO2 + NO2
    """
    # Constant rate
    rrates = 2.0e-11

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
def reaction0064(nh, p, t, dens):
    """
    OH + HONO -> H2O + NO2
    """
    # Constant rate
    alpha = 3.0e-12 # A_Factor
    beta  = 0.0
    gamma = -250     # E/R
    br = 1.0        # Braching Ratio 
    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma / t)

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
def reaction0065(nh, p, t, dens):
    """
    OH + HNO3 -> H2O + NO3
    """

    rrates = np.zeros(nh, dtype=np.float64)
    rrates[:] = 7.2e-15 * np.exp(785./t)
    
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
def reaction0066(nh, p, t, dens):
    """
    OH + HO2NO2 -> H2O + NO2 + O2
    """
    # Constant rate
    alpha = 4.5e-13 # A_Factor
    beta  = 0.0
    gamma = -610     # E/R
    br = 1.0        # Braching Ratio 
    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma / t)

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
def reaction0067(nh, p, t, dens):
    """
    HO2 + NO2 + M -> HO2NO2 + M
    """
    # Constant rate
    k0 = 1.9e-31
    n =  3.4
    kinf = 4.0e-12
    m = 0.3
    rrates = np.zeros(nh, dtype=np.float64)

    #Pag 430; JPL2020
    for ih in range(nh):  
        k0x = k0 * ((298.0 / t[ih])**(n))
        kinfx = kinf * ((298.0 / t[ih])**(m))
        tmp = k0x * dens[ih]
        val = (kinfx * tmp) / (kinfx + tmp)
        c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
        ktot = val * (0.6**(c))

        rrates[ih] = ktot

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
def reaction0068(nh, p, t, dens):
    """
    HO2 + NO3 -> O2 + HNO3
    """
    # Constant rate
    br = 0.3 
    rrates = 3.5e-12 * br   #Mellouki et al. (1993) - they determined the branching ratio

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
def reaction0069(nh, p, t, dens):
    """
    HO2 + NO3 -> OH + NO2 + O2
    """
    # Constant rate
    br = 0.7 
    rrates = 3.5e-12 * br   #Mellouki et al. (1993) - they determined the branching ratio

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
def reaction0070(nh, p, t, dens):
    """
    NO2 + O3 -> NO3 + O2
    """
    # Constant rate
    alpha = 1.2e-13 # A_Factor
    beta  = 0.0
    gamma = 2450    # E/R
    br = 1.0        # Braching Ratio 
    rrates = alpha * br * ((t/300.0)**beta) * np.exp(-gamma / t)

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
def reaction0071(nh, p, t, dens):
    """
    NO2 + NO3 + M -> N2O5+ M
    """
    # Constant rate
    k0 = 2.4e-30
    n =  3.0
    kinf = 1.6e-12
    m = -0.1
    rrates = np.zeros(nh, dtype=np.float64)

    #Pag 430; JPL2020
    for ih in range(nh):  
        k0x = k0 * ((298.0 / t[ih])**(n))
        kinfx = kinf * ((298.0 / t[ih])**(m))
        tmp = k0x * dens[ih]
        val = (kinfx * tmp) / (kinfx + tmp)
        c = (1.0 + (np.log10(tmp / kinfx))**2.0)**(-1.0)
        ktot = val * (0.6**(c))

        rrates[ih] = ktot

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
def reaction0072(nh, p, t, dens):
    """
    NO2 + NO3  -> NO + NO2 + O2
    """
    # Constant rate
    rrates = 8.2e-14*np.exp(-1480/t)

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

