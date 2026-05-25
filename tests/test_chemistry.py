import pytest

import numpy as np
import os
os.environ["ISOCHEM_USE_JIT"] = "0"   # or "0" to disable
import isochem
curr = os.getcwd()

#########################################################################################################################################################

def test_bimolecular():
    '''
    Test that the reaction rate coefficients for bimolecular reactions are calculated without error
    '''

    #JPL2020 - OH + H2 -> H2O + H
    temp = 298.
    br = 1.0
    A = 2.8e-12
    n = 0.0
    gamma = 1800.0

    rrate = isochem.reactions.bimolecular(br,A,n,gamma,temp)

    result = 6.666575033306925e-15    #Listed in table 1B (HOx) of JPL2020

    assert np.allclose(rrate,result,rtol=1.0e-6)



    #JPL2020 - O(1D) + N2 + M -> N2O + M
    temp = 298.
    press = 101325. #Pa
    dens = press / (isochem.dict.const_dict.phys_const["k_B"] * temp) * 1e-6 #cm-3

    A = 2.8e-36
    n = -0.9
    br = 1.0
    gamma = 0.0

    rrate = isochem.reactions.bimolecular(br, A, n, gamma, temp) * dens

    result = 6.9e-17    #Listed in table 2-1 of JPL2020

    assert np.allclose(rrate,result,rtol=1.0e-6)



    #JPL2020 - O(1D) + N2 -> O + N2
    A = 2.15e-11
    n = 0.0
    gamma = 110.0
    br = 1.0

    rrate = isochem.reactions.bimolecular(br, A, n, gamma, temp)

    result = 3.1e-11    #Listed in table 1A of JPL2020

    assert np.allclose(rrate,result,rtol=1.0e-6)

    #JPL2020 - O(1D) + CO2 -> O + CO2
    A = 7.5e-11
    n = 0.0
    gamma = -115.0
    br = 1.0

    rrate = isochem.reactions.bimolecular(br, A, n, gamma, temp)

    result = 1.1e-10    #Listed in table 1A of JPL2020


    assert np.allclose(rrate,result,rtol=1.0e-6)


#########################################################################################################################################################
    
def test_chemical_activation():

    '''
    Test that the reaction rate coefficients for chemically activated reactions are calculated without error
    '''

    #JPL2020 - OH + CO + M -> CO2 + H + M
    temp = 298.
    press = 101325. #Pa
    dens = press / (isochem.dict.const_dict.phys_const["k_B"] * temp) * 1e-6 #cm-3

    k0 = 6.9e-33
    n = 2.1
    kinf = 1.1e-12
    m = -1.3
    A = 1.85e-13
    B = 65.

    rrate = isochem.reactions.termolecular(k0, n, kinf, m, dens, temp)
    rrate_ca = isochem.reactions.chemical_activation(k0, n, kinf, m, A, B, dens, temp)

    rrate_tot = rrate + rrate_ca

    result = 2.4e-13   #Listed in table 2-2 of JPL2020

    assert np.allclose(rrate_tot,result,rtol=1.0e-6)

#########################################################################################################################################################
    
def test_mass_reactions():
    '''
    Test that the parent and product masses are the same for each reaction
    '''

    #Initialising dummy variables
    reaction_ids = isochem.chemistry.get_reaction_ids()
    gasID = np.array([2,7,22,45],dtype='int32')
    isoID = np.zeros(4,dtype='int32')
    h = np.zeros(3) ; p = np.ones(3) ; t = np.ones(3)
    n = np.ones((3,4))

    #Calculating all available reactions
    rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates = isochem.chemistry.reaction_rate_coefficients(reaction_ids, gasID, isoID, h, p, t, n)
    
    for ir in range(len(reaction_ids)):

        
        #Calculating the mass of the parents
        m_parents = 0.
        for ip in range(ns[ir]):
            m_parents += isochem.dict.gas_dict.get_molwt(sID[ip,ir],sISO[ip,ir]) * sf[ip,ir]
        m_products = 0.
        for ip in range(npr[ir]):
            m_products += isochem.dict.gas_dict.get_molwt(pID[ip,ir],pISO[ip,ir]) * pf[ip,ir]

        print(ir,m_parents,m_products,pf[:,ir])

        assert np.allclose(m_parents,m_products,rtol=1.0e-6)
