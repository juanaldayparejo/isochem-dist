import pytest
import isochem
import numpy as np
import os
curr = os.getcwd()

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
