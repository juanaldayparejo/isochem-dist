import numpy as np
from numba import njit,jit
from pchempy import *
from pchempy.Python.reactions import *

def list_available_reactions():
    """
        FUNCTION NAME : list_available_reactions()
        
        DESCRIPTION : Print the available reactions in the chemistry network
        
        INPUTS : None

        OPTIONAL INPUTS: None
        
        OUTPUTS : None
            
        CALLING SEQUENCE:
        
            list_available_reactions()
        
        MODIFICATION HISTORY : Juan Alday (13/04/2025)
        
    """
    
    #Initialising dummy variables
    reaction_ids = np.arange(1, 52)
    gasID = np.array([2,7,22,45],dtype='int32')
    isoID = np.zeros(4,dtype='int32')
    h = np.zeros(3) ; p = np.ones(3) ; t = np.ones(3)
    n = np.ones((3,4),dtype='float64')

    rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates = reaction_rates(reaction_ids, gasID, isoID, h, p, t, n)
    
    for i in range(len(reaction_ids)):

        for j in range(ns[i]):
    
            #Finding name of first gas
            if sISO[j,i]!=0:
                sname = gas_info[str(sID[j,i])]["isotope"][str(sISO[j,i])]["name"]
            else:
                sname = gas_info[str(sID[j,i])]["name"]
            
            if sf[j,i]>1:
                sname = str(int(sf[j,i]))+'*'+sname
            
            if j==0:
                strx = sname
                if ns[i]==1:
                    strx = strx+' ---> '
                else:
                    strx = strx+' + '
            else:
                strx = strx+sname+' ---> '
                
        for j in range(npr[i]):
            
            if pISO[j,i]!=0:
                pname = gas_info[str(pID[j,i])]["isotope"][str(pISO[j,i])]["name"]
            else:
                pname = gas_info[str(pID[j,i])]["name"]
                
            if pf[j,i]>1:
                pname = str(int(pf[j,i]))+'*'+pname
            
            strx = strx+pname
            if j<npr[i]-1:
                strx = strx+' + '
        
        print('Reaction '+str(reaction_ids[i])+':',strx)

###############################################################################################################################

@jit(nopython=True)
def reaction_rates(reaction_ids, gasID, isoID, h, p, t, N):
    """
        FUNCTION NAME : reaction_rates()
        
        DESCRIPTION : Calculate the reaction rates for each reaction included in the chemistry network
        
        INPUTS :
        
            reaction_ids(nreaction) :: Reaction IDs of the reactions included in the chemistry network
            gasID(ngas) :: Gas ID of the gases present in the atmosphere
            isoID(ngas) :: Isotope ID of the gases present in the atmosphere
            h(nlay) :: Altitude of each layer (km)
            P(nlay) :: Pressure of each layer (Pa)
            T(nlay) :: Temperature of each layer (K)
            N(nlay,ngas) :: Number density of each gas in each layer (m-3)

        OPTIONAL INPUTS: None
        
        OUTPUTS :
            
            rtype(nreactions) :: Reaction type for each reaction
                                 1 =     a + hv ---> b + c   or   a + c ---> b + c
                                 2 =     a + a ---> b + c
                                 3 =     a + b ---> c + d
            ns(nreactions) :: Number of source species in each reaction (either 1 or 2)
            sf(2,nreactions) :: Number of molecules for each source species
            sID(2,nreactions) :: Gas ID of each source species
            sISO(2,nreactions) :: Isotope ID of each source species
            npr(nreactions) :: Number of product species in each reaction (up to 4)
            pf(4,nreactions) :: Number of molecules for each product
            pID(4,nreactions) :: Gas ID of each product species
            pISO(4,nreactions) :: Isotope ID of each product species
            rrates(nlay,nreactions) :: Reaction rates for each reaction in each layer
            
        CALLING SEQUENCE:
        
            rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates = reaction_rates(reaction_ids, gasID, isoID, h, P, T, N)
        
        MODIFICATION HISTORY : Juan Alday (13/04/2025)
        
    """
    
    nreactions = len(reaction_ids)
    nlay = len(h)
    nh = len(h)
    ngas = len(gasID)
    
    # Initialise dens, co2, o2, n2, o as numpy arrays of length nlay
    dens = np.zeros(nlay)
    co2 = np.zeros(nlay)
    o2 = np.zeros(nlay)
    n2 = np.zeros(nlay)
    o = np.zeros(nlay)

    # Calculating the total atmospheric density in cm^-3
    for ilay in range(nlay):
        dens[ilay] = 0.0
        for igas in range(ngas):
            dens[ilay] += N[ilay, igas] * 1.0e-6

    # Calculating the number density of certain species (cm^-3)
    for igas in range(ngas):
        if gasID[igas] == 2 and isoID[igas] == 0:
            co2[:] = N[:, igas] * 1.0e-6
        elif gasID[igas] == 7 and isoID[igas] == 0:
            o2[:] = N[:, igas] * 1.0e-6
        elif gasID[igas] == 22 and isoID[igas] == 0:
            n2[:] = N[:, igas] * 1.0e-6
        elif gasID[igas] == 45 and isoID[igas] == 0:
            o[:] = N[:, igas] * 1.0e-6
    
    # Initialize arrays
    rtype = np.zeros(nreactions, dtype=np.int32)
    ns = np.zeros(nreactions, dtype=np.int32)
    sf = np.zeros((2, nreactions), dtype=np.int32)
    sID = np.zeros((2, nreactions), dtype=np.int32)
    sISO = np.zeros((2, nreactions), dtype=np.int32)
    npr = np.zeros(nreactions, dtype=np.int32)
    pf = np.zeros((4, nreactions), dtype=np.int32)
    pID = np.zeros((4, nreactions), dtype=np.int32)
    pISO = np.zeros((4, nreactions), dtype=np.int32)
    rrates = np.zeros((nlay, nreactions), dtype=np.float64)
    
    #Start the reaction rates calculation
    for ir in range(nreactions):
        
        if reaction_ids[ir]==1:
            #O + O2 + CO2 -> O3 + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0001(nh, p, t, co2)

        elif reaction_ids[ir]==2:
            #O + O + CO2 -> O2 + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0002(nh, p, t, co2)

        elif reaction_ids[ir]==3:
            #O + O3 -> O2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0003(nh, p, t, dens)

        elif reaction_ids[ir]==4:
            #O(1D) + CO2 -> O + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0004(nh, p, t, co2)
            
        elif reaction_ids[ir]==5:
            #O(1D) + H2O -> OH + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0005(nh, p, t, dens)
            
        elif reaction_ids[ir]==6:
            #O(1D) + H2 -> OH + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0006(nh, p, t, dens)

        elif reaction_ids[ir]==7:
            #O(1D) + O2 -> O + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0007(nh, p, t, o2)
            
        elif reaction_ids[ir]==8:
            #O(1D) + O3 -> O2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0008(nh, p, t, dens)
            
        elif reaction_ids[ir]==9:
            #O(1D) + O3 -> O2 + O + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0009(nh, p, t, dens)
            
        elif reaction_ids[ir]==10:
            #O + HO2 -> OH + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0010(nh, p, t, dens)
            
        elif reaction_ids[ir]==11:
            #O + OH -> O2 + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0011(nh, p, t, dens)
            
        elif reaction_ids[ir]==12:
            #H + O3 -> OH + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0012(nh, p, t, dens)
            
        elif reaction_ids[ir]==13:
            #H + HO2 -> OH + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0013(nh, p, t, dens)
            
        elif reaction_ids[ir]==14:
            #H + HO2 -> H2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0014(nh, p, t, dens)
            
        elif reaction_ids[ir]==15:
            #H + HO2 -> H2O + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0015(nh, p, t, dens)
            
        elif reaction_ids[ir]==16:
            #OH + HO2 -> H2O + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0016(nh, p, t, dens)
            
        elif reaction_ids[ir]==17:
            #HO2 + HO2 -> H2O2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0017(nh, p, t, dens)
            
        elif reaction_ids[ir]==18:
            #OH + H2O2 -> H2O + HO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0018(nh, p, t, dens)
            
        elif reaction_ids[ir]==19:
            #OH + H2 -> H2O + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0019(nh, p, t, dens)
            
        elif reaction_ids[ir]==20:
            #H + O2 + CO2 -> HO2 + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0020(nh, p, t, co2)
            
        elif reaction_ids[ir]==21:
            #O + H2O2 -> OH + HO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0021(nh, p, t, dens)
            
        elif reaction_ids[ir]==22:
            #OH + OH -> H2O + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0022(nh, p, t, dens)
            
        elif reaction_ids[ir]==23:
            #OH + O3 -> HO2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0023(nh, p, t, dens)
            
        elif reaction_ids[ir]==24:
            #HO2 + O3 -> OH + O2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0024(nh, p, t, dens)
            
        elif reaction_ids[ir]==25:
            #HO2 + HO2 + CO2 -> H2O2 + O2 + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0025(nh, p, t, co2)
            
        elif reaction_ids[ir]==26:
            #OH + OH + CO2 -> H2O2 + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0026(nh, p, t, co2)
            
        elif reaction_ids[ir]==27:
            #H + H + CO2 -> H2 + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0027(nh, p, t, co2)
            
        elif reaction_ids[ir]==28:
            #O + NO2 + M -> NO + O2 + M
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0028(nh, p, t, dens)
            
        elif reaction_ids[ir]==29:
            #NO + O3 -> NO2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0029(nh, p, t, dens)
            
        elif reaction_ids[ir]==30:
            #NO + HO2 -> NO2 + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0030(nh, p, t, dens)
            
        elif reaction_ids[ir]==31:
            #N + NO -> N2 + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0031(nh, p, t, dens)
            
        elif reaction_ids[ir]==32:
            #N + O2 -> NO + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0032(nh, p, t, dens)
            
        elif reaction_ids[ir]==33:
            #NO2 + H -> NO + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0033(nh, p, t, dens)
            
        elif reaction_ids[ir]==34:
            #N + O -> NO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0034(nh, p, t, dens)
            
        elif reaction_ids[ir]==35:
            #N + HO2 -> NO + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0035(nh, p, t, dens)
            
        elif reaction_ids[ir]==36:
            #N + OH -> NO + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0036(nh, p, t, dens)
            
        elif reaction_ids[ir]==37:
            #N(2D) + O -> N + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0037(nh, p, t, o)
            
        elif reaction_ids[ir]==38:
            #N(2D) + N2 -> N + N2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0038(nh, p, t, n2)
            
        elif reaction_ids[ir]==39:
            #N(2D) + CO2 -> NO + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0039(nh, p, t, dens)
            
        elif reaction_ids[ir]==40:
            #OH + CO -> CO2 + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0040(nh, p, t, dens)
            
        elif reaction_ids[ir]==41:
            #OH + CO -> HOCO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0041(nh, p, t, dens)
            
        elif reaction_ids[ir]==42:
            #O + CO + M -> CO2 + M
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0042(nh, p, t, dens)
            
        elif reaction_ids[ir]==43:
            #O(1D) + N2 + CO2 -> N2O + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0043(nh, p, t, dens)
            
        elif reaction_ids[ir]==44:
            #O + NO + CO2 -> NO2 + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0044(nh, p, t, dens)
            
        elif reaction_ids[ir]==45:
            #O(1D) + N2 -> O + N2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0045(nh, p, t, n2)
            
        elif reaction_ids[ir]==46:
            #O(1D) + N2O -> N2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0046(nh, p, t, dens)
            
        elif reaction_ids[ir]==47:
            #O(1D) + N2O -> NO + NO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0047(nh, p, t, dens)
            
        elif reaction_ids[ir]==48:
            #O + NO2 + M -> NO + O2 + M
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0048(nh, p, t, dens)
            
        elif reaction_ids[ir]==49:
            #O + NO2 + M -> NO3 + M
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0049(nh, p, t, dens)
            
        elif reaction_ids[ir]==50:
            #O + NO3 -> O2 + NO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0050(nh, p, t, dens)
            
        elif reaction_ids[ir]==51:
            #N + NO2 -> N2O + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0051(nh, p, t, dens)
            
        else:
            raise ValueError(f"Error: Reaction ID {reaction_ids[ir]} is not recognized.")


    return rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates

#############################################################################################################################

@jit(nopython=True)
def calc_jacobian_chemistry(nlay, ngas, ilay, c, nreactions, rtype, ns, sID_pos, sf, npr, pID_pos, pf, rrates):
    """
    Optimized routine to calculate the values of the chemical Jacobian matrix.

    Parameters:
    -----------
    nlay :: Number of atmospheric layers.
    ngas :: Number of gas species.
    ilay :: Level index at which to calculate the Jacobian matrix.
    c(nlay,ngas) :: Number density of each species.
    nreactions :: Number of reactions.
    rtype(nreactions) :: Reaction types.
    ns(nreactions) :: Number of source species.
    sID_pos(2,nreactions) :: Position indices of source species in the gasID array.
    sf(2,nreactions) :: Number of molecules for each source.
    npr(nreactions) :: Number of product species.
    pID_pos(4,nreactions) :: Position indices of product species in the gasID array.
    pf(4,nreactions) :: Number of molecules for each product.
    rrates(nlay,nreactions) :: Reaction rates (nlay, nreactions).

    Returns:
    --------
    Jmat(ngas,ngas) :: Jacobian matrix of chemical species.
    """

    # Initialize the Jacobian matrix with zeros
    Jmat = np.zeros((ngas, ngas), dtype=np.float64)

    eps = 1e-10

    for ir in range(nreactions):
        
        if rtype[ir] == 1:
            # photodissociations (a + hv -> b + c + d + e)
            # or reactions a + c -> b + c + d + e
            # or reactions a + ice -> b + c + d + e
            ################################################################################
            
            ind_phot_2 = sID_pos[0, ir]
            ind_phot_4 = pID_pos[0, ir] 
            ind_phot_6 = pID_pos[1, ir]
            ind_phot_8 = pID_pos[2, ir]
            ind_phot_10 = pID_pos[3, ir]

            Jmat[ind_phot_2, ind_phot_2] -= sf[0, ir] * rrates[ilay, ir]

            if npr[ir] >= 1:
                Jmat[ind_phot_4, ind_phot_2] += pf[0, ir] * rrates[ilay, ir]
            if npr[ir] >= 2:
                Jmat[ind_phot_6, ind_phot_2] += pf[1, ir] * rrates[ilay, ir]
            if npr[ir] >= 3:
                Jmat[ind_phot_8, ind_phot_2] += pf[2, ir] * rrates[ilay, ir]
            if npr[ir] >= 4:
                Jmat[ind_phot_10, ind_phot_2] += pf[3, ir] * rrates[ilay, ir]


        elif rtype[ir] == 2:
            # Reactions a + a -> b + c + d + e
            ################################################################################
            
            ind_3_2 = sID_pos[0, ir]
            ind_3_4 = pID_pos[0, ir]
            ind_3_6 = pID_pos[1, ir]
            ind_3_8 = pID_pos[2, ir]
            ind_3_10 = pID_pos[3, ir]

            Jmat[ind_3_2, ind_3_2] -= sf[0, ir] * rrates[ilay, ir] * c[ilay, ind_3_2]

            if npr[ir] >= 1:
                Jmat[ind_3_4, ind_3_2] += pf[0, ir] * rrates[ilay, ir] * c[ilay, ind_3_2]
            if npr[ir] >= 2:
                Jmat[ind_3_6, ind_3_2] += pf[1, ir] * rrates[ilay, ir] * c[ilay, ind_3_2]
            if npr[ir] >= 3:
                Jmat[ind_3_8, ind_3_2] += pf[2, ir] * rrates[ilay, ir] * c[ilay, ind_3_2]
            if npr[ir] >= 4:
                Jmat[ind_3_10, ind_3_2] += pf[3, ir] * rrates[ilay, ir] * c[ilay, ind_3_2]

        elif rtype[ir] == 3:
            # Reactions a + b -> c + d + e + f
            ################################################################################
            
            ind_4_2 = sID_pos[0, ir]
            ind_4_4 = sID_pos[1, ir]
            ind_4_6 = pID_pos[0, ir]
            ind_4_8 = pID_pos[1, ir]
            ind_4_10 = pID_pos[2, ir]
            ind_4_12 = pID_pos[3, ir]

            eps_4 = abs(c[ilay, ind_4_2]) / (abs(c[ilay, ind_4_2]) + abs(c[ilay, ind_4_4]) + eps)
            eps_4 = min(eps_4, 1.0)

            Jmat[ind_4_2, ind_4_2] -= sf[0, ir] * rrates[ilay, ir] * (1.0 - eps_4) * c[ilay, ind_4_4]
            Jmat[ind_4_2, ind_4_4] -= sf[0, ir] * rrates[ilay, ir] * eps_4 * c[ilay, ind_4_2]
            Jmat[ind_4_4, ind_4_2] -= sf[1, ir] * rrates[ilay, ir] * (1.0 - eps_4) * c[ilay, ind_4_4]
            Jmat[ind_4_4, ind_4_4] -= sf[1, ir] * rrates[ilay, ir] * eps_4 * c[ilay, ind_4_2]

            if npr[ir] >= 1:
                Jmat[ind_4_6, ind_4_2] += pf[0, ir] * rrates[ilay, ir] * (1.0 - eps_4) * c[ilay, ind_4_4]
                Jmat[ind_4_6, ind_4_4] += pf[0, ir] * rrates[ilay, ir] * eps_4 * c[ilay, ind_4_2]
            if npr[ir] >= 2:
                Jmat[ind_4_8, ind_4_2] += pf[1, ir] * rrates[ilay, ir] * (1.0 - eps_4) * c[ilay, ind_4_4]
                Jmat[ind_4_8, ind_4_4] += pf[1, ir] * rrates[ilay, ir] * eps_4 * c[ilay, ind_4_2]
            if npr[ir] >= 3:
                Jmat[ind_4_10, ind_4_2] += pf[2, ir] * rrates[ilay, ir] * (1.0 - eps_4) * c[ilay, ind_4_4]
                Jmat[ind_4_10, ind_4_4] += pf[2, ir] * rrates[ilay, ir] * eps_4 * c[ilay, ind_4_2]
            if npr[ir] >= 4:
                Jmat[ind_4_12, ind_4_2] += pf[3, ir] * rrates[ilay, ir] * (1.0 - eps_4) * c[ilay, ind_4_4]
                Jmat[ind_4_12, ind_4_4] += pf[3, ir] * rrates[ilay, ir] * eps_4 * c[ilay, ind_4_2]

        else:
            raise ValueError(f"Error: Reaction type must be 1, 2, or 3. Reaction {ir}, type {rtype[ir]}")

    return Jmat

#############################################################################################################################

@jit(nopython=True)
def locate_gas_reactions(ngas, gasID, isoID, nreactions, ns, sID, sISO, npr, pID, pISO):
    """
    Routine to find the location of the sources/products in each reaction
    in the Gas ID array defining the gases in the atmosphere.

    Inputs:
    -------
    ngas :: Number of gas species in the atmosphere.
    gasID(ngas) :: Array of gas IDs present in the atmosphere.
    isoID(ngas) :: Array of isotope IDs corresponding to the gases.
    nreactions :: Number of reactions.
    ns(nreactions) :: Number of sources in each reaction.
    sID(2,nreactions) :: Array of source gas IDs in each reaction.
    sISO(2,nreactions) :: Array of source isotope IDs in each reaction.
    npr(nreactions) :: Number of products in each reaction.
    pID(4,nreactions) :: Array of product gas IDs in each reaction.
    pISO(4,nreactions) :: Array of product isotope IDs in each reaction.

    Outputs:
    --------
    sID_pos(2,nreactions) :: Array indicating the positions of source gases in the gasID array.
    pID_pos(4,nreactions) :: Array indicating the positions of product gases in the gasID array.
    """

    # Initialize output arrays
    sID_pos = np.zeros((2, nreactions), dtype=np.int32)
    pID_pos = np.zeros((4, nreactions), dtype=np.int32)

    # Loop through each reaction
    for ir in range(nreactions):
        # Process source gases
        for j in range(ns[ir]):
            igasx = 0
            for igas in range(ngas):
                if sID[j, ir] == gasID[igas] and sISO[j, ir] == isoID[igas]:
                    sID_pos[j, ir] = igas
                    igasx = 1
                    break
            if igasx == 0:
                raise ValueError(f"Error: Reaction {ir+1}/{nreactions} involves a gas not present in the atmosphere (source). "
                                 f"GasID: {sID[j, ir]}, IsoID: {sISO[j, ir]}")

        # Process product gases
        for j in range(npr[ir]):
            igasx = 0
            for igas in range(ngas):
                if pID[j, ir] == gasID[igas] and pISO[j, ir] == isoID[igas]:
                    pID_pos[j, ir] = igas
                    igasx = 1
                    break
            if igasx == 0:
                raise ValueError(f"Error: Reaction {ir+1}/{nreactions} involves a gas not present in the atmosphere (product). "
                                 f"GasID: {pID[j, ir]}, IsoID: {pISO[j, ir]}")

    return sID_pos, pID_pos

############################################################################################################################

def list_reactions(ns, sID, sISO, sf, npr, pID, pISO, pf):
    """
        FUNCTION NAME : list_reactions()
        
        DESCRIPTION : List the reactions included in the chemistry network
        
        INPUTS :
        
            ns(nreactions)          :: Number of sources in each reaction
            sID(2,nreactions)       :: Array of source gas IDs in each reaction
            sISO(2,nreactions)      :: Array of source isotope IDs in each reaction
            sf(2,nreactions)        :: Array of number of molecules for each source in each reaction
            npr(nreactions)         :: Number of products in each reaction
            pID(4,nreactions)       :: Array of product gas IDs in each reaction
            pISO(4,nreactions)      :: Array of product isotope IDs in each reaction
            pf(4,nreactions)        :: Array of number of molecules for each product in each reaction

        OPTIONAL INPUTS: None
        
        OUTPUTS : None
            
        CALLING SEQUENCE:
        
            list_reactions = list_reactions(ns, sID, sISO, sf, npr, pID, pISO, pf)
        
        MODIFICATION HISTORY : Juan Alday (13/12/2025)
        
    """

    nreactions = ns.shape[0]

    list_reactions = []
    for i in range(nreactions):

        for j in range(ns[i]):
    
            #Finding name of first gas
            if sISO[j,i]!=0:
                sname = gas_info[str(sID[j,i])]["isotope"][str(sISO[j,i])]["name"]
            else:
                sname = gas_info[str(sID[j,i])]["name"]
            
            if sf[j,i]>1:
                sname = str(int(sf[j,i]))+'*'+sname
            
            if j==0:
                strx = sname
                if ns[i]==1:
                    strx = strx+' ---> '
                else:
                    strx = strx+' + '
            else:
                strx = strx+sname+' ---> '
                
        for j in range(npr[i]):
            
            if pISO[j,i]!=0:
                pname = gas_info[str(pID[j,i])]["isotope"][str(pISO[j,i])]["name"]
            else:
                pname = gas_info[str(pID[j,i])]["name"]
                
            if pf[j,i]>1:
                pname = str(int(pf[j,i]))+'*'+pname
            
            strx = strx+pname
            if j<npr[i]-1:
                strx = strx+' + '
        
        list_reactions.append(strx)
        
    return list_reactions

