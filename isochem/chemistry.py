import numpy as np
from isochem.jit import jit
import numba
from isochem import *
from isochem.reactions import *
import inspect, re
import isochem
import isochem.reactions

cache = True

################################################################################################################################

def get_reaction_ids():
    reaction_nums = []
    for name, obj in reactions.__dict__.items():  # inspect all module attributes
        # Check if it is a numba jitted function
        if isinstance(obj, numba.core.registry.CPUDispatcher):
            m = re.match(r"reaction(\d{4})", name)
            if m:
                reaction_nums.append(int(m.group(1)))
    return np.array(sorted(reaction_nums))

################################################################################################################################

def list_available_reactions():
    """
        FUNCTION NAME : list_available_reactions()
        
        DESCRIPTION : Print all the available reactions in the chemistry network
        
        INPUTS : None

        OPTIONAL INPUTS: None
        
        OUTPUTS : None
            
        CALLING SEQUENCE:
        
            list_available_reactions()
        
        MODIFICATION HISTORY : Juan Alday (13/04/2025)
        
    """
    
    #Initialising dummy variables
    reaction_ids = get_reaction_ids()
    gasID = np.array([2,7,22,45],dtype='int32')
    isoID = np.zeros(4,dtype='int32')
    h = np.zeros(3) ; p = np.ones(3) ; t = np.ones(3)
    n = np.ones((3,4))

    rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates = reaction_rate_coefficients(reaction_ids, gasID, isoID, h, p, t, n)
    
    for i in range(len(reaction_ids)):

        for j in range(ns[i]):
    
            #Finding name of first gas
            sname = isochem.dict.gas_dict.id_to_name(sID[j,i], sISO[j,i])
            
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
            
            pname = isochem.dict.gas_dict.id_to_name(pID[j,i], pISO[j,i])
                
            if pf[j,i]>1:
                pname = str(int(pf[j,i]))+'*'+pname
            
            strx = strx+pname
            if j<npr[i]-1:
                strx = strx+' + '
        
        print('Reaction '+str(reaction_ids[i])+':',strx)

################################################################################################################################

def list_reactions(reaction_ids,include_13c=False, include_15n=False):
    """
        FUNCTION NAME : list_available_reactions()
        
        DESCRIPTION : Print the available reactions in a specified chemistry network
        
        INPUTS : None

        OPTIONAL INPUTS: None
        
        OUTPUTS : None
            
        CALLING SEQUENCE:
        
            list_available_reactions()
        
        MODIFICATION HISTORY : Juan Alday (13/04/2025)
        
    """
    
    #Initialising dummy variables
    gasID = np.array([2,7,22,45],dtype='int32')
    isoID = np.zeros(4,dtype='int32')
    h = np.zeros(3) ; p = np.ones(3) ; t = np.ones(3)
    n = np.ones((3,4))

    if include_13c:
        print("Including associated 13C reactions in the model...")
    if include_15n:
        print("Including associated 15N reactions in the model...")

    rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates = reaction_rate_coefficients(reaction_ids, gasID, isoID, h, p, t, n, include_13c=include_13c, include_15n=include_15n)
    
    for i in range(len(rtype)):

        for j in range(ns[i]):
    
            #Finding name of first gas
            sname = isochem.dict.gas_dict.id_to_name(sID[j,i], sISO[j,i])

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
            
            pname = isochem.dict.gas_dict.id_to_name(pID[j,i], pISO[j,i])
                
            if pf[j,i]>1:
                pname = str(int(pf[j,i]))+'*'+pname
            
            strx = strx+pname
            if j<npr[i]-1:
                strx = strx+' + '
        
        print("Reaction number " + str(i+1) + ": " + strx)

###############################################################################################################################

@jit()
def reaction_rate_coefficients(reaction_ids, gasID, isoID, h, p, t, N, include_13c=False, include_15n=False, isotopic_fractionation=True):
    """
        FUNCTION NAME : reaction_rate_coefficients()
        
        DESCRIPTION : Calculate the reaction rate coefficients for each reaction included in the chemistry network
        
        INPUTS :
        
            reaction_ids(nreaction) :: Reaction IDs of the reactions included in the chemistry network
            gasID(ngas) :: Gas ID of the gases present in the atmosphere
            isoID(ngas) :: Isotope ID of the gases present in the atmosphere
            h(nlay) :: Altitude of each layer (km)
            P(nlay) :: Pressure of each layer (Pa)
            T(nlay) :: Temperature of each layer (K)
            N(nlay,ngas) :: Number density of each gas in each layer (m-3)

        OPTIONAL INPUTS:

            include_13c :: Whether to include reactions involving 13C isotopes in the model (default: False)
            include_15n :: Whether to include reactions involving 15N isotopes in the model (default: False)
            isotopic_fractionation :: Whether to apply isotopic fractionation factors to the reaction rate coefficients (default: True)
        
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
            rrates(nlay,nreactions) :: Reaction rate coefficients for each reaction in each layer (s-1 if rtype=1 or cm3 s-1 if rtype=2 or 3)
            
        CALLING SEQUENCE:
        
            rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates = reaction_rate_coefficients(reaction_ids, gasID, isoID, h, P, T, N, include_13c, include_15n, isotopic_fractionation)
        
        MODIFICATION HISTORY : Juan Alday (13/04/2025)
        
    """
    
    nreactions = len(reaction_ids)
    nlay = len(h)
    nh = len(h)
    ngas = len(gasID)
    
    if include_13c:
        if include_15n:
            raise ValueError("Error: model is not set up to include both 13C and 15N isotopes. Please choose one or the other.")
        mreactions = len(reaction_ids) * 2
    elif include_15n:
        if include_13c:
            raise ValueError("Error: model is not set up to include both 13C and 15N isotopes. Please choose one or the other.")
        mreactions = len(reaction_ids) * 2
    else:
        mreactions = len(reaction_ids)
    
    # Initialise dens, co2, o2, n2, o as numpy arrays of length nlay
    dens = np.zeros(nlay)
    co2 = np.zeros(nlay)
    o2 = np.zeros(nlay)
    n2 = np.zeros(nlay)
    o = np.zeros(nlay)

    # Calculating the total atmospheric density in cm^-3
    dens = np.sum(N, axis=1) * 1.0e-6  # Convert from m^-3 to cm^-3

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
    rtype = np.zeros(mreactions, dtype=np.int32)
    ns = np.zeros(mreactions, dtype=np.int32)
    sf = np.zeros((2, mreactions), dtype=np.int32)
    sID = np.zeros((2, mreactions), dtype=np.int32)
    sISO = np.zeros((2, mreactions), dtype=np.int32)
    npr = np.zeros(mreactions, dtype=np.int32)
    pf = np.zeros((4, mreactions), dtype=np.int32)
    pID = np.zeros((4, mreactions), dtype=np.int32)
    pISO = np.zeros((4, mreactions), dtype=np.int32)
    rrates = np.zeros((nlay, mreactions), dtype=np.float64)
    
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
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0004(nh, p, t, dens)
            
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
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0037(nh, p, t, dens)
            
        elif reaction_ids[ir]==38:
            #N(2D) + N2 -> N + N2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0038(nh, p, t, dens)
            
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
            #O + NO2 + M -> NO3 + M
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0048(nh, p, t, dens)
            
        elif reaction_ids[ir]==49:
            #O + NO3 -> O2 + NO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0049(nh, p, t, dens)
            
        elif reaction_ids[ir]==50:
            #N + NO2 -> N2O + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0050(nh, p, t, dens)
            
        elif reaction_ids[ir]==51:
            #NO + NO3 -> NO2 + NO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0051(nh, p, t, dens)
            
        elif reaction_ids[ir]==52:
            #NO2 + O3 -> NO3 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0052(nh, p, t, dens)
            
        elif reaction_ids[ir]==53:
            #NO3 + NO3 -> 2NO2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0053(nh, p, t, dens)
            
        elif reaction_ids[ir]==54:
            #O2 + HOCO -> HO2 + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0054(nh, p, t, dens)
            
        elif reaction_ids[ir]==55:
            #O + H2 -> OH + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0055(nh, p, t, dens)

        elif reaction_ids[ir]==56:
            #N + O3 -> NO + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0056(nh, p, t, dens)

        elif reaction_ids[ir]==57:
            #N(2D) + NO -> N2 + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0057(nh, p, t, dens)

        elif reaction_ids[ir]==58:
            #H + NO3 -> OH + NO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0058(nh, p, t, dens)
            
        elif reaction_ids[ir]==59:
            #OH + NO + M -> HONO + M
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0059(nh, p, t, dens)
            
        elif reaction_ids[ir]==60:
             #OH + NO2 + M-> HNO3 + M
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0060(nh, p, t, dens)
            
        elif reaction_ids[ir]==61:
            #OH + NO3 -> HO2 + NO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0061(nh, p, t, dens)
            
        elif reaction_ids[ir]==62:
            #OH + HONO -> H2O + NO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0062(nh, p, t, dens)
            
        elif reaction_ids[ir]==63:
            #OH + HNO3 -> H2O + NO3
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0063(nh, p, t, dens)

        elif reaction_ids[ir]==64:
            #OH + HO2NO2 -> H2O + NO2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0064(nh, p, t, dens)

        elif reaction_ids[ir]==65:
            #HO2 + NO2 + M -> HO2NO2 + M
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0065(nh, p, t, dens)

        elif reaction_ids[ir]==66:
            #HO2 + NO3 -> O2 + HNO3
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0066(nh, p, t, dens)

        elif reaction_ids[ir]==67:
            #HO2 + NO3 -> OH + NO2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0067(nh, p, t, dens)
            
        elif reaction_ids[ir]==68:
            #NO2 + O3 -> NO3 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0068(nh, p, t, dens)
            
        elif reaction_ids[ir]==69:
            #NO2 + NO3 + M -> N2O5 + M
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0069(nh, p, t, dens)
            
        elif reaction_ids[ir]==70:
            #NO2 + NO3  -> NO + NO2 + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0070(nh, p, t, dens)
            
        elif reaction_ids[ir]==71:
            #CO2+ + O2 -> O2+ + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0071(nh, p, t, dens)
            
        elif reaction_ids[ir]==72:
            #CO2+ + O -> O+ + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0072(nh, p, t, dens)
            
        elif reaction_ids[ir]==73:
            #CO2+ + O -> O2+ + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0073(nh, p, t, dens)
            
        elif reaction_ids[ir]==74:
            #O2+ + e- -> O + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0074(nh, p, t, dens)
            
        elif reaction_ids[ir]==75:
            #O+ + CO2 -> O2+ + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0075(nh, p, t, dens)
            
        elif reaction_ids[ir]==76:
            #CO2+ + e- -> CO + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0076(nh, p, t, dens)
            
        elif reaction_ids[ir]==77:
            #CO2+ + NO -> NO+ + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0077(nh, p, t, dens)
            
        elif reaction_ids[ir]==78:
            #O2+ + NO -> NO+ + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0078(nh, p, t, dens)
            
        elif reaction_ids[ir]==79:
            #O2+ + N2 -> NO+ + NO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0079(nh, p, t, dens)
            
        elif reaction_ids[ir]==80:
            #O2+ + N -> NO+ + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0080(nh, p, t, dens)
            
        elif reaction_ids[ir]==81:
            #O+ + N2 -> NO+ + N
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0081(nh, p, t, dens)
            
        elif reaction_ids[ir]==82:
            #NO+ + e- -> N + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0082(nh, p, t, dens)
            
        elif reaction_ids[ir]==83:
            #CO+ + CO2 -> CO2+ + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0083(nh, p, t, dens)
            
        elif reaction_ids[ir]==84:
            #CO+ + O -> O+ + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0084(nh, p, t, dens)
            
        elif reaction_ids[ir]==85:
            #C+ + CO2 -> CO+ + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0085(nh, p, t, dens)
            
        elif reaction_ids[ir]==86:
            #N2+ + CO2 -> CO2+ + N2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0086(nh, p, t, dens)
            
        elif reaction_ids[ir]==87:
            #N2+ + O -> NO+ + N
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0087(nh, p, t, dens)
            
        elif reaction_ids[ir]==88:
            #N2+ + CO -> CO+ + N2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0088(nh, p, t, dens)
            
        elif reaction_ids[ir]==89:
            #N2+ + e– -> N + N
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0089(nh, p, t, dens)
            
        elif reaction_ids[ir]==90:
            #N2+ + O -> O+ + N2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0090(nh, p, t, dens)
            
        elif reaction_ids[ir]==91:
            #N+ + CO2 -> CO2+ + N
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0091(nh, p, t, dens)
            
        elif reaction_ids[ir]==92:
            #CO+ + H -> H+ + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0092(nh, p, t, dens)
            
        elif reaction_ids[ir]==93:
            #O+ + H -> H+ + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0093(nh, p, t, dens)
            
        elif reaction_ids[ir]==94:
            #H+ + O -> O+ + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0094(nh, p, t, dens)
            
        elif reaction_ids[ir]==95:
            #CO2+ + H2 -> HCO2+ + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0095(nh, p, t, dens)
            
        elif reaction_ids[ir]==96:
            #HCO2+ + e– -> H + O + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0096(nh, p, t, dens)
            
        elif reaction_ids[ir]==97:
            #HCO2+ + e- -> OH + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0097(nh, p, t, dens)
            
        elif reaction_ids[ir]==98:
            #HCO2+ + e- -> H + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0098(nh, p, t, dens)
            
        elif reaction_ids[ir]==99:
            #HCO2+ + O -> HCO+ + O2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0099(nh, p, t, dens)
            
        elif reaction_ids[ir]==100:
            #HCO2+ + CO -> HCO+ + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0100(nh, p, t, dens)
            
        elif reaction_ids[ir]==101:
            #H+ + CO2 -> HCO+ + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0101(nh, p, t, dens)
            
        elif reaction_ids[ir]==102:
            #CO2+ + H -> HCO+ + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0102(nh, p, t, dens)
            
        elif reaction_ids[ir]==103:
            #CO+ + H2 -> HCO+ + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0103(nh, p, t, dens)
            
        elif reaction_ids[ir]==104:
            #HCO+ + e- -> CO + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0104(nh, p, t, dens)
            
        elif reaction_ids[ir]==105:
            #CO2+ + H2O -> H2O+ + CO2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0105(nh, p, t, dens)
            
        elif reaction_ids[ir]==106:
            #CO+ + H2O -> H2O+ + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0106(nh, p, t, dens)
            
        elif reaction_ids[ir]==107:
            #O+ + H2O → H2O+ + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0107(nh, p, t, dens)
            
        elif reaction_ids[ir]==108:
            #N2+ + H2O -> H2O+ + N2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0108(nh, p, t, dens)
            
        elif reaction_ids[ir]==109:
            #N+ + H2O -> H2O+ + N
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0109(nh, p, t, dens)
            
        elif reaction_ids[ir]==110:
            #H+ + H2O -> H2O+ + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0110(nh, p, t, dens)
            
        elif reaction_ids[ir]==111:
            #H2O+ + O2 -> O2+ + H2O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0111(nh, p, t, dens)
            
        elif reaction_ids[ir]==112:
            #H2O+ + CO -> HCO+ + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0112(nh, p, t, dens)
            
        elif reaction_ids[ir]==113:
            #H2O+ + O -> O2+ + H2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0113(nh, p, t, dens)
            
        elif reaction_ids[ir]==114:
            #H2O+ + NO -> NO+ + H2O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0114(nh, p, t, dens)
            
        elif reaction_ids[ir]==115:
            #H2O+ + e- -> H + H + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0115(nh, p, t, dens)
            
        elif reaction_ids[ir]==116:
            #H2O+ + e- -> H + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0116(nh, p, t, dens)
            
        elif reaction_ids[ir]==117:
            #H2O+ + e- -> H2 + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0117(nh, p, t, dens)
            
        elif reaction_ids[ir]==118:
            #H2O+ + H2O -> H3O+ + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0118(nh, p, t, dens)
            
        elif reaction_ids[ir]==119:
            #H2O+ + H2 -> H3O+ + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0119(nh, p, t, dens)
            
        elif reaction_ids[ir]==120:
            #HCO+ + H2O -> H3O+ + CO
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0120(nh, p, t, dens)
            
        elif reaction_ids[ir]==121:
            #H3O+ + e- -> OH + H + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0121(nh, p, t, dens)
            
        elif reaction_ids[ir]==122:
            #H3O+ + e- -> H2O + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0122(nh, p, t, dens)
            
        elif reaction_ids[ir]==123:
            #H3O+ + e- -> OH + H2
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0123(nh, p, t, dens)
            
        elif reaction_ids[ir]==124:
            #H3O+ + e- -> O + H2 + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0124(nh, p, t, dens)
            
        elif reaction_ids[ir]==125:
            #O+ + H2 -> OH+ + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0125(nh, p, t, dens)
            
        elif reaction_ids[ir]==126:
            #OH+ + O -> O2+ + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0126(nh, p, t, dens)
            
        elif reaction_ids[ir]==127:
            #OH+ + CO2 -> HCO2+ + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0127(nh, p, t, dens)
            
        elif reaction_ids[ir]==128:
            #OH+ + CO -> HCO+ + O
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0128(nh, p, t, dens)
            
        elif reaction_ids[ir]==129:
            #OH+ + NO -> NO+ + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0129(nh, p, t, dens)
            
        elif reaction_ids[ir]==130:
            #OH+ + H2 -> H2O+ + H
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0130(nh, p, t, dens)
            
        elif reaction_ids[ir]==131:
            #OH+ + O2 -> O2+ + OH
            rrates[:,ir], rtype[ir], ns[ir], sID[:,ir], sISO[:,ir], sf[:,ir], npr[ir], pID[:,ir], pISO[:,ir], pf[:,ir], ref = reaction0131(nh, p, t, dens)
            


        else:
            raise ValueError(f"Error: Reaction ID {reaction_ids[ir]} is not recognized.")


    if include_13c:
        
        if include_15n:
            raise ValueError("Error: model is not set up to include both 13C and 15N isotopes. Please choose one or the other.")

        ix = nreactions
        nreactions_c13 = 0
        # Adjust reaction rates for 13C isotopologues
        for ir in range(nreactions):
            
            if reaction_ids[ir]==39:
                #N(2D) + (13C)O2 -> NO + (13C)O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_13c.reaction0039(nh, p, t, dens)
                nreactions_c13 += 1
                ix += 1
            elif reaction_ids[ir]==40:
                #OH + (13C)O -> (13C)O2 + H
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_13c.reaction0040(nh, p, t, dens)
                nreactions_c13 += 1
                ix += 1
            elif reaction_ids[ir]==41:
                #OH + (13C)O -> HO(13C)O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_13c.reaction0041(nh, p, t, dens)
                nreactions_c13 += 1
                ix += 1
            elif reaction_ids[ir]==42:
                #(13C)O + CO + M -> (13C)O2 + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_13c.reaction0042(nh, p, t, dens)
                nreactions_c13 += 1
                ix += 1
                
        nreactions_tot = nreactions + nreactions_c13
        
    else:
        
        nreactions_tot = nreactions



    if include_15n:

        if include_13c:
            raise ValueError("Error: model is not set up to include both 13C and 15N isotopes. Please choose one or the other.")

        ix = nreactions
        nreactions_n15 = 0
        # Adjust reaction rates for 15N isotopologues
        for ir in range(nreactions):
            
            if reaction_ids[ir]==28:
                #O + (15N)O2 + M -> (15N)O + O2 + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0028(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==29:
                #(15N)O + O3 -> (15N)O2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0029(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==30:
                #(15N)O + HO2 -> (15N)O2 + OH
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0030(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==31:
                #(15N) + NO -> (15N)N + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0031A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #N + (15N)O -> (15N)N + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0031B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==32:
                #(15N) + O2 -> (15N)O + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0032(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==33:
                #(15N)O2 + H -> (15N)O + OH
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0033(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==34:
                #(15N) + O -> (15N)O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0034(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==35:
                #(15N) + HO2 -> (15N)O + OH
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0035(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==36:
                #(15N) + OH -> (15N)O + H
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0036(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==37:
                #(15N)(2D) + O -> (15N) + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0037(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==38:
                #(15N)(2D) + N2 -> (15N) + N2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0038(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==39:
                #(15N)(2D) + CO2 -> (15N)O + CO
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0039(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==43:
                #O(1D) + 15NN + M -> (15N)NO + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0043A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #O(1D) + 15NN + M -> N(15N)O + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0043B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==44:
                #O + (15N)O + CO2 -> (15N)O2 + CO2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0044(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==46:
                #O(1D) + (15N)NO -> 15NN + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0046A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #O(1D) + N(15N)O -> 15NN + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0046B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==47:
                #O(1D) + (15N)NO -> (15N)O + NO
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0047A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #O(1D) + N(15N)O -> (15N)O + NO
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0047B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==48:
                #O + (15N)O2 + M -> (15N)O3 + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0048(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==49:
                #O + (15N)O3 -> O2 + (15N)O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0049(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==50:
                #15N + NO2 -> (15N)NO + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0050A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #15N + NO2 -> N(15N)O + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0050B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #N + (15N)O2 -> (15N)NO + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0050C(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #N + (15N)O2 -> N(15N)O + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0050D(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==51:
                #(15N)O + NO3 -> (15N)O2 + NO2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0051A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #NO + (15N)O3 -> (15N)O2 + NO2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0051B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #(15N)O + (15N)O3 -> (15N)O2 + (15N)O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0051C(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==52:
                #(15N)O2 + O3 -> (15N)O3 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0052(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==53:
                #(15N)O3 + NO3 -> (15N)O2 + NO2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0053A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #(15N)O3 + (15N)O3 -> (15N)O2 + (15N)O2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0053B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==56:
                #(15N) + O3 -> (15N)O + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0056(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==57:
                #(15N)(2D) + NO -> (15N)N + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0057A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #N(2D) + (15N)O -> (15N)N + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0057B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==58:
                #H + (15N)O3 -> OH + (15N)O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0058(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==59:
                #OH + (15N)O + M -> HO(15N)O + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0059(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==60:
                #OH + (15N)O2 + M-> H(15N)O3 + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0060(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==61:
                #OH + (15N)O3 -> HO2 + (15N)O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0061(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==62:
                #OH + HO(15N)O -> H2O + (15N)O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0062(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==63:
                #OH + H(15N)O3 -> H2O + (15N)O3
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0063(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==64:
                #OH + HO2(15N)O2 -> H2O + (15N)O2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0064(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==65:
                #HO2 + (15N)O2 + M -> HO2(15N)O2 + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0065(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==66:
                #HO2 + NO3 -> O2 + HNO3
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0066(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==67:
                #HO2 + (15N)O3 -> OH + (15N)O2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0067(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==68:
                #(15N)O2 + O3 -> (15N)O3 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0068(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

            elif reaction_ids[ir]==69:
                #(15N)O2 + NO3 + M -> (15N)NO5 + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0069A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #NO2 + (15N)O3 + M -> (15N)NO5 + M
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0069B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==70:
                #(15N)O2 + NO3  -> (15N)O + NO2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0070A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #(15N)O2 + NO3  -> NO + (15N)O2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0070B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #NO2 + (15N)O3  -> (15N)O + NO2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0070C(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #NO2 + (15N)O3  -> NO + (15N)O2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0070D(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #(15N)O2 + (15N)O3  -> (15N)O + (15N)O2 + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0070E(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==77:
                #CO2+ + (15N)O -> (15N)O+ + CO2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0077(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==78:
                #O2+ + (15N)O -> (15N)O+ + O2
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0078(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==79:
                #O2+ + (15N)N -> (15N)O+ + NO
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0079A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #O2+ + (15N)N -> NO+ + (15N)O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0079B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==80:
                #O2+ + (15N) -> (15N)O+ + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0080(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==81:
                #O+ + (15N)N -> (15N)O+ + N
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0081A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #O+ + (15N)N -> NO+ + (15N)
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0081B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==82:
                #(15N)O+ + e- -> (15N) + O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0082(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==86:
                #(15N)N+ + CO2 -> CO2+ + (15N)N
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0086(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==87:
                #(15N)N+ + O -> (15N)O+ + N
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0087A(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

                #(15N)N+ + O -> NO+ + (15N)
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0087B(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==88:
                #(15N)N+ + CO -> CO+ + (15N)N
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0088(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==89:
                #(15N)N+ + e– -> (15N) + N
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0089(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==90:
                #(15N)N+ + O -> O+ + (15N)N
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0090(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==91:
                #(15N)+ + CO2 -> CO2+ + (15N)
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0091(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==108:
                #(15N)N+ + H2O -> H2O+ + (15N)N
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0108(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==109:
                #(15N)+ + H2O -> H2O+ + (15N)
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0109(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==114:
                #H2O+ + (15N)O -> (15N)O+ + H2O
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0114(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1
            elif reaction_ids[ir]==129:
                #OH+ + (15N)O -> (15N)O+ + OH
                rrates[:,ix], rtype[ix], ns[ix], sID[:,ix], sISO[:,ix], sf[:,ix], npr[ix], pID[:,ix], pISO[:,ix], pf[:,ix], ref = isochem.reactions_15n.reaction0129(nh, p, t, dens, isotopic_fractionation=isotopic_fractionation)
                nreactions_n15 += 1
                ix += 1

        nreactions_tot = nreactions + nreactions_n15
        
    else:
        
        nreactions_tot = nreactions

        
    # Trim arrays to the actual number of reactions including minor isotopes
    rtype = rtype[:nreactions_tot]
    ns = ns[:nreactions_tot]
    sf = sf[:,:nreactions_tot]
    sID = sID[:,:nreactions_tot]
    sISO = sISO[:,:nreactions_tot]
    npr = npr[:nreactions_tot]
    pf = pf[:,:nreactions_tot]
    pID = pID[:,:nreactions_tot]
    pISO = pISO[:,:nreactions_tot]
    rrates = rrates[:,:nreactions_tot]

    return rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates

#############################################################################################################################

@jit()
def calc_chemistry_system(nlay, ngas, ilay, Nlay, nreactions, rtype, ns, sID_pos, sf, npr, pID_pos, pf, rrates):
    """
    Optimized routine to calculate the values of the chemical Jacobian matrix.

    Parameters:
    -----------
    nlay :: Number of atmospheric layers.
    ngas :: Number of gas species.
    ilay :: Level index at which to calculate the Jacobian matrix.
    Nlay(nlay,ngas) :: Number density of each species (m-3)
    nreactions :: Number of reactions.
    rtype(nreactions) :: Reaction types.
    ns(nreactions) :: Number of source species.
    sID_pos(2,nreactions) :: Position indices of source species in the gasID array.
    sf(2,nreactions) :: Number of molecules for each source.
    npr(nreactions) :: Number of product species.
    pID_pos(4,nreactions) :: Position indices of product species in the gasID array.
    pf(4,nreactions) :: Number of molecules for each product.
    rrates(nlay,nreactions) :: Reaction rate coefficients (nlay, nreactions). (s^-1 for rtype=1, cm^3 s^-1 for rtype=2 and 3)

    Returns:
    --------
    Jmat(ngas,ngas) :: Jacobian matrix of chemical species (s-1).
    prod(ngas) :: Production rate of chemical species (cm-3 s-1)
    loss(ngas) :: Loss rate of chemical species (cm-3 s-1)
    """

    c = Nlay * 1.0e-6  # Convert from m^-3 to cm^-3

    # Initialize the Jacobian matrix with zeros
    Jmat = np.zeros((ngas, ngas), dtype=np.float64)
    prod = np.zeros(ngas, dtype=np.float64)
    loss = np.zeros(ngas, dtype=np.float64)

    eps = 1e-30

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
            loss[ind_phot_2] += sf[0, ir] * rrates[ilay, ir] * c[ilay, ind_phot_2]

            if npr[ir] >= 1:
                Jmat[ind_phot_4, ind_phot_2] += pf[0, ir] * rrates[ilay, ir]
                prod[ind_phot_4] += pf[0, ir] * rrates[ilay, ir] * c[ilay, ind_phot_2]
            if npr[ir] >= 2:
                Jmat[ind_phot_6, ind_phot_2] += pf[1, ir] * rrates[ilay, ir]
                prod[ind_phot_6] += pf[1, ir] * rrates[ilay, ir] * c[ilay, ind_phot_2]
            if npr[ir] >= 3:
                Jmat[ind_phot_8, ind_phot_2] += pf[2, ir] * rrates[ilay, ir]
                prod[ind_phot_8] += pf[2, ir] * rrates[ilay, ir] * c[ilay, ind_phot_2]
            if npr[ir] >= 4:
                Jmat[ind_phot_10, ind_phot_2] += pf[3, ir] * rrates[ilay, ir]
                prod[ind_phot_10] += pf[3, ir] * rrates[ilay, ir] * c[ilay, ind_phot_2]


        elif rtype[ir] == 2:
            # Reactions a + a -> b + c + d + e
            ################################################################################
            
            ind_3_2 = sID_pos[0, ir]
            ind_3_4 = pID_pos[0, ir]
            ind_3_6 = pID_pos[1, ir]
            ind_3_8 = pID_pos[2, ir]
            ind_3_10 = pID_pos[3, ir]

            Jmat[ind_3_2, ind_3_2] -= sf[0, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]
            loss[ind_3_2] += sf[0, ir] * rrates[ilay, ir] * c[ilay, ind_3_2] * c[ilay, ind_3_2]

            if npr[ir] >= 1:
                Jmat[ind_3_4, ind_3_2] += pf[0, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]
                prod[ind_3_4] += pf[0, ir] * rrates[ilay, ir] * c[ilay, ind_3_2] * c[ilay, ind_3_2]
            if npr[ir] >= 2:
                Jmat[ind_3_6, ind_3_2] += pf[1, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]
                prod[ind_3_6] += pf[1, ir] * rrates[ilay, ir] * c[ilay, ind_3_2] * c[ilay, ind_3_2]
            if npr[ir] >= 3:
                Jmat[ind_3_8, ind_3_2] += pf[2, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]
                prod[ind_3_8] += pf[2, ir] * rrates[ilay, ir] * c[ilay, ind_3_2] * c[ilay, ind_3_2]
            if npr[ir] >= 4:
                Jmat[ind_3_10, ind_3_2] += pf[3, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]
                prod[ind_3_10] += pf[3, ir] * rrates[ilay, ir] * c[ilay, ind_3_2] * c[ilay, ind_3_2]


        elif rtype[ir] == 3:
            # Reactions a + b -> c + d + e + f
            ################################################################################
            
            ind_4_2 = sID_pos[0, ir]
            ind_4_4 = sID_pos[1, ir]
            ind_4_6 = pID_pos[0, ir]
            ind_4_8 = pID_pos[1, ir]
            ind_4_10 = pID_pos[2, ir]
            ind_4_12 = pID_pos[3, ir]

            Jmat[ind_4_2, ind_4_2] -= sf[0,ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
            Jmat[ind_4_2, ind_4_4] -= sf[0,ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
            Jmat[ind_4_4, ind_4_2] -= sf[1,ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
            Jmat[ind_4_4, ind_4_4] -= sf[1,ir] * rrates[ilay, ir] * c[ilay, ind_4_2]

            loss[ind_4_2] += sf[0,ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]
            loss[ind_4_4] += sf[1,ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]

            if npr[ir] >= 1:
                Jmat[ind_4_6, ind_4_2] += pf[0, ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
                Jmat[ind_4_6, ind_4_4] += pf[0, ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
                prod[ind_4_6] += pf[0, ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]
            if npr[ir] >= 2:
                Jmat[ind_4_8, ind_4_2] += pf[1, ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
                Jmat[ind_4_8, ind_4_4] += pf[1, ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
                prod[ind_4_8] += pf[1, ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]
            if npr[ir] >= 3:
                Jmat[ind_4_10, ind_4_2] += pf[2, ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
                Jmat[ind_4_10, ind_4_4] += pf[2, ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
                prod[ind_4_10] += pf[2, ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]
            if npr[ir] >= 4:
                Jmat[ind_4_12, ind_4_2] += pf[3, ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
                Jmat[ind_4_12, ind_4_4] += pf[3, ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
                prod[ind_4_12] += pf[3, ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]

        else:
            raise ValueError(f"Error: Reaction type must be 1, 2, or 3. Reaction {ir}, type {rtype[ir]}")

    prod *= 1.0e6  # Convert from cm^-3 to m^-3
    loss *= 1.0e6  # Convert from cm^-3 to m^-3

    return Jmat, prod, loss

#############################################################################################################################


@jit()
def calc_jacobian_chemistry(nlay, ngas, ilay, Nlay, nreactions, rtype, ns, sID_pos, sf, npr, pID_pos, pf, rrates):
    """
    Optimized routine to calculate the values of the chemical Jacobian matrix.

    Parameters:
    -----------
    nlay :: Number of atmospheric layers.
    ngas :: Number of gas species.
    ilay :: Level index at which to calculate the Jacobian matrix.
    Nlay(nlay,ngas) :: Number density of each species (m-3)
    nreactions :: Number of reactions.
    rtype(nreactions) :: Reaction types.
    ns(nreactions) :: Number of source species.
    sID_pos(2,nreactions) :: Position indices of source species in the gasID array.
    sf(2,nreactions) :: Number of molecules for each source.
    npr(nreactions) :: Number of product species.
    pID_pos(4,nreactions) :: Position indices of product species in the gasID array.
    pf(4,nreactions) :: Number of molecules for each product.
    rrates(nlay,nreactions) :: Reaction rate coefficients (nlay, nreactions). (s^-1 for rtype=1, cm^3 s^-1 for rtype=2 and 3)

    Returns:
    --------
    Jmat(ngas,ngas) :: Jacobian matrix of chemical species (s-1).
    """

    c = Nlay * 1.0e-6  # Convert from m^-3 to cm^-3

    # Initialize the Jacobian matrix with zeros
    Jmat = np.zeros((ngas, ngas), dtype=np.float64)

    eps = 1e-30

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

            Jmat[ind_3_2, ind_3_2] -= sf[0, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]

            if npr[ir] >= 1:
                Jmat[ind_3_4, ind_3_2] += pf[0, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]
            if npr[ir] >= 2:
                Jmat[ind_3_6, ind_3_2] += pf[1, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]
            if npr[ir] >= 3:
                Jmat[ind_3_8, ind_3_2] += pf[2, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]
            if npr[ir] >= 4:
                Jmat[ind_3_10, ind_3_2] += pf[3, ir] * rrates[ilay, ir] * 2. * c[ilay, ind_3_2]

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

            Jmat[ind_4_2, ind_4_2] -= sf[0,ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
            Jmat[ind_4_2, ind_4_4] -= sf[0,ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
            Jmat[ind_4_4, ind_4_2] -= sf[1,ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
            Jmat[ind_4_4, ind_4_4] -= sf[1,ir] * rrates[ilay, ir] * c[ilay, ind_4_2]

            loss[ind_4_2] += sf[0,ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]
            loss[ind_4_4] += sf[1,ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]

            if npr[ir] >= 1:
                Jmat[ind_4_6, ind_4_2] += pf[0, ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
                Jmat[ind_4_6, ind_4_4] += pf[0, ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
                prod[ind_4_6] += pf[0, ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]
            if npr[ir] >= 2:
                Jmat[ind_4_8, ind_4_2] += pf[1, ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
                Jmat[ind_4_8, ind_4_4] += pf[1, ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
                prod[ind_4_8] += pf[1, ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]
            if npr[ir] >= 3:
                Jmat[ind_4_10, ind_4_2] += pf[2, ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
                Jmat[ind_4_10, ind_4_4] += pf[2, ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
                prod[ind_4_10] += pf[2, ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]
            if npr[ir] >= 4:
                Jmat[ind_4_12, ind_4_2] += pf[3, ir] * rrates[ilay, ir] * c[ilay, ind_4_4]
                Jmat[ind_4_12, ind_4_4] += pf[3, ir] * rrates[ilay, ir] * c[ilay, ind_4_2]
                prod[ind_4_12] += pf[3, ir] * rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]

        else:
            raise ValueError(f"Error: Reaction type must be 1, 2, or 3. Reaction {ir}, type {rtype[ir]}")

    return Jmat

#############################################################################################################################

@jit()
def calc_prod_loss_chemistry(nlay, ngas, ilay, Nlay, nreactions, rtype, ns, sID_pos, sf, npr, pID_pos, pf, rrates):
    """
    Optimized routine to calculate the production and loss rates for each species

    Parameters:
    -----------
    nlay :: Number of atmospheric layers.
    ngas :: Number of gas species.
    ilay :: Level index at which to calculate the Jacobian matrix.
    Nlay(nlay,ngas) :: Number density of each species (m-3)
    nreactions :: Number of reactions.
    rtype(nreactions) :: Reaction types.
    ns(nreactions) :: Number of source species.
    sID_pos(2,nreactions) :: Position indices of source species in the gasID array.
    sf(2,nreactions) :: Number of molecules for each source.
    npr(nreactions) :: Number of product species.
    pID_pos(4,nreactions) :: Position indices of product species in the gasID array.
    pf(4,nreactions) :: Number of molecules for each product.
    rrates(nlay,nreactions) :: Reaction rate coefficients (nlay, nreactions). (s^-1 for rtype=1, cm^3 s^-1 for rtype=2 and 3)

    Returns:
    --------
    prod(ngas) :: Production rate of each species (m-3 s-1).
    loss(ngas) :: Loss rate of each species (m-3 s-1)
    """

    c = Nlay * 1.0e-6  # Convert from m^-3 to cm^-3

    # Initialize the Jacobian matrix with zeros
    prod = np.zeros(ngas, dtype=np.float64)
    loss = np.zeros(ngas, dtype=np.float64)
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

            term = rrates[ilay, ir] * c[ilay, ind_phot_2]

            loss[ind_phot_2] += sf[0, ir] * term

            if npr[ir] >= 1:
                prod[ind_phot_4] += pf[0, ir] * term
            if npr[ir] >= 2:
                prod[ind_phot_6] += pf[1, ir] * term
            if npr[ir] >= 3:
                prod[ind_phot_8] += pf[2, ir] * term
            if npr[ir] >= 4:
                prod[ind_phot_10] += pf[3, ir] * term


        elif rtype[ir] == 2:
            # Reactions a + a -> b + c + d + e
            ################################################################################
            
            ind_3_2 = sID_pos[0, ir]
            ind_3_4 = pID_pos[0, ir]
            ind_3_6 = pID_pos[1, ir]
            ind_3_8 = pID_pos[2, ir]
            ind_3_10 = pID_pos[3, ir]

            term = rrates[ilay, ir] * c[ilay, ind_3_2] * c[ilay, ind_3_2]

            loss[ind_3_2] += sf[0, ir] * term

            if npr[ir] >= 1:
                prod[ind_3_4] += pf[0, ir] * term
            if npr[ir] >= 2:
                prod[ind_3_6] += pf[1, ir] * term
            if npr[ir] >= 3:
                prod[ind_3_8] += pf[2, ir] * term
            if npr[ir] >= 4:
                prod[ind_3_10] += pf[3, ir] * term

        elif rtype[ir] == 3:
            # Reactions a + b -> c + d + e + f
            ################################################################################
            
            ind_4_2 = sID_pos[0, ir]
            ind_4_4 = sID_pos[1, ir]
            ind_4_6 = pID_pos[0, ir]
            ind_4_8 = pID_pos[1, ir]
            ind_4_10 = pID_pos[2, ir]
            ind_4_12 = pID_pos[3, ir]

            term = rrates[ilay, ir] * c[ilay, ind_4_2] * c[ilay, ind_4_4]

            loss[ind_4_2] += term
            loss[ind_4_4] += term

            if npr[ir] >= 1:
                prod[ind_4_6] += pf[0, ir] * term
            if npr[ir] >= 2:
                prod[ind_4_8] += pf[1, ir] * term
            if npr[ir] >= 3:
                prod[ind_4_10] += pf[2, ir] * term
            if npr[ir] >= 4:
                prod[ind_4_12] += pf[3, ir] * term

        else:
            raise ValueError(f"Error: Reaction type must be 1, 2, or 3. Reaction {ir}, type {rtype[ir]}")

    return prod,loss

#############################################################################################################################

@jit()
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
                                 f"GasID: {sID[j, ir]}, IsoID: {sISO[j, ir]}.")

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
                                 f"GasID: {pID[j, ir]}, IsoID: {pISO[j, ir]}.")

    return sID_pos, pID_pos

############################################################################################################################
