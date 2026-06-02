import numpy as np
from isochem.jit import jit
import numba
from isochem import *
import inspect, re
import isochem
from isochem.reactions_database import reaction_network
from isochem.reactions_15n_database import reaction_network_15n
from isochem.reactions_13c_database import reaction_network_13c 

cache = True

################################################################################################################################

def get_reaction_ids():
    """
        FUNCTION NAME : get_reaction_ids()
        
        DESCRIPTION : Get all the IDs that are defined in the reaction database
        
        INPUTS : None

        OPTIONAL INPUTS: None
        
        OUTPUTS : None
            
        CALLING SEQUENCE:
        
            get_reaction_ids()
        
        MODIFICATION HISTORY : Juan Alday (13/04/2025)
        
    """

    #Initialising dummy variables
    reaction_ids = np.arange(1,1000,1)

    reactions_sel = []
    for i in range(len(reaction_ids)):

        #Storing metadata
        ireaction = reaction_ids[i]

        if reaction_network[ireaction]['id'] > 0:
            reactions_sel.append(ireaction)
            
    reactions_sel = np.array(reactions_sel,dtype="int32")
    return reactions_sel

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
    reaction_ids = np.arange(1,1000,1)
    gasID = np.array([2,7,22,45],dtype='int32')
    isoID = np.zeros(4,dtype='int32')
    h = np.zeros(3) ; p = np.ones(3) ; t = np.ones(3)
    n = np.ones((3,4))
    ti = np.ones(3)

    for i in range(len(reaction_ids)):

        #Storing metadata
        ireaction = reaction_ids[i]

        if reaction_network[ireaction]['id'] > 0:

            rtype = reaction_network[ireaction]['rtype']
            ns = reaction_network[ireaction]['nreactants']
            sID = reaction_network[ireaction]['reactant_ids'][0:2]
            sISO = reaction_network[ireaction]['reactant_iso_ids'][0:2]
            sf = reaction_network[ireaction]['reactant_numbers'][0:2]

            npr = reaction_network[ireaction]['nproducts']
            pID = reaction_network[ireaction]['product_ids'][0:4]
            pISO = reaction_network[ireaction]['product_iso_ids'][0:4]
            pf = reaction_network[ireaction]['product_numbers'][0:4]

            for j in range(ns):
        
                #Finding name of first gas
                sname = isochem.dict.gas_dict.id_to_name(sID[j], sISO[j])
                
                if sf[j]>1:
                    sname = str(int(sf[j]))+'*'+sname
                
                if j==0:
                    strx = sname
                    if ns==1:
                        strx = strx+' ---> '
                    else:
                        strx = strx+' + '
                else:
                    strx = strx+sname+' ---> '
                
            for j in range(npr):
                
                pname = isochem.dict.gas_dict.id_to_name(pID[j], pISO[j])
                    
                if pf[j]>1:
                    pname = str(int(pf[j]))+'*'+pname
                
                strx = strx+pname
                if j<npr-1:
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
    h = np.zeros(3) ; p = np.ones(3) ; t = np.ones(3) ; ti = np.ones(3) ; te = np.ones(3)
    n = np.ones((3,4))

    if include_13c:
        print("Including associated 13C reactions in the model...")
    if include_15n:
        print("Including associated 15N reactions in the model...")

    rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates = reaction_rate_coefficients(reaction_ids, gasID, isoID, h, p, t, te, ti, n, include_13c=include_13c, include_15n=include_15n)
    
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

@jit(cache=False)
def reaction_rate_coefficients(reaction_ids, gasID, isoID, h, p, t, te, ti, N, include_13c=False, include_15n=False, isotopic_fractionation=True):
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
            Te(nlay) :: Electron temperature (K)
            Ti(nlay) :: Ion temperature (K)
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

    if include_15n:
        reaction_network_isotope = reaction_network_15n
    if include_13c: 
        reaction_network_isotope = reaction_network_13c

    nreactions = len(reaction_ids)

    nreactions = len(reaction_ids)
    nlay = len(h)
    nh = len(h)
    ngas = len(gasID)
    
    if include_13c:
        if include_15n:
            raise ValueError("Error: model is not set up to include both 13C and 15N isotopes. Please choose one or the other.")
        mreactions = len(reaction_ids) * 5
    elif include_15n:
        if include_13c:
            raise ValueError("Error: model is not set up to include both 13C and 15N isotopes. Please choose one or the other.")
        mreactions = len(reaction_ids) * 5
    else:
        mreactions = len(reaction_ids)
    
    # Calculating the total atmospheric density in cm^-3
    dens = np.sum(N, axis=1) * 1.0e-6  # Convert from m^-3 to cm^-3
    numdens = N * 1.0e-6  # Convert from m^-3 to cm^-3 for each species

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
        
        #Storing metadata
        ireaction = reaction_ids[ir]

        if reaction_network[ireaction]['id'] < 0:
            raise ValueError("error :: reaction id "+(str(reaction_ids[ir]))+" not found ")

        rtype[ir] = reaction_network[ireaction]['rtype']
        ns[ir] = reaction_network[ireaction]['nreactants']
        sID[:,ir] = reaction_network[ireaction]['reactant_ids'][0:2]
        sISO[:,ir] = reaction_network[ireaction]['reactant_iso_ids'][0:2]
        sf[:,ir] = reaction_network[ireaction]['reactant_numbers'][0:2]

        npr[ir] = reaction_network[ireaction]['nproducts']
        pID[:,ir] = reaction_network[ireaction]['product_ids'][0:4]
        pISO[:,ir] = reaction_network[ireaction]['product_iso_ids'][0:4]
        pf[:,ir] = reaction_network[ireaction]['product_numbers'][0:4]

        #Getting the ambient density for this reaction if needed
        ambient_density = np.ones(nlay)  # Default to 1 if no ambient density is needed
        if reaction_network[ireaction]['ambient_id']>=0:
            if reaction_network[ireaction]['ambient_id']==0:
                ambient_density = dens
            else:
                iambient = np.where( (gasID==reaction_network[ireaction]['ambient_id']) )[0]
                ambient_density = np.sum(numdens[:,iambient], axis=1)

        #Checking reaction rate type
        ratetype = reaction_network[ireaction]['ratetype']

        if ratetype == 0:   #Bimolecular

            #Calculating reaction rate
            alpha = reaction_network[ireaction]['alpha']
            n = reaction_network[ireaction]['n']
            gamma = reaction_network[ireaction]['gamma']
            br = reaction_network[ireaction]['branching']

            rrates[:,ir] = isochem.reactions_database.bimolecular(br,alpha,n,gamma,t)

            #Multiplying by ambient density if needed
            rrates[:,ir] *= ambient_density

        elif ratetype == 1:   #Termolecular

            #Calculating reaction rate
            k0 = reaction_network[ireaction]['k0']
            n = reaction_network[ireaction]['n']
            kinf = reaction_network[ireaction]['kinf']
            m = reaction_network[ireaction]['m']

            rrates[:,ir] = isochem.reactions_database.termolecular(k0, n, kinf, m, t, ambient_density)

        elif ratetype == 2:   #Chemical activation

            #Calculating reaction rate
            k0 = reaction_network[ireaction]['k0']
            n = reaction_network[ireaction]['n']
            kinf = reaction_network[ireaction]['kinf']
            m = reaction_network[ireaction]['m']
            A = reaction_network[ireaction]['A']
            B = reaction_network[ireaction]['B']

            rrates[:,ir] = isochem.reactions_database.chemical_activation(k0, n, kinf, m, A, B, t, ambient_density)

        elif ratetype == 3:   #Ion reactions

            #Calculating reaction rate
            alpha = reaction_network[ireaction]['alpha']
            n = reaction_network[ireaction]['n']
            gamma = reaction_network[ireaction]['gamma']
            br = reaction_network[ireaction]['branching']
            
            #If there is an electron in the products use electron temperature
            use_te = False
            for ispec in range(ns[ir]):
                if sID[ispec,ir]==1000:
                    use_te = True

            if use_te:
                rrates[:,ir] = isochem.reactions_database.ion_reaction(br,alpha,n,gamma,te)
            else:
                rrates[:,ir] = isochem.reactions_database.ion_reaction(br,alpha,n,gamma,ti)

            #Multiplying by ambient density if needed
            rrates[:,ir] *= ambient_density

        else:

            raise ValueError("error in reaction_rate_coefficients :: ratetype must be 0,1,2 or 3")

    #Calculating isotopic chemistry
    if include_15n or include_13c:

        ix = nreactions
        nreactions_isotope = 0
        # Adjust reaction rates for 15N isotopologues
        for ir in range(nreactions):
            
            ireaction = reaction_ids[ir]

            #Cheking if reaction exists for N-15
            if reaction_network_isotope[ireaction]["id"] > 0:
                
                for ibranch in range(reaction_network_isotope[ireaction]["nbranch"]):

                    #Storing metadata
                    nreactants = reaction_network_isotope[ireaction]['nreactants'][ibranch]
                    nproducts = reaction_network_isotope[ireaction]['nproducts'][ibranch]

                    rtype[ix] = reaction_network_isotope[ireaction]['rtype'][ibranch]
                    ns[ix] = nreactants
                    sID[:nreactants,ix] = reaction_network_isotope[ireaction]['reactant_ids'][:nreactants, ibranch]
                    sISO[:nreactants,ix] = reaction_network_isotope[ireaction]['reactant_iso_ids'][:nreactants, ibranch]
                    sf[:nreactants,ix] = reaction_network_isotope[ireaction]['reactant_numbers'][:nreactants, ibranch]

                    npr[ix] = nproducts
                    pID[:nproducts,ix] = reaction_network_isotope[ireaction]['product_ids'][:nproducts, ibranch]
                    pISO[:nproducts,ix] = reaction_network_isotope[ireaction]['product_iso_ids'][:nproducts, ibranch]
                    pf[:nproducts,ix] = reaction_network_isotope[ireaction]['product_numbers'][:nproducts, ibranch]

                    fractionation_type = reaction_network_isotope[ireaction]["fractionation_type"]
                    branching_factor = reaction_network_isotope[ireaction]["branching_factor"][ibranch]

                    fractionation_factor = branching_factor
                    if isotopic_fractionation is True:

                        if fractionation_type == 0: #Mass-dependent fractionation

                            #Mass-dependent fractionation
                            for ireactant in range(ns[ix]):
                                if sISO[ireactant,ix] != 0:
                                    fractionation_factor *= (isochem.get_molwt(sID[ireactant,ix], sISO[ireactant,ix]) / isochem.get_molwt(sID[ireactant,ix], 0))**(-0.5*sf[ireactant,ix])

                        elif fractionation_type == 1: #Specified fractionation factor

                            fractionation_factor *= reaction_network_isotope[ireaction]["fractionation_factor"][ibranch]

                        else:

                            raise ValueError("error in reaction_rate_coefficients :: ratetype must be 0,1,2 or 3")

                    rrates[:,ix] = rrates[:,ir] * fractionation_factor

                    nreactions_isotope += 1
                    ix += 1

        nreactions_tot = nreactions + nreactions_isotope

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
