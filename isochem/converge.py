import h5py
import numpy as np
import sys,os
from numba import jit
import isochem

########################################################################################################################

def initialise_run(atm_file,xs_file,sol_file,reaction_ids,planet='Mars'):
    """
        FUNCTION NAME : initialise_run()
        
        DESCRIPTION : Initialise the run with atmospheric, cross section, and solar files
        
        INPUTS :
        
            atm_file :: HDF5 file containing the atmospheric profile
            xs_file :: HDF5 file containing the cross sections
            sol_file :: HDF5 file containing the solar flux
            
        OPTIONAL INPUTS:
        
            planet :: Planet name for which the model is being run (default: 'Mars')
        
        OUTPUTS :
        
        CALLING SEQUENCE:
        
            initialise_run(atm_file,xs_file,sol_file)
        
        MODIFICATION HISTORY : Juan Alday (15/12/2025)
        
    """
    
    ### Reading input file with atmospheric profiles
    ############################################################################################
    
    hlay,Tlay,gasID,isoID,Nlay1,ztime = isochem.utils.read_hdf5(atm_file)
    nlay = Nlay1.shape[0]
    ngas = Nlay1.shape[1]
    nt = len(ztime)

    #Using the last timestep as the initial condition for the model
    Nlay = np.zeros((nlay,ngas),dtype='float64')
    Nlay[:,:] = Nlay1[:,:,nt-1]

    #Calculating VMRs and total density
    N0lay = np.sum(Nlay,axis=1)
    VMRlay = np.transpose(np.transpose(Nlay)/N0lay)

    #Calculating the pressure
    Play = N0lay * isochem.phys_const['k_B'] * Tlay
    
    #Setting up the cross sections
    ############################################################################################
    
    wl,wc,wu,solflux,sID_xs,sISO_xs,xs,nreactions_phot,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr = isochem.photolysis.setup_photolysis(xs_file,sol_file,gasID,isoID,Tlay)
    
    #Initialising the boundary conditions
    ############################################################################################
    
    #Reading the boundary conditions from the dictionary
    typeubc,valueubc = calc_upper_bc(gasID,isoID,planet)
    typelbc,valuelbc = calc_lower_bc(gasID,isoID,planet)
    
    #Initialising some diffusion parameters
    ############################################################################################
    
    #Reading the coefficients for molecular diffusion
    A,s,B = read_moldiff_params(gasID,isoID,planet)
    
    #Initialising the molecular weights array
    mmol = isochem.utils.calc_mmol(gasID,isoID)

    
    
    
########################################################################################################################

@jit(nopython=True)
def converge_rosenbrock(gasID, isoID, hlay, Play, Tlay, Nlay,                                                           #Atmospheric profiles
                        reaction_ids,                                                                                   #Chemical network
                        wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,  #Photolysis
                        mmol,A,s,B,                                                                                     #Diffusion
                        typelbc,valuelbc,typeubc,valueubc,                                                              #Boundary conditions
                        fix_species,                                                                                    #Fixed species
                        dt, dtmin, dtmax,                                                                               #Timestep parameters
                        planet='Mars',
                        ):
    """
        FUNCTION NAME : converge_rosenbrock()
        
        DESCRIPTION : Initialise the run with atmospheric, cross section, and solar files
        
        INPUTS :
            
        OPTIONAL INPUTS:
        
        OUTPUTS :
        
        CALLING SEQUENCE:
        
        MODIFICATION HISTORY : Juan Alday (15/12/2025)
        
    """
    
    nlay = Nlay.shape[0]                   #Number of atmospheric layers
    ngas = Nlay.shape[1]                   #Number of gases in atmosphere
    nreactions_chem = len(reaction_ids)    #Number of reactions in chemical network
    nreactions_phot = xsr.shape[1]         #Number of photolysis reactions
    nreactions = nreactions_phot + nreactions_chem
    
    Nini = np.zeros((nlay,ngas),dtype='float64') # Initial number density profiles (m-3)
    Nini[:,:] = Nlay[:,:]
    
    converged = False
    while converged is False:
    
        Ncurr = np.zeros((nlay,ngas),dtype='float64') # Current number density profiles (m-3)
        Ncurr[:,:] = Nini[:,:]
        
        N0curr = np.sum(Ncurr,axis=1) #Total number density profile (m-3)
        Pcurr = N0curr * isochem.phys_const['k_B'] * Tlay  #Pressure profile (Pa)
    
        #CHEMISTRY
        ############################################################################################
    
        #Calculating the photolysis rates
        rrates_phot = isochem.photolysis.photolysis_rates(hlay,gasID,isoID,Ncurr,wl,wu,wc,sID_xs,sISO_xs,xs,xsr,solflux,
                                                        planet='Mars',zen=zen,tau_aero=0.,radius=3393.,galb=0.3,dist_sun=1.5)
        
        #Calculating the chemical reaction rates
        rtype_chem, ns_chem, sf_chem, sID_chem, sISO_chem, npr_chem, pf_chem, pID_chem, pISO_chem, rrates_chem = isochem.chemistry.reaction_rates(reaction_ids, gasID, isoID, hlay, Pcurr, Tlay, Ncurr)
        
        #Combining photolysis and chemical reaction rates
        rrates = np.zeros((nlay, nreactions), dtype='float64')
        rtype = np.zeros(nreactions, dtype=np.int32)
        ns = np.zeros(nreactions, dtype=np.int32)
        sf = np.zeros((2, nreactions), dtype=np.int32)
        sID = np.zeros((2, nreactions), dtype=np.int32)
        sISO = np.zeros((2, nreactions), dtype=np.int32)
        npr = np.zeros(nreactions, dtype=np.int32)
        pf = np.zeros((4, nreactions), dtype=np.int32)
        pID = np.zeros((4, nreactions), dtype=np.int32)
        pISO = np.zeros((4, nreactions), dtype=np.int32)
        
        #Filling arrays with photolysis information
        rtype[0:nreactions_phot] = 1
        ns[0:nreactions_phot] = 1
        sf[:, 0:nreactions_phot] = 1
        sID[:, 0:nreactions_phot] = sID_phot
        sISO[:, 0:nreactions_phot] = sISO_phot
        npr[0:nreactions_phot] = npr_phot
        pf[:, 0:nreactions_phot] = pf_phot
        pID[:, 0:nreactions_phot] = pID_phot
        pISO[:, 0:nreactions_phot] = pISO_phot
        rrates[:, 0:nreactions_phot] = rrates_phot
        
        #Filling arrays with chemical information
        rtype[nreactions_phot:nreactions] = rtype_chem
        ns[nreactions_phot:nreactions] = ns_chem
        sf[:, nreactions_phot:nreactions] = sf_chem
        sID[:, nreactions_phot:nreactions] = sID_chem
        sISO[:, nreactions_phot:nreactions] = sISO_chem
        npr[nreactions_phot:nreactions] = npr_chem
        pf[:, nreactions_phot:nreactions] = pf_chem
        pID[:, nreactions_phot:nreactions] = pID_chem
        pISO[:, nreactions_phot:nreactions] = pISO_chem
        rrates[:, nreactions_phot:nreactions] = rrates_chem
        
        #Indexing the positions of the source and products in the atmospheric arrays
        sID_pos, pID_pos = isochem.chemistry.locate_gas_reactions(ngas, gasID, isoID, nreactions, ns, sID, sISO, npr, pID, pISO)
        
        #Calculating the chemical Jacobian matrix in each layer
        J_chem = np.zeros((nlay, ngas, ngas), dtype='float64')
        for ilay in range(nlay):
            J_chem[ilay, :, :] = isochem.chemistry.calc_jacobian_chemistry(nlay, ngas, ilay, Nlay, nreactions, rtype, ns, sID_pos, sf, npr, pID_pos, pf, rrates)
            
        #Fixing the species that need to be fixed
        for ilay in range(nlay):
            for igas in range(ngas):
                if fix_species[ilay, igas] == 1:
                    J_chem[ilay, igas, :] = 0.0
               
        #Fixing the species if boundary conditions are of fixed density type
        for igas in range(ngas):
            if typelbc[igas] == 1:
                J_chem[0, igas, :] = 0.0
            if typeubc[igas] == 1:
                J_chem[nlay-1, igas, :] = 0.0
        

        #DIFFUSION
        ############################################################################################
        
        N0 = np.sum(Nlay, axis=1)   #Total number density profile (m-3)
        
        if planet=='Mars':
        
            #Calculating the eddy diffusion coefficient profile
            Keddy = isochem.mars.calc_Keddy(hlay,N0curr,K0)   #m2 s-1
            
            #Calculating the gravity field
            grav = isochem.mars.calc_grav(hlay)   #m s-2
    
        #Calculating the molecular diffusion coefficients
        Dmol = isochem.diffusion.calc_Dmoldiff(N0curr,Tlay,A,s) #Molecular diffusion coefficient (m2 s-1) for each gas and level
        
        #Calculating the mean molecular weight
        mmean = isochem.diffusion.calc_mmean(Ncurr,mmol)   #Mean molecular weight at each level (g mol-1)
    
        scaleH0 = np.zeros(nlay,dtype='float64')  #Mean scale height at each level (m)
        
        scaleH0 = isochem.diffusion.calc_scaleH(Tlay,grav,mmean)  #Mean scale height (m)
        scaleH = np.zeros((nlay,ngas),dtype='float64')  #Scale height for each gas (m)
    
        #Calculating the scale height
        call calc_scaleH_mean(nlay,h,T,mmean,scaleH0)
    
        #Calculating the species-dependent scale height
        call calc_scaleH(nlay,ngas,h,T,mmol,scaleH)
        
        