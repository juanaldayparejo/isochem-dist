import h5py
import numpy as np
import sys,os
from isochem.jit import jit
import isochem

########################################################################################################################

def initialise_run(atm_file,xs_file,sol_file,planet='Mars'):
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
    Nlay = np.zeros((nlay,ngas))
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
    typeubc,valueubc = isochem.utils.calc_upper_bc(gasID,isoID,planet)
    typelbc,valuelbc = isochem.utils.calc_lower_bc(gasID,isoID,planet)
    
    #Initialising some diffusion parameters
    ############################################################################################
    
    #Reading the coefficients for molecular diffusion
    A,s,B = isochem.utils.read_moldiff_params(gasID,isoID,planet)
    
    #Initialising the molecular weights array
    mmol = isochem.utils.calc_mmol(gasID,isoID)

    return gasID, isoID, hlay, Play, Tlay, Nlay,\
        wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,\
        mmol,A,s,B,\
        typelbc,valuelbc,typeubc,valueubc
            
########################################################################################################################

@jit(nopython=True)
def calc_jacobian_system(gasID, isoID, hlay, Play, Tlay, Nlay,                                                           #Atmospheric profiles
                         reaction_ids,                                                                                   #Chemical network
                         wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,  #Photolysis
                         mmol,A,s,B,                                                                                     #Diffusion
                         typelbc,valuelbc,typeubc,valueubc,                                                              #Boundary conditions
                         fix_species=None,                                                                               #Fixed species                                                                             #Timestep parameters
                         planet='Mars',zen=60., tau_dust=0., radius=3393., galb=0.3, dist_sun=1.5,K0=1.0e14,Ktype=3,
                         include_chemistry=True,
                         include_diffusion=True,
                         include_13c=False,
                         ):
    """
        FUNCTION NAME : calc_jacobian_system()
        
        DESCRIPTION : Initialise the run with atmospheric, cross section, and solar files
        
        INPUTS :
            
        OPTIONAL INPUTS:
        
        OUTPUTS :
        
        CALLING SEQUENCE:
        
        MODIFICATION HISTORY : Juan Alday (15/12/2025)
        
    """
    
    nlay = Nlay.shape[0]                   #Number of atmospheric layers
    ngas = Nlay.shape[1]                   #Number of gases in atmosphere

    #CHEMISTRY
    ############################################################################################

    if include_chemistry is True:

        #Calculating the photolysis rates
        nreactions_phot = xsr.shape[1]         #Number of photolysis reactions
        rrates_phot = isochem.photolysis.photolysis_rates(hlay,gasID,isoID,Nlay,wl,wu,wc,sID_xs,sISO_xs,xs,xsr,solflux,
                                                        planet=planet,zen=zen,tau_aero=tau_dust,radius=radius,galb=galb,dist_sun=dist_sun)
        
        #Calculating the chemical reaction rates
        rtype_chem, ns_chem, sf_chem, sID_chem, sISO_chem, npr_chem, pf_chem, pID_chem, pISO_chem, rrates_chem = \
            isochem.chemistry.reaction_rate_coefficients(reaction_ids, gasID, isoID, hlay, Play, Tlay, Nlay, include_13c=include_13c)
        nreactions_chem = len(ns_chem)    #Number of chemical reactions
        
        #Combining photolysis and chemical reaction rates
        nreactions = nreactions_phot + nreactions_chem
        rrates = np.zeros((nlay, nreactions))
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
        J_chem = np.zeros((nlay, ngas, ngas))
        for ilay in range(nlay):
            J_chem[ilay, :, :] = isochem.chemistry.calc_jacobian_chemistry(nlay, ngas, ilay, Nlay, nreactions, rtype, ns, sID_pos, sf, npr, pID_pos, pf, rrates)
            
        #Fixing the species that need to be fixed
        for ilay in range(nlay):
            for igas in range(ngas):
                if fix_species is not None:
                    if fix_species[ilay, igas] == 1:
                        J_chem[ilay, igas, :] = 0.0
                
        #Fixing the species if boundary conditions are of fixed density type
        for igas in range(ngas):
            if typelbc[igas] == 1:
                J_chem[0, igas, :] = 0.0
            if typeubc[igas] == 1:
                J_chem[nlay-1, igas, :] = 0.0
                
    else:
        
        J_chem = np.zeros((nlay, ngas, ngas))
    
    #DIFFUSION
    ############################################################################################
    
    if include_diffusion is True:
    
        N0 = np.sum(Nlay, axis=1)   #Total number density profile (m-3)
        
        if planet=='Mars':
        
            #Calculating the eddy diffusion coefficient profile
            Keddy = isochem.mars.calc_Keddy(hlay,N0,K0,Ktype=Ktype)   #m2 s-1
            
            #Calculating the gravity field
            grav = isochem.mars.calc_grav(hlay)   #m s-2

        #Calculating the molecular diffusion coefficients
        Dmol = isochem.diffusion.calc_Dmoldiff(N0,Tlay,A,s) #Molecular diffusion coefficient (m2 s-1) for each gas and level
        
        #Calculating the mean molecular weight
        mmean = isochem.diffusion.calc_mmean(Nlay,mmol)   #Mean molecular weight at each level (g mol-1)

        #Calculating the mean scale height
        scaleH0 = isochem.diffusion.calc_scaleH(Tlay,grav,mmean)  #Mean scale height (m)
        
        #Calculating the species-dependent scale height
        scaleH = np.zeros((nlay,ngas))  #Scale height for each gas (m)
        for igas in range(ngas):
            scaleH[:,igas] = isochem.diffusion.calc_scaleH(Tlay,grav,mmol[igas])  #Scale height for each gas (m)
        
        #Calculating the diffusion coefficients to fill the Jacobian
        ksi,klsi,ksim1,klsim1 = isochem.diffusion.calc_diffusion_coefficients(hlay,Tlay,scaleH0,scaleH,Keddy,Dmol,B)

        #Calculating the Jacobian matrix for diffusion
        J_diff = isochem.diffusion.calc_jacobian_diffusion(ksi,klsi,ksim1,klsim1,typelbc,typeubc,fix_species)

    else:
        
        J_diff = np.zeros((nlay, nlay, ngas))

    return J_diff,J_chem

########################################################################################################################

@jit(nopython=True)
def run_model_rosenbrock(gasID, isoID, hlay, Play, Tlay, Nlay,                                                           #Atmospheric profiles
                         reaction_ids,                                                                                   #Chemical network
                         wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,  #Photolysis
                         mmol,A,s,B,                                                                                     #Diffusion
                         typelbc,valuelbc,typeubc,valueubc,                                                              #Boundary conditions
                         dt,                                                                                             #Timestep parameters
                         fix_species=None, 
                         planet='Mars',
                         max_iter=1000,
                         dtmin=1.0e-6,
                         time=0.0,
                         print_progress=True,
                         include_chemistry=True,
                         include_diffusion=True,
                         include_13c=False):
    """
        FUNCTION NAME : run_rosenbrock()
        
        DESCRIPTION : Run the photochemical model using a second-order Rosenbrock solver
        
        INPUTS :
            
        OPTIONAL INPUTS:
        
        OUTPUTS :
        
        CALLING SEQUENCE:
        
        MODIFICATION HISTORY : Juan Alday (15/12/2025)
        
    """
    
    #Initialising system
    ############################################################################################
    
    nlay = Nlay.shape[0]                   #Number of atmospheric layers
    ngas = Nlay.shape[1]                   #Number of gases in atmosphere
    
    Ncurr = np.zeros((nlay,ngas))
    Ncurr[:,:] = Nlay[:,:]

    k_B = 1.38065e-23               # J K-1 Boltzmann constant
    N0curr = np.sum(Ncurr, axis=1)  #Total number density profile (m-3)
    Pcurr = N0curr * k_B * Tlay     #Pressure profile (Pa)
    
    #Convergence loop
    ############################################################################################
    
    converged = False
    
    itera = 1
    iterc = 1
    
    dtcur = dt
    e = 0.0
    egas = np.zeros(ngas)
    while converged is False:
        
        #Printing values just once every 100 iterations
        if print_progress is True:
            if iterc==1:
                print('Iteration',itera)
                print('dt',dtcur)
                print('e',e)
                for igas in range(ngas):
                    print('gas',gasID[igas],isoID[igas],'err',egas[igas])

        #Calculating the Jacobian matrices for diffusion and chemistry
        J_diff,J_chem = calc_jacobian_system(gasID, isoID, hlay, Pcurr, Tlay, Ncurr,                                         
                            reaction_ids,                                                                                   
                            wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,  
                            mmol,A,s,B,                                                                                   
                            typelbc,valuelbc,typeubc,valueubc,                                                            
                            fix_species,
                            include_chemistry=include_chemistry,
                            include_diffusion=include_diffusion,
                            include_13c=include_13c)
        
    
        #Constructing the block tridiagonal matrix
        JA_tri,JB_tri,JC_tri = construct_blocktridiag(J_diff,J_chem)
        
        #Evaluating the system at n
        FN = eval_fn_system(JA_tri,JB_tri,JC_tri,hlay,Ncurr,typelbc,valuelbc,typeubc,valueubc)

        #Calculating the new matrix on the left hand side of the system
        A_tri, B_tri, C_tri = calc_lhs_rosenbrock_system(JA_tri,JB_tri,JC_tri,dt)

        #Solving for g1
        g1 = blktri(A_tri, B_tri, C_tri, FN)
        
        #Evaluating the system at (n + deltat * g1)
        FN = eval_fn_system(JA_tri,JB_tri,JC_tri,hlay,Ncurr,typelbc,valuelbc,typeubc,valueubc)
        
        #Calculating the rhs for the second step
        FN = FN - 2.0 * g1
        
        #Solving for g2
        g2 = blktri(A_tri, B_tri, C_tri, FN)
        
        #Calculating the new density profiles
        Nnew = np.zeros((nlay,ngas))
        Nnew[:,:] = Ncurr[:,:] + (1.5 * dt * g1[:,:]) + (0.5 * dt * g2[:,:])
        
        
        #Assessing convergence
        err = np.zeros((nlay,ngas))
        err[:,:] = np.abs(Nnew[:,:] - (Ncurr[:,:] + dt * g1[:,:]))
        
        rtol = 1.0e-4 #relative tolerance
        atol = 0.05   #absolute tolerance
        
        for ilay in range(nlay):
            for igas in range(ngas):
                
                lerr = np.abs( (err[ilay,igas]) /
                            (Ncurr[ilay,igas]*rtol+atol) )
                if lerr>e:
                    e = lerr
                if lerr>egas[igas]:
                    egas[igas] = lerr
                
        if e<=0.0:
            e=0.1
            
        #Computing next timestep
        coef = np.maximum(0.1, np.minimum(2.0, 1.0 / np.sqrt(e)))
        dtnew = np.maximum(dtmin, dtcur * coef)
        
        #Updating timestep and profiles
        if e<=1.1:
            
            Ncurr = np.zeros((nlay,ngas))
            Ncurr[:,:] = Nnew[:,:]
            time += dtcur
            dtcur = dtnew
            
        else:

            if dtcur==dtmin:
                Ncurr = np.zeros((nlay,ngas))
                Ncurr[:,:] = Nnew[:,:]
                time += dtcur
            dtcur = dtnew

        if itera == max_iter:
            converged = True
        itera += 1
        iterc += 1
        if iterc==101:
            iterc = 1

    return Nnew, dtnew, time
    
########################################################################################################################

@jit(nopython=True)
def run_model_implicit(gasID, isoID, hlay, Play, Tlay, Nlay,                                                           #Atmospheric profiles
                        reaction_ids,                                                                                   #Chemical network
                        wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,  #Photolysis
                        mmol,A,s,B,                                                                                     #Diffusion
                        typelbc,valuelbc,typeubc,valueubc,                                                              #Boundary conditions
                        dt,                                                                                             #Timestep parameters
                        fix_species=None, 
                        planet='Mars',
                        max_iter=1000,
                        time=0.0,
                        print_progress=True,
                        include_chemistry=True,
                        include_diffusion=True,
                        include_13c=False):
    """
        FUNCTION NAME : run_model_implicit()
        
        DESCRIPTION : Run the photochemical model using an implicit solver
        
        INPUTS :
            
        OPTIONAL INPUTS:
        
        OUTPUTS :
        
        CALLING SEQUENCE:
        
        MODIFICATION HISTORY : Juan Alday (15/12/2025)
        
    """
    
    #Initialising system
    ############################################################################################
    
    nlay = Nlay.shape[0]                   #Number of atmospheric layers
    ngas = Nlay.shape[1]                   #Number of gases in atmosphere
    
    Ncurr = np.zeros((nlay,ngas))
    Ncurr[:,:] = Nlay[:,:]

    k_B = 1.38065e-23               # J K-1 Boltzmann constant
    N0curr = np.sum(Ncurr, axis=1)  #Total number density profile (m-3)
    Pcurr = N0curr * k_B * Tlay     #Pressure profile (Pa)
    
    #Convergence loop
    ############################################################################################
    
    converged = False
    
    itera = 1
    dtcur = dt
    while converged is False:
        
        #Calculating the Jacobian matrices for diffusion and chemistry
        J_diff,J_chem = calc_jacobian_system(gasID, isoID, hlay, Pcurr, Tlay, Ncurr,                                         
                            reaction_ids,                                                                                   
                            wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,  
                            mmol,A,s,B,                                                                                   
                            typelbc,valuelbc,typeubc,valueubc,                                                            
                            fix_species,
                            include_chemistry=include_chemistry,
                            include_diffusion=include_diffusion,
                            include_13c=include_13c)
        
    
        #Constructing the block tridiagonal matrix
        JA_tri,JB_tri,JC_tri = construct_blocktridiag(J_diff,J_chem)
        
        #Evaluating the system at n
        FN = eval_fn_system(JA_tri,JB_tri,JC_tri,hlay,Ncurr,typelbc,valuelbc,typeubc,valueubc)

        #Calculating the new matrix on the left hand side of the system
        A_tri, B_tri, C_tri = calc_lhs_implicit_system(JA_tri,JB_tri,JC_tri,dt)

        #Solving for deltan
        deltan = blktri(A_tri, B_tri, C_tri, FN)
        
        #Calculating the new density profiles
        Nnew = np.zeros((nlay,ngas))
        Nnew[:,:] = Ncurr[:,:] + deltan[:,:]
        
        #Updating profiles
        Ncurr[:,:] = Nnew[:,:]
        N0curr = np.sum(Ncurr, axis=1)  #Total number density profile (m-3)
        Pcurr = N0curr * k_B * Tlay     #Pressure profile (Pa)
        
        #Increasing time and iteration number
        time += dtcur
        if itera == max_iter:
            converged = True
        itera += 1

    return Nnew, time
    
#########################################################################################################################
        
@jit(nopython=True)
def construct_blocktridiag(J_diff,J_chem):
    '''
    Function to construct the block tridiagonal matrix to solve the diffusion + chemistry system

    Inputs
    ------
    
    J_diff(nlay,nlay,ngas) :: Jacobian matrix for the diffusion
    J_chem(nlay,ngas,ngas) :: Jacobian matrix for the chemistry

    Outputs
    --------

    A(nlay,ngas,ngas) :: Subdiagonal of the block tridiagonal matrix (from 2 to NLAY)
    B(nlay,ngas,ngas) :: Diagonal of the block tridiagonal matrix (from 1 to NLAY)
    C(nlay,ngas,ngas) :: Superdiagonal of the block tridiagonal matrix (from 1 to NLAY-1)
    '''
    
    nlay = np.shape(J_diff)[0]
    ngas = np.shape(J_diff)[2]
    
    A_tri = np.zeros((nlay,ngas,ngas))
    B_tri = np.zeros((nlay,ngas,ngas))
    C_tri = np.zeros((nlay,ngas,ngas))
    
    for ilay in range(nlay):
        
        #Diagonal
        B_tri[ilay,:,:] = J_chem[ilay,:,:]
        for igas in range(ngas):
            B_tri[ilay,igas,igas] = B_tri[ilay,igas,igas] + J_diff[ilay,ilay,igas]
    

        #Sub-diagonal
        if ilay>0:
            for igas in range(ngas):
                A_tri[ilay,igas,igas] = A_tri[ilay,igas,igas] + J_diff[ilay,ilay-1,igas]

        #Super-diagonal
        if ilay<nlay-1:
            for igas in range(ngas):
                C_tri[ilay,igas,igas] = C_tri[ilay,igas,igas] + J_diff[ilay,ilay+1,igas]

    return A_tri,B_tri,C_tri

#########################################################################################################################

@jit(nopython=True)
def eval_fn_system(A_tri,B_tri,C_tri,alt,num,typelbc,valuelbc,typeubc,valueubc):
    '''
    Function to evaluate the system of equations at the current step

    Inputs
    ------
    
    A_tri,B_tri,C_tri(nlay,ngas,ngas) :: Subdiagonal, diagonal and superdiagonal of the block tridiagonal matrix
    alt(nh) :: Altitude (m)
    num(nh,ngas) :: Number density of each gas (m-3)
    typelbc(ngas) :: Type of lower boundary condition (following pchempy)
    valuelbc(ngas) :: Value for the lower boundary condition (all in SI units)
    typeubc(ngas) :: Type of upper boundary condition (following pchempy)
    valueubc(ngas) :: Value for the upper boundary condition (all in SI units)
    
    Outputs
    --------

    Fn(nlay,ngas) :: Function evaluated at a given step
    '''
    
    nlay = np.shape(A_tri)[0]
    ngas = np.shape(A_tri)[1]
    
    FN = np.zeros((nlay,ngas))
    X = np.zeros((nlay,ngas))
    
    #Bottom layer
    ###################################################################################

    ilay = 0

    X[ilay,:] = matmul(B_tri[ilay,:,:],num[ilay,:]) + matmul(C_tri[ilay,:,:],num[ilay+1,:])

    for igas in range(ngas):
        
        if typelbc[igas]==1:  #Fixed density
            FN[ilay,igas] = X[ilay,igas] + valuelbc[igas]
        elif typelbc[igas]==2:  #Fixed flux
            FN[ilay,igas] = X[ilay,igas] + (valuelbc[igas])/(alt[1]-alt[0]) 


    #Top layer
    ###################################################################################

    ilay = nlay - 1

    X[ilay,:] = matmul(B_tri[ilay,:,:],num[ilay,:]) + matmul(A_tri[ilay,:,:],num[ilay-1,:])

    for igas in range(ngas):
        
        if typeubc[igas]==1:   #Fixed density
            FN[ilay,igas] = X[ilay,igas] + valueubc[igas]
        elif typeubc[igas]==2: #Fixed flux
            FN[ilay,igas] = X[ilay,igas] - (valueubc[igas])/(alt[1]-alt[0]) 
        elif typeubc[igas]==3: #Fixed velocity
            FN[ilay,igas] = X[ilay,igas] - num[ilay,igas] * (valueubc[igas])/(alt[1]-alt[0])
            
    #In-between layers
    ###################################################################################
    
    for ilay in range(1,nlay-1):
        
        FN[ilay,:] = matmul(A_tri[ilay,:,:],num[ilay-1,:]) + matmul(B_tri[ilay,:,:],num[ilay,:]) + matmul(C_tri[ilay,:,:],num[ilay+1,:])
        
    return FN


#########################################################################################################################

@jit(nopython=True)
def calc_lhs_rosenbrock_system(A_tri,B_tri,C_tri,deltat):
    '''
    Function to calculate the left hand side of the system of equations for a Rosenbrock solver:
    
        (I - gamma*dt*J)*deltaX = F(n)
        
    This function takes the block tridiagonal matrix J and calculates the term (I - gamma*dt*J)

    Inputs
    ------
    
    A_tri,B_tri,C_tri(nlay,ngas,ngas) :: Subdiagonal, diagonal and superdiagonal of the block tridiagonal matrix (Jacobian matrix)
    deltat :: Difference in time between iterations (s)
    
    Outputs
    --------
    A_tri,B_tri,C_tri(nlay,ngas,ngas) :: Subdiagonal, diagonal and superdiagonal of the block tridiagonal matrix (left-hand side of the system)
    
    '''
    
    nlay = np.shape(A_tri)[0]
    ngas = np.shape(A_tri)[1]
    
    A_tri_lhs = np.zeros((nlay,ngas,ngas))
    B_tri_lhs = np.zeros((nlay,ngas,ngas))
    C_tri_lhs = np.zeros((nlay,ngas,ngas))
    
    gamma = 1. + 1./np.sqrt(2.)
    
    #Calculating the matrix (I - gamma*dt*J) 
    for ilay in range(nlay):
        for igas in range(ngas):
            for jgas in range(ngas):
                if igas==jgas:
                    B_tri_lhs[ilay,igas,jgas] = 1.0 - deltat * gamma * B_tri[ilay,igas,jgas]
                else:
                    B_tri_lhs[ilay,igas,jgas] =  - deltat * gamma * B_tri[ilay,igas,jgas]

        A_tri_lhs[ilay,:,:] = - A_tri[ilay,:,:] * deltat * gamma
        C_tri_lhs[ilay,:,:] = - C_tri[ilay,:,:] * deltat * gamma

    return A_tri_lhs,B_tri_lhs,C_tri_lhs

#########################################################################################################################

@jit(nopython=True)
def calc_lhs_implicit_system(A_tri,B_tri,C_tri,deltat):
    '''
    Function to calculate the left hand side of the system of equations assuming a implicit solution:
    
        (I/deltaT - J)*deltaX = F(n)
        
    This function takes the block tridiagonal matrix J and calculates the term (I/deltat - J)

    Inputs
    ------
    
    A_tri,B_tri,C_tri(nlay,ngas,ngas) :: Subdiagonal, diagonal and superdiagonal of the block tridiagonal matrix (Jacobian matrix)
    deltat :: Difference in time between iterations (s)
    
    Outputs
    --------
    A_tri,B_tri,C_tri(nlay,ngas,ngas) :: Subdiagonal, diagonal and superdiagonal of the block tridiagonal matrix (left-hand side of the system)
    
    '''
    
    nlay = np.shape(A_tri)[0]
    ngas = np.shape(A_tri)[1]
    
    A_tri_lhs = np.zeros((nlay,ngas,ngas))
    B_tri_lhs = np.zeros((nlay,ngas,ngas))
    C_tri_lhs = np.zeros((nlay,ngas,ngas))
    
    #Calculating the matrix (I/dt - J) 
    for ilay in range(nlay):
        for igas in range(ngas):
            for jgas in range(ngas):
                if igas==jgas:
                    B_tri_lhs[ilay,igas,jgas] = 1.0/deltat - B_tri[ilay,igas,jgas]
                else:
                    B_tri_lhs[ilay,igas,jgas] = - B_tri[ilay,igas,jgas]

        A_tri_lhs[ilay,:,:] = - A_tri[ilay,:,:]
        C_tri_lhs[ilay,:,:] = - C_tri[ilay,:,:]

    return A_tri_lhs,B_tri_lhs,C_tri_lhs

#########################################################################################################################

@jit(nopython=True)
def matmul(A, B):
    '''
    Perform matrix multiplication
    '''
    
    # Check if B is a vector (1D array)
    if B.ndim == 1:
        # Get the dimensions of A and B
        n, m = A.shape
        p = B.shape[0]
        
        if m != p:
            raise ValueError("Inner dimensions must match for multiplication")

        # Initialize the result vector
        C = np.zeros(n)
        
        # Perform matrix-vector multiplication
        for i in range(n):
            for j in range(m):
                C[i] += A[i, j] * B[j]
                
    # Otherwise, B is a matrix (2D array)
    else:
        # Get the dimensions of the input matrices
        n, m = A.shape
        m2, p = B.shape
        
        if m != m2:
            raise ValueError("Inner dimensions must match for multiplication")
        
        # Initialize the result matrix
        C = np.zeros((n, p))
        
        # Perform matrix-matrix multiplication
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    C[i, j] += A[i, k] * B[k, j]
    
    return C

#########################################################################################################################

@jit(nopython=True)
def blktri(A, B, C, R):
    """
    This function solves a tri-block-diagonal matrix problem.
    Arguments:
    M : int : Size of blocks
    N : int : Number of blocks
    A, B, C : ndarray : Arrays of shape (N, M, M)
    R : ndarray : Array of shape (N, M)
    
    Returns:
    X : ndarray : Solution vector of shape (N, M)
    """
    
    N = A.shape[0]
    M = A.shape[1]
    
    #Copying matrices so that they can be overwritten
    Ac = np.zeros((N,M,M)) ; Bc = np.zeros((N,M,M)) ; Cc = np.zeros((N,M,M)) ; Rc = np.zeros((N,M))
    Ac[:,:,:] = A[:,:,:]
    Bc[:,:,:] = B[:,:,:]
    Cc[:,:,:] = C[:,:,:]
    Rc[:,:] = R[:,:]
    
    # Result vector X
    X = np.zeros((N, M))
    
    # Forward elimination blocks
    for i in range(N):
        MM1 = M - 1

        for j in range(MM1):
            JP1 = j + 1
            T = 1.0 / Bc[i, j, j]
            Bc[i, j, JP1:] *= T
            if i != M:
                Cc[i, j, :] *= T
            Rc[i, j] *= T

            for k in range(JP1, M):
                Bc[i, k, JP1:] -= Bc[i, k, j] * Bc[i, j, JP1:]
                if i != M:
                    Cc[i, k, :] -= Bc[i, k, j] * Cc[i, j, :]
                Rc[i, k] -= Bc[i, k, j] * Rc[i, j]

        # Upper (Gauss-Jordan) elimination in B
        T = 1.0 / Bc[i, M-1, M-1]
        Cc[i, M-1, :] *= T
        Rc[i, M-1] *= T

        for j in range(MM1):
            MP = M - j - 1
            MPM1 = MP - 1

            for k in range(MPM1):
                MR = MP - k - 1
                if i != N-1:
                    Cc[i, MR, :] -= Bc[i, MR, MP] * Cc[i, MP, :]
                Rc[i, MR] -= Bc[i, MR, MP] * Rc[i, MP]

        # B(i) is now the unit matrix, eliminate A(i+1)
        if i != N-1:
            for j in range(M):
                for k in range(M):
                    Bc[i+1, k, :] -= Ac[i+1, k, j] * Cc[i, j, :]
                    Rc[i+1, k] -= Ac[i+1, k, j] * Rc[i, j]

    # Back substitution
    X[N-1, :] = Rc[N-1, :]
    NM1 = N - 1
    for j in range(NM1):
        JB = N - j - 1
        for k in range(M):
            KR = M - k - 1
            S = np.sum(Cc[JB-1, KR, :] * X[JB, :])
            X[JB-1, KR] = Rc[JB-1, KR] - S

    return X

