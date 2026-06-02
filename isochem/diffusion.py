import numpy as np
import matplotlib.pyplot as plt
import sys,os
from isochem.jit import jit
import isochem.dict.diffusion_dict as diffusion_dict
import isochem.dict.gas_dict as gas_dict

cache = True


########################################################################################################################

@jit(cache=cache)
def calc_alpha_therm(gasid):
    '''
    Routine to calculate the thermal diffusion coefficient for each species.

    Inputs
    ------
    
    gasid(ngas) :: Gas ID of each species in each layer

    Outputs
    --------

    alpha_therm(ngas) :: Thermal diffusion coefficient for each gas (dimensionless)
    '''

    alpha_therm = np.zeros(len(gasid))
    for i in range(len(gasid)):
        alpha_therm[i] = diffusion_dict.get_thermal_diffusion_coefficient(gasid[i])

    return alpha_therm

########################################################################################################################

@jit(cache=cache)
def calc_Ddiff(gasid,isoid,numdens,temp,temp_elec,temp_ions,planet="Mars"):
    '''
    Routine to calculate the molecular or ambipolar diffusion coefficients for each species at each level.
    The parameterisations here are taken from Krasnopolsky's book 
    
        D_i = A_i * temp**s_i / numdens   (for molecular diffusion of neutrals)
        D_i = kB * (Te+Ti) / (m_i * np.sum(nu_ij))  (for ambipolar diffusion of ions)

    The diffusion coefficients of H and H2 in CO2 are provided in this reference.
    For the rest of the species, we set A and s to 1.0 and 0.75 (Cangi et al. 2020)

    Inputs
    ------
    
    gasid(ngas) :: Gas ID of each species in each layer
    isoid(ngas) :: Isotope ID of each species in each layer
    numdens(nh,ngas) :: Atmospheric number density (m-3)
    temp(nh) :: Atmospheric temperature (K)
    temp_elec(nh) :: Electron temperature (K)
    temp_ions(nh) :: Ion temperature (K)

    Outputs
    --------

    D(nh,ngas) :: Molecular diffusion coefficient (m2 s-1)
    '''

    ngas = len(gasid)
    nh = len(temp)

    num = np.sum(numdens,axis=1)  #Total number density (m-3)

    polarizabity = np.zeros(ngas)
    thermal_diffusion_coefficient = np.zeros(ngas)
    A_moldiff = np.zeros(ngas)
    s_moldiff = np.zeros(ngas)
    for i in range(ngas):

        gasid_i = gasid[i]

        #Getting the molecular diffusion parameters
        if planet=="Mars":
            A_moldiff[i], s_moldiff[i] = diffusion_dict.get_molecular_diffusion_parameters_co2(gasid_i)
        else:
            raise Exception("Diffusion coefficients for planet "+str(planet)+" are not available in the dictionary")

        #Getting the polarizability
        polarizabity[i] = diffusion_dict.get_polarizability(gasid_i)

    #Calculating the molecaular diffusion coefficient
    D = np.zeros((nh,ngas))
    for i in range(ngas):
        
        gasid_i = gasid[i]
        isoid_i = isoid[i]

        if gasid_i<1000:

            #Molecular diffusion for neutrals
            Ai = A_moldiff[i]
            si = s_moldiff[i]

            D[:,i] = Ai * temp[:]**si / (num[:] * 1.0e-6)  #cm2 s-1
        
        elif gasid_i>1000:

            #Ambipolar diffusion for ions
            k_B = 1.380649e-16   #erg/K (cgs units - g cm2 s-2 K-1)
            N_A = 6.02214e23     #mol-1 (cgs units)

            mi = gas_dict.get_molwt(gasid_i,isoid_i) / N_A  #Mass of the ion in grams

            #Calculating the collision frequency with neutrals
            nu_ij = np.zeros(nh)
            for j in range(ngas):

                gasid_j = gasid[j]
                isoid_j = isoid[j]

                if gasid_j < 1000:  #Only collisions with neutrals are considered

                    polarizability_j = polarizabity[j]  #cm3
                    q = -4.803e-10 #statcoulombs (cgs units - g^(1/2)·cm^(3/2)·s⁻¹)
                    mj = gas_dict.get_molwt(gasid_j,isoid_j) / N_A  #Mass of the neutral in grams
                    reduced_mass = mi * mj / (mi + mj)  #reduced mass in grams

                    nu_ij[:] += 2. * np.pi * numdens[:,j] * 1.0e-6 * ( polarizability_j * q**2. / reduced_mass ) ** 0.5

            D[:,i] = k_B * (temp_elec + temp_ions) / (mi * nu_ij)  #cm2 s-1

    D = D * 1.0e-4 #Changing units to m2 s-1
    
    return D

########################################################################################################################

@jit(cache=cache)
def calc_Dmoldiff(num,temp,A,s):
    '''
    Routine to calculate the molecular diffusion coefficients for each species at each level.
    The main citation for this is Hunten (1973). It is calculated as:
    
        D_i = A_i * temp**s_i / numdens

    The diffusion coefficients of H and H2 in CO2 are provided in this reference.
    For the rest of the species, we set A and s to 1.0 and 0.75 (Cangi et al. 2020)

    Inputs
    ------
    
    num(nh) :: Atmospheric number density (m-3)
    temp(nh) :: Atmospheric temperature (K)
    A(ngas) :: Constant for each gas
    s(ngas) :: Constant for each gas

    Optional inputs
    ----------------

    None
    
    Outputs
    --------

    D(nh) :: Molecular diffusion coefficient (m2 s-1)
    '''
    
    nh = len(num)
    ngas = len(A)
    
    #Calculating the molecaular diffusion coefficient
    D = np.zeros((nh,ngas))
    for i in range(ngas):
        
        Ai = A[i]
        si = s[i]
        
        D[:,i] = Ai * temp[:]**si / (num[:] * 1.0e-6)
        
    D = D * 1.0e-4 #Changing units to m2 s-1
    
    return D


@jit()
def calc_scaleH(temp,grav,mmol):
    '''
    Function to calculate the scale height

    Inputs
    ------
    
    temp(nh) :: Temperature (K)
    grav(nh) :: Gravity acceleration (m s-2)
    mmol(nh) :: Molecular weight (g mol-1)
    

    Outputs
    --------

    scaleH(nh) :: Scale height (m)
    '''
    
    k_B = 1.380649e-23   #m2 kg s-2 K-1
    N_A = 6.02214e23     #mol-1
    
    scaleH = k_B * temp / ( (mmol/N_A/1.0e3) * grav )
    
    return scaleH


@jit(cache=cache)
def calc_mmean(num_gas,mmol):
    '''
    Function to calculate mean molecular weight in each layer

    Inputs
    ------
    
    num_gas(nh,ngas) :: Number density of each gas (m-3)
    mmol(ngas) :: Molecular weight of each gas (g mol-1)
    

    Outputs
    --------

    mmean(nh) :: Mean molecular weight at each level (g mol-1)
    '''
    
    mmean = np.sum(num_gas * mmol,axis=1) / np.sum(num_gas,axis=1)
    
    return mmean


@jit(cache=cache)
def calc_diffusion_coefficients(h,temp,temp_elec,temp_ions,
        scaleH0,scaleH,
        K,D,alpha,
        moltype,
        typelbc,valuelbc,typeubc,valueubc):
    '''
    Function to calculate the gravity field

    Inputs
    ------
    
    h(nh) :: Altitude (m)
    temp(nh) :: Temperature (K)
    temp_elec(nh) :: Electron temperature (K)
    temp_ion(nh) :: Ion temperature (K)
    scaleH0(nh) :: Mean scale height (m)
    scaleH(nh,ngas) :: Scale height for each gas (m)
    K(nh) :: Eddy diffusion coefficient (m2 s-1)
    D(nh,ngas) :: Molecular diffusion coefficient (m2 s-1)
    alpha(ngas) :: Molecular thermal diffusion coefficient of each gas
    moltype(ngas) :: Molecular type (0 - neutral; 1 - electron; 2 - ion)
    typelbc(ngas) :: Type of lower boundary condition (following isochem)
    valuelbc(ngas) :: Value for the lower boundary condition
    typeubc(ngas) :: Type of upper boundary condition (following isochem)
    valueubc(ngas) :: Value for the upper boundary condition
    
    Outputs
    --------

    Adiff(nh,ngas), Bdiff(nh,ngas), Cdiff(nh,ngas), Ddiff(nh,ngas) :: Coefficients in each layer to calculate the Jacobian (s-1)
    '''
    
    delz = h[1] - h[0]   #Width of each layer, assumed to be constant (m)
    
    nh = np.shape(scaleH)[0] 
    ngas = np.shape(scaleH)[1]
    
    Adiff = np.zeros((nh,ngas))
    Bdiff = np.zeros((nh,ngas))
    Cdiff = np.zeros((nh,ngas))
    Ddiff = np.zeros((nh,ngas))

    iions = np.where(moltype==2)[0]
    ineutrals = np.where(moltype==0)[0]
     
    for ih in range(len(h)):

        #Lower boundary
        if ih==0:

            D_jph = (D[ih,:]+D[ih+1,:])/2.  #D(i,j+1/2)
            K_jph = np.zeros_like(D_jph)
            K_jph[ineutrals] = (K[ih]+K[ih+1])/2.    #K(j+1/2) for neutrals, no eddy diffusion for ions

            H0_jph = (scaleH0[ih]+scaleH0[ih+1])/2.  #H0(j+1/2)
            H_jph = (scaleH[ih,:]+scaleH[ih+1,:])/2.  #H(i,j+1/2)

            T_j = temp[ih]  #T(j)
            T_jp1 = temp[ih+1]  #T(j+1)
            T_jph = (temp[ih]+temp[ih+1])/2.  #T(j+1/2)

            Ti_j = temp_ions[ih]  #Ti(j)
            Ti_jp1 = temp_ions[ih+1]  #Ti(j+1)
            Ti_jph = (temp_ions[ih]+temp_ions[ih+1])/2.  #Ti(j+1/2)

            sigma_jph = D_jph * ( 1./H_jph + (1.+alpha)*(T_jp1-T_j)/T_jph/delz ) + K_jph * (1./H0_jph + (T_jp1-T_j)/T_jph/delz)
            sigma_jph[iions] = D_jph[iions] * ( 1./H_jph[iions] + (1.+alpha[iions])*(Ti_jp1-Ti_j)/Ti_jph/delz ) + K_jph[iions] * (1./H0_jph + (Ti_jp1-Ti_j)/Ti_jph/delz)

            for igas in range(ngas):

                if typelbc[igas]==1: #Fixed density
                    Adiff[ih,igas] = 0.
                    Bdiff[ih,igas] = 0.
                    Cdiff[ih,igas] = 0.
                    Ddiff[ih,igas] = 0.
                elif typelbc[igas]==2: #Fixed flux
                    Adiff[ih,igas] = (K_jph[igas] + D_jph[igas])/delz/delz + sigma_jph[igas]/2/delz
                    Bdiff[ih,igas] = -(K_jph[igas] + D_jph[igas])/delz/delz + sigma_jph[igas]/2/delz
                    Cdiff[ih,igas] = 0.
                    Ddiff[ih,igas] = valuelbc[igas]/delz
                elif typelbc[igas]==3: #Fixed velocity
                    Adiff[ih,igas] = (K_jph[igas] + D_jph[igas])/delz/delz + sigma_jph[igas]/2/delz
                    Bdiff[ih,igas] = -(K_jph[igas] + D_jph[igas])/delz/delz + sigma_jph[igas]/2/delz + valuelbc[igas]/delz
                    Cdiff[ih,igas] = 0.
                    Ddiff[ih,igas] = 0.
                else:
                    raise Exception("Unknown type of lower boundary condition for diffusion: "+str(typelbc[igas]))


        #Upper boundary
        elif ih==len(h)-1:

            D_jmh = (D[ih,:]+D[ih-1,:])/2.  #D(i,j-1/2)
            K_jmh = np.zeros_like(D_jmh)
            K_jmh[ineutrals] = (K[ih]+K[ih-1])/2.    #K(j-1/2), no eddy diffusion for ions
            
            H0_jmh = (scaleH0[ih]+scaleH0[ih-1])/2.  #H0(j-1/2)
            H_jmh = (scaleH[ih,:]+scaleH[ih-1,:])/2.  #H(i,j-1/2)

            T_j = temp[ih]  #T(j)
            T_jm1 = temp[ih-1]  #T(j-1)
            T_jmh = (temp[ih]+temp[ih-1])/2.  #T(j-1/2)

            Ti_j = temp_ions[ih]  #T(j)
            Ti_jm1 = temp_ions[ih-1]  #T(j-1)
            Ti_jmh = (temp_ions[ih]+temp_ions[ih-1])/2.  #T(j-1/2)

            sigma_jmh = D_jmh * ( 1./H_jmh + (1.+alpha)*(T_j-T_jm1)/T_jmh/delz ) + K_jmh * (1./H0_jmh + (T_j-T_jm1)/T_jmh/delz)
            sigma_jmh[iions] = D_jmh[iions] * ( 1./H_jmh[iions] + (1.+alpha[iions])*(Ti_j-Ti_jm1)/Ti_jmh/delz ) + K_jmh[iions] * (1./H0_jmh + (Ti_j-Ti_jm1)/Ti_jmh/delz)

            for igas in range(ngas):

                if typelbc[igas]==1: #Fixed density
                    Adiff[ih,igas] = 0.
                    Bdiff[ih,igas] = 0.
                    Cdiff[ih,igas] = 0.
                    Ddiff[ih,igas] = 0.
                elif typelbc[igas]==2: #Fixed flux
                    Adiff[ih,igas] = 0.
                    Bdiff[ih,igas] = -(K_jmh[igas] + D_jmh[igas])/delz/delz - sigma_jmh[igas]/2/delz
                    Cdiff[ih,igas] = (K_jmh[igas] + D_jmh[igas])/delz/delz - sigma_jmh[igas]/2/delz
                    Ddiff[ih,igas] = -valueubc[igas]/delz
                elif typelbc[igas]==3: #Fixed velocity
                    Adiff[ih,igas] = 0.
                    Bdiff[ih,igas] = -(K_jmh[igas] + D_jmh[igas])/delz/delz - sigma_jmh[igas]/2/delz - valueubc[igas]/delz
                    Cdiff[ih,igas] = (K_jmh[igas] + D_jmh[igas])/delz/delz - sigma_jmh[igas]/2/delz
                    Ddiff[ih,igas] = 0.
                else:
                    raise Exception("Unknown type of upper boundary condition for diffusion: "+str(typeubc[igas]))


        #In-between layers
        else:

            D_jph = (D[ih,:]+D[ih+1,:])/2.  #D(i,j+1/2)
            D_jmh = (D[ih,:]+D[ih-1,:])/2.  #D(i,j-1/2)

            K_jph = np.zeros_like(D_jph)
            K_jmh = np.zeros_like(D_jmh)
            K_jph[ineutrals] = (K[ih]+K[ih+1])/2.    #K(j+1/2) for neutrals, no eddy diffusion for ions
            K_jmh[ineutrals] = (K[ih]+K[ih-1])/2.    #K(j-1/2) for neutrals, no eddy diffusion for ions

            H0_jph = (scaleH0[ih]+scaleH0[ih+1])/2.  #H0(j+1/2)
            H0_jmh = (scaleH0[ih]+scaleH0[ih-1])/2.  #H0(j-1/2)

            H_jph = (scaleH[ih,:]+scaleH[ih+1,:])/2.  #H(i,j+1/2)
            H_jmh = (scaleH[ih,:]+scaleH[ih-1,:])/2.  #H(i,j-1/2)

            T_j = temp[ih]  #T(j)
            T_jm1 = temp[ih-1]  #T(j-1)
            T_jp1 = temp[ih+1]  #T(j+1)
            T_jph = (temp[ih]+temp[ih+1])/2.  #T(j+1/2)
            T_jmh = (temp[ih]+temp[ih-1])/2.  #T(j-1/2)

            Ti_j = temp_ions[ih]  #T(j)
            Ti_jm1 = temp_ions[ih-1]  #T(j-1)
            Ti_jp1 = temp_ions[ih+1]  #T(j+1)
            Ti_jph = (temp_ions[ih]+temp_ions[ih+1])/2.  #T(j+1/2)
            Ti_jmh = (temp_ions[ih]+temp_ions[ih-1])/2.  #T(j-1/2)

            sigma_jph = D_jph * ( 1./H_jph + (1.+alpha)*(T_jp1-T_j)/T_jph/delz ) + K_jph * (1./H0_jph + (T_jp1-T_j)/T_jph/delz)
            sigma_jmh = D_jmh * ( 1./H_jmh + (1.+alpha)*(T_j-T_jm1)/T_jmh/delz ) + K_jmh * (1./H0_jmh + (T_j-T_jm1)/T_jmh/delz)

            sigma_jph[iions] = D_jph[iions] * ( 1./H_jph[iions] + (1.+alpha[iions])*(Ti_jp1-Ti_j)/Ti_jph/delz ) + K_jph[iions] * (1./H0_jph + (Ti_jp1-Ti_j)/Ti_jph/delz)
            sigma_jmh[iions] = D_jmh[iions] * ( 1./H_jmh[iions] + (1.+alpha[iions])*(Ti_j-Ti_jm1)/Ti_jmh/delz ) + K_jmh[iions] * (1./H0_jmh + (Ti_j-Ti_jm1)/Ti_jmh/delz)

            Adiff[ih,:] = (K_jph + D_jph)/delz/delz + sigma_jph/2/delz
            Bdiff[ih,:] = -(K_jph + D_jph)/delz/delz + sigma_jph/2/delz - (K_jmh + D_jmh)/delz/delz - sigma_jmh/2/delz
            Cdiff[ih,:] = (K_jmh + D_jmh)/delz/delz - sigma_jmh/2/delz
            Ddiff[ih,:] = 0.

    return Adiff,Bdiff,Cdiff,Ddiff


@jit(cache=cache)
def calc_diffusion_system(A,B,C,D,Nlay,fix_species=None):
    '''
    Function to calculate the gravity field

    Inputs
    ------
    
    A,B,C,D (nh,ngas) :: Diffusion coefficients for each layer and gas (s-1)
    Nlay(nh,ngas) :: Number density of each gas in each layer (m-3)
    fix_species(nh,ngas) :: Flag indicating if a given gas at a given layer must be fixed 
    
    Outputs
    --------

    J(nlay,nlay,ngas) :: Jacobian matrix (s-1)
    dphidz(nlay,ngas) :: Diffusion term evaluated at density N (m-3 s-1)
    '''
    
    nlay = np.shape(A)[0]
    ngas = np.shape(A)[1]

    J = np.zeros((nlay,nlay,ngas))  #jacobian matrix for diffusion (s-1)
    dphidz = np.zeros((nlay,ngas))  #diffusion term evaluated at density n
    
    for igas in range(ngas):
    

        #Lower boundary
        ###############################################

        ilay = 0

        J[ilay,ilay,igas] = B[ilay,igas]
        J[ilay,ilay+1,igas] = A[ilay,igas]
        dphidz[ilay,igas] = (A[ilay,igas] * Nlay[ilay+1,igas]) + (B[ilay,igas] * Nlay[ilay,igas]) + D[ilay,igas]

        #Upper boundary
        ###############################################

        ilay = nlay - 1
        
        J[ilay,ilay,igas] = B[ilay,igas] 
        J[ilay,ilay-1,igas] = C[ilay,igas]
        dphidz[ilay,igas] = (B[ilay,igas] * Nlay[ilay,igas]) + (C[ilay,igas] * Nlay[ilay-1,igas]) + D[ilay,igas]

        #Inbetween layers
        ###############################################

        for ilay in range(1,nlay-1):
            J[ilay,ilay,igas] = B[ilay,igas]
            J[ilay,ilay-1,igas] = C[ilay,igas]
            J[ilay,ilay+1,igas] = A[ilay,igas]
            dphidz[ilay,igas] = (A[ilay,igas] * Nlay[ilay+1,igas]) + (B[ilay,igas] * Nlay[ilay,igas]) + (C[ilay,igas] * Nlay[ilay-1,igas]) + D[ilay,igas]

    #Re-computing the Jacobian matrix if some species is fixed
    if fix_species is not None:
        for igas in range(ngas):
            for ilay in range(nlay):
                if(fix_species[ilay,igas]==1):
                    J[ilay,:,igas] = 0.0
                    dphidz[ilay,igas] = 0.0

    return J, dphidz