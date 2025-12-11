import numpy as np
import matplotlib.pyplot as plt
import sys,os
from numba import jit,njit


########################################################################################################################

@jit(nopython=True)
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


@jit(nopython=True)
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


@jit(nopython=True)
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


@jit(nopython=True)
def calc_diffusion_coefficients(h,temp,scaleH0,scaleH,K,D,B):
    '''
    Function to calculate the gravity field

    Inputs
    ------
    
    h(nh) :: Altitude (m)
    temp(nh) :: Temperature (K)
    scaleH0(nh) :: Mean scale height (m)
    scaleH(nh,ngas) :: Scale height for each gas (m)
    K(nh) :: Eddy diffusion coefficient (m2 s-1)
    D(nh,ngas) :: Molecular diffusion coefficient (m2 s-1)
    B(ngas) :: Molecular thermal diffusion coefficient of each gas
    

    Outputs
    --------

    ksi(nh,ngas), klsi(nh,ngas), ksim1(nh,ngas), klsim1(nh,ngas) :: Coefficients in each layer to calculate the Jacobian (s-1)
    '''
    
    delz = h[1] - h[0]   #Width of each layer, assumed to be constant (m)
    
    nh = np.shape(scaleH)[0] 
    ngas = np.shape(scaleH)[1]
    
    ksi = np.zeros((nh,ngas))
    klsi = np.zeros((nh,ngas))
    ksim1 = np.zeros((nh,ngas))
    klsim1 = np.zeros((nh,ngas))
    
    for ih in range(len(h)):

        #Coefficient between layers i and i-1
        if ih>0:
            
            K_i = (K[ih]+K[ih-1])/2.
            D_i = (D[ih,:]+D[ih-1,:])/2.
            T_i = (temp[ih]+temp[ih-1])/2.
            H0_i = (scaleH0[ih]+scaleH0[ih-1])/2.
            H_i = (scaleH[ih,:]+scaleH[ih-1,:])/2.
      
            ksim1[ih,:] = (D_i+K_i)/delz/delz
            klsim1[ih,:] = (K_i/delz/delz) * (1.0 - delz/H0_i - (temp[ih]-temp[ih-1])/T_i ) + (D_i/delz/delz) * (1.0 - delz/H_i - (1.0+B)*(temp[ih]-temp[ih-1])/T_i )


        #Coefficients between layers i and i+1
        if ih<len(h)-1:
            
            K_i = (K[ih]+K[ih+1])/2.
            D_i = (D[ih,:]+D[ih+1,:])/2.
            T_i = (temp[ih]+temp[ih+1])/2.
            H0_i = (scaleH0[ih]+scaleH0[ih+1])/2.
            H_i = (scaleH[ih,:]+scaleH[ih+1,:])/2.
            
            ksi[ih,:] = (D_i+K_i)/delz/delz
            klsi[ih,:] = (K_i/delz/delz) * (1.0 - delz/H0_i - (temp[ih+1]-temp[ih])/T_i ) + (D_i/delz/delz) * (1.0 - delz/H_i - (1.0+B)*(temp[ih+1]-temp[ih])/T_i)
    
    
    return ksi,klsi,ksim1,klsim1


@jit(nopython=True)
def calc_jacobian_diffusion(ksi,klsi,ksim1,klsim1,typelbc,typeubc,fix_species):
    '''
    Function to calculate the gravity field

    Inputs
    ------
    
    ksi,klsi,ksim1,klsim1 (nh,ngas) :: Diffusion coefficients for each layer and gas
    typelbc(ngas) :: Type of lower boundary condition (following isochem)
    valuelbc(ngas) :: Value for the lower boundary condition
    typeubc(ngas) :: Type of upper boundary condition (following isochem)
    valueubc(ngas) :: Value for the upper boundary condition
    fix_species(nh,ngas) :: Flag indicating if a given gas at a given layer must be fixed 
    

    Outputs
    --------

    J(nlay,nlay,ngas) :: Jacobian matrix (s-1)
    '''
    
    nlay = np.shape(ksi)[0]
    ngas = np.shape(ksi)[1]
    J = np.zeros((nlay,nlay,ngas))
    
    for igas in range(ngas):
    

        #Lower boundary
        ###############################################

        ilay = 0
        
        if typelbc[igas]==1: #Fixed density
            J[ilay,ilay,igas] = 0.
            J[ilay,ilay+1,igas] = 0.
        elif typelbc[igas]==2:  #Fixed flux
            J[ilay,ilay,igas] = -klsi[ilay,igas]
            J[ilay,ilay+1,igas] = ksi[ilay,igas]


        #Upper boundary
        ###############################################

        ilay = nlay - 1
        
        if typeubc[igas]==1:  #Fixed density
            J[ilay,ilay,igas] = 0.
            J[ilay,ilay-1,igas] = 0.
        elif typeubc[igas]==2: #Fixed flux
            J[ilay,ilay,igas] = -ksim1[ilay,igas]
            J[ilay,ilay-1,igas] = klsim1[ilay,igas]
        elif typeubc[igas]==3: #Fixed velocity
            J[ilay,ilay,igas] = -ksim1[ilay,igas]
            J[ilay,ilay-1,igas] = klsim1[ilay,igas]


        #Inbetween layers
        ###############################################

        for ilay in range(1,nlay-1):
            J[ilay,ilay,igas] = - (klsi[ilay,igas]+ksim1[ilay,igas])
            J[ilay,ilay-1,igas] = klsim1[ilay,igas]
            J[ilay,ilay+1,igas] = ksi[ilay,igas]



    #Re-computing the Jacobian matrix if some species is fixed
    for igas in range(ngas):
        for ilay in range(nlay):
            if(fix_species[ilay,igas]==1):
                J[ilay,:,igas] = 0.0


    return J