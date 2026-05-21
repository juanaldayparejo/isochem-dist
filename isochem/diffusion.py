import numpy as np
import matplotlib.pyplot as plt
import sys,os
from isochem.jit import jit

cache = True

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
def calc_diffusion_coefficients(h,temp,scaleH0,scaleH,K,D,alpha,typelbc,valuelbc,typeubc,valueubc):
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
    alpha(ngas) :: Molecular thermal diffusion coefficient of each gas
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
     
    for ih in range(len(h)):

        #Lower boundary
        if ih==0:

            K_jph =  (K[ih]+K[ih+1])/2.    #K(j+1/2) 
            D_jph = (D[ih,:]+D[ih+1,:])/2.  #D(i,j+1/2)
            H0_jph = (scaleH0[ih]+scaleH0[ih+1])/2.  #H0(j+1/2)
            H_jph = (scaleH[ih,:]+scaleH[ih+1,:])/2.  #H(i,j+1/2)

            T_j = temp[ih]  #T(j)
            T_jp1 = temp[ih+1]  #T(j+1)
            T_jph = (temp[ih]+temp[ih+1])/2.  #T(j+1/2)

            sigma_jph = D_jph * ( 1./H_jph + (1.+alpha)*(T_jp1-T_j)/T_jph/delz ) + K_jph * (1./H0_jph + (T_jp1-T_j)/T_jph/delz)

            for igas in range(ngas):

                if typelbc[igas]==1: #Fixed density
                    Adiff[ih,igas] = 0.
                    Bdiff[ih,igas] = 0.
                    Cdiff[ih,igas] = 0.
                    Ddiff[ih,igas] = 0.
                elif typelbc[igas]==2: #Fixed flux
                    Adiff[ih,igas] = (K_jph + D_jph[igas])/delz/delz + sigma_jph[igas]/2/delz
                    Bdiff[ih,igas] = -(K_jph + D_jph[igas])/delz/delz + sigma_jph[igas]/2/delz
                    Cdiff[ih,igas] = 0.
                    Ddiff[ih,igas] = valuelbc[igas]/delz
                elif typelbc[igas]==3: #Fixed velocity
                    Adiff[ih,igas] = (K_jph + D_jph[igas])/delz/delz + sigma_jph[igas]/2/delz
                    Bdiff[ih,igas] = -(K_jph + D_jph[igas])/delz/delz + sigma_jph[igas]/2/delz + valuelbc[igas]/delz
                    Cdiff[ih,igas] = 0.
                    Ddiff[ih,igas] = 0.
                else:
                    raise Exception("Unknown type of lower boundary condition for diffusion: "+str(typelbc[igas]))


        #Upper boundary
        elif ih==len(h)-1:

            K_jmh = (K[ih]+K[ih-1])/2.    #K(j-1/2)
            D_jmh = (D[ih,:]+D[ih-1,:])/2.  #D(i,j-1/2)
            H0_jmh = (scaleH0[ih]+scaleH0[ih-1])/2.  #H0(j-1/2)
            H_jmh = (scaleH[ih,:]+scaleH[ih-1,:])/2.  #H(i,j-1/2)

            T_j = temp[ih]  #T(j)
            T_jm1 = temp[ih-1]  #T(j-1)
            T_jmh = (temp[ih]+temp[ih-1])/2.  #T(j-1/2)

            sigma_jmh = D_jmh * ( 1./H_jmh + (1.+alpha)*(T_j-T_jm1)/T_jmh/delz ) + K_jmh * (1./H0_jmh + (T_j-T_jm1)/T_jmh/delz)

            for igas in range(ngas):

                if typelbc[igas]==1: #Fixed density
                    Adiff[ih,igas] = 0.
                    Bdiff[ih,igas] = 0.
                    Cdiff[ih,igas] = 0.
                    Ddiff[ih,igas] = 0.
                elif typelbc[igas]==2: #Fixed flux
                    Adiff[ih,igas] = 0.
                    Bdiff[ih,igas] = -(K_jmh + D_jmh[igas])/delz/delz - sigma_jmh[igas]/2/delz
                    Cdiff[ih,igas] = (K_jmh + D_jmh[igas])/delz/delz - sigma_jmh[igas]/2/delz
                    Ddiff[ih,igas] = -valueubc[igas]/delz
                elif typelbc[igas]==3: #Fixed velocity
                    Adiff[ih,igas] = 0.
                    Bdiff[ih,igas] = -(K_jmh + D_jmh[igas])/delz/delz - sigma_jmh[igas]/2/delz - valueubc[igas]/delz
                    Cdiff[ih,igas] = (K_jmh + D_jmh[igas])/delz/delz - sigma_jmh[igas]/2/delz
                    Ddiff[ih,igas] = 0.
                else:
                    raise Exception("Unknown type of upper boundary condition for diffusion: "+str(typeubc[igas]))


        #In-between layers
        else:

            K_jph =  (K[ih]+K[ih+1])/2.    #K(j+1/2) 
            K_jmh = (K[ih]+K[ih-1])/2.    #K(j-1/2)

            D_jph = (D[ih,:]+D[ih+1,:])/2.  #D(i,j+1/2)
            D_jmh = (D[ih,:]+D[ih-1,:])/2.  #D(i,j-1/2)

            H0_jph = (scaleH0[ih]+scaleH0[ih+1])/2.  #H0(j+1/2)
            H0_jmh = (scaleH0[ih]+scaleH0[ih-1])/2.  #H0(j-1/2)

            H_jph = (scaleH[ih,:]+scaleH[ih+1,:])/2.  #H(i,j+1/2)
            H_jmh = (scaleH[ih,:]+scaleH[ih-1,:])/2.  #H(i,j-1/2)

            T_j = temp[ih]  #T(j)
            T_jm1 = temp[ih-1]  #T(j-1)
            T_jp1 = temp[ih+1]  #T(j+1)
            T_jph = (temp[ih]+temp[ih+1])/2.  #T(j+1/2)
            T_jmh = (temp[ih]+temp[ih-1])/2.  #T(j-1/2)

            sigma_jph = D_jph * ( 1./H_jph + (1.+alpha)*(T_jp1-T_j)/T_jph/delz ) + K_jph * (1./H0_jph + (T_jp1-T_j)/T_jph/delz)
            sigma_jmh = D_jmh * ( 1./H_jmh + (1.+alpha)*(T_j-T_jm1)/T_jmh/delz ) + K_jmh * (1./H0_jmh + (T_j-T_jm1)/T_jmh/delz)

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