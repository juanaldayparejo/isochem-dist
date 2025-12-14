import isochem
from isochem.jit import jit
import numpy as np


########################################################################################################################

@jit(nopython=True)
def calc_Keddy(h,num,K0,Ktype=3):
    '''
    Function to calculate the Eddy diffusion coefficient with altitude.
    
    If Ktype=1 then the Eddy diffusion coefficient is constant with altitude:
    
        K(z) = K0 (K0 in cm2 s-1)
        
    If Ktype=2 then the Eddy diffusion coefficient varies with altitude as:

        K(z) = K0 * n**-1/2
        
    If Ktype=3 then the Eddy diffusion coefficient varies with altitude as:

        K(z) = K0 * n(z)**-1/2   for n(z)>=1.3e40/K0**2
        K(z) = K0**2 / np.sqrt(1.3e40)   for n(z)<1.3e40/K0**2

    Inputs
    ------
    
    h(nh) :: Altitude (m)
    num(nh) :: Atmospheric number density (m-3)
    K0 :: Eddy diffusion coefficient

    Optional inputs
    ----------------

    Ktype :: Type of Eddy diffusion coefficient variation with altitude (default=3)
    
    Outputs
    --------

    K(nh) :: Eddy diffusion coefficient (m2 s-1)
    '''
    
    nh = len(h)
    
    #Calculating the Eddy diffusion coefficient in cm2 s-1
    K = np.zeros(nh)
    for i in range(nh):
        
        if Ktype == 1:
            K[i] = K0
        elif Ktype == 2:
            K[i] = K0 * (num[i]*1.0e-6)**(-1./2.)
        elif Ktype == 3:
            nz0 = 1.3e40 / (K0**2)  #Number density at z0
            iabove = np.where( num*1.0e-6 < nz0 )[0]
            ibelow = np.where( num*1.0e-6 >= nz0 )[0]
            
            if i in iabove:
                K[i] = K0**2 / np.sqrt(1.3e40)
            elif i in ibelow:
                K[i] = K0 * (num[i]*1.0e-6)**(-1./2.)
    
    K = K * 1.0e-4  #Changing units to m2 s-1
    
    return K

########################################################################################################################

@jit(nopython=True)
def calc_grav(h):
    '''
    Function to calculate the gravity field for Mars

    Inputs
    ------
    
    h(nh) :: Altitude (m)
    

    Outputs
    --------

    g(nh) :: Gravity acceleration (m s-2)
    '''
    
    Mplanet = 6.4169e23   #Mass of Mars (kg)
    Rplanet = 3396.0e3    #Radius of Mars (m)
    G = 6.67e-11          #Gravitational constant (m3 kg-1 s-2)
    
    grav = Mplanet * G / (h+Rplanet)**2.0
    
    return grav