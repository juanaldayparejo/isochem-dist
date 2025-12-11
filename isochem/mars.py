import isochem


########################################################################################################################

@jit(nopython=True)
def calc_Keddy(h,num,K0):
    '''
    Function to calculate the Eddy diffusion coefficient with altitude using a function of the type:

        K(z) = K0 * n**-1/2

    Inputs
    ------
    
    h(nh) :: Altitude (m)
    num(nh) :: Atmospheric number density (m-3)
    K0 :: Eddy diffusion coefficient

    Optional inputs
    ----------------

    None
    
    Outputs
    --------

    K(nh) :: Eddy diffusion coefficient (m2 s-1)
    '''
    
    nh = len(h)
    
    z0 = 30.   #km
    
    #Calculating the density at 30 km
    nz0 = np.interp(z0*1.0e3,h,num*1.0e-6)
    
    #Calculating the Eddy diffusion coefficient in cm2 s-1
    K = np.zeros(nh)
    for i in range(nh):
        
        if h[i]>z0*1.0e3:
            K[i] = K0 * (num[i]*1.0e-6)**(-1./2.)
        else:
            K[i] = K0 * nz0**(-1./2.)
    
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