import h5py
import numpy as np
import sys,os
from isochem import *
import isochem
import inspect, re

'''
Set of routines needed to run the 1D photochemical model

read_hdf5 :: Read the HDF5 file with the correct format
write_hdf5 :: Write the HDF5 file with the correct format

calc_mmol :: Calculate the molecular weight of the gases in the atmosphere
calc_mmean :: Calculate the mean molecular weight of the atmosphere

calc_upper_bc :: Calculate the upper boundary conditions from the dictionary
calc_lower_bc :: Calculate the lower boundary conditions from the dictionary

read_profiles_mars :: Read input profiles for the atmosphere of Mars

ini_layer :: Given some vertical profiles, initialise the properties of each layer

adjust_vmr :: Adjust the VMRs so that they add up to 1

file_lines :: Read the number of lines in a file

'''

######################################################################################

def isochem_path():
    import os
    isochem_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../')
    return isochem_path

######################################################################################

def read_hdf5(runname):
    '''
    Function to read the output file of a simulation

    Outputs
    -------

    h(nlay) :: Altitude (m)
    T(nlay) :: Temperature (K)
    gasID(ngas) :: ID of the gases
    isoID(ngas) :: ID of the isotopes
    time(nt) :: Time of simulation in each period (seconds)
    N(nlay,ngas,nt) :: Number density of each species (m-3)
    '''

    f = h5py.File(runname+'.h5','r')

    h = np.array(f['h'])
    gasID = np.array(f['gasID'],dtype='int32')
    isoID = np.array(f['isoID'],dtype='int32')
    T = np.array(f['T'])
    N = np.array(f['N'])
    time = np.array(f['time'])

    time = np.resize(time,(time.shape[1]))


    return h,T,gasID,isoID,N,time

######################################################################################

def write_ini_hdf5(runname,h,T,gasID,isoID,N):
    '''
    Function to write the initial HDF5 file for a simulation

    Inputs
    -------

    runname :: Simulation run name
    h(nh) :: Altitude (km)
    T(nh) :: Temperature (K)
    gasID(ngas) :: Gas IDs
    isoID(ngas) :: Isotope IDs
    N(nlay,ngas) :: Number density of each species (m-3)
    '''

    nlay = len(h)
    ngas = len(gasID)

    hf = h5py.File(runname+'.h5','w')
    hf.create_dataset('h', data=h)
    hf.create_dataset('T', data=T)
    hf.create_dataset('gasID', data=gasID)
    hf.create_dataset('isoID', data=isoID)
    hf.create_dataset('N', data=N[:,:, np.newaxis], maxshape=(nlay,ngas,None))
    ztime = np.array([0.])
    hf.create_dataset('time', data=ztime[:,np.newaxis], maxshape=(1,None))
    hf.close()

######################################################################################

def combine_isotopes(gasID,isoID,gasID_comb,N):
    '''
    Function to combine the isotopologues into a single species

    Inputs
    -------

    gasID(ngas) :: Gas IDs
    isoID(ngas) :: Isotope IDs
    gasID_comb :: Gas ID of the isotopes to combine
    N(nlay,ngas) :: Number density of each species (m-3)
    
    Outputs
    -------
    
    gasID_new(ngasnew) :: New Gas IDs
    isoID_new(ngasnew) :: New Isotope IDs
    N_new(nlay,ngasnew) :: New Number density of each species (m-3
    
    '''

    ngas = len(gasID)

    igas = np.where(gasID==gasID_comb)[0]
    
    if len(igas)>1:
        ngasnew = ngas - len(igas) + 1
        gasID_new = np.zeros(ngasnew,dtype='int32')
        isoID_new = np.zeros(ngasnew,dtype='int32')
        N_new = np.zeros((N.shape[0],ngasnew))

        k = 0
        for i in range(ngas):
            if i==igas[0]:
                gasID_new[k] = gasID_comb
                isoID_new[k] = 0
                N_new[:,k] = np.sum(N[:,igas],axis=1)
                k = k + 1
            elif i in igas[1:]:
                continue
            else:
                gasID_new[k] = gasID[i]
                isoID_new[k] = isoID[i]
                N_new[:,k] = N[:,i]
                k = k + 1
    
    return gasID_new,isoID_new,N_new

#############################################################################

def read_gasname(gasID,isoID):
    '''
    Routine to read the name of the gases from the python dictionary

    Inputs
    ------
    gasID(ngas) :: Gas ID
    isoID(ngas) :: Isotope ID
    '''

    ngas = len(gasID)
    gasname = ['']*ngas
    for i in range(ngas):

        if isoID[i]!=0:
            gasname[i] = isochem.dict.gas_dict.gas_info[str(gasID[i])]['isotope'][str(isoID[i])]['name']
        else:
            gasname[i] = isochem.dict.gas_dict.gas_info[str(gasID[i])]['name']

    return gasname

#############################################################################

def read_gaslabel(gasID,isoID):
    '''
    Routine to read the name of the gases from the python dictionary
    with the LaTeX notation to include in plots

    Inputs
    ------
    gasID(ngas) :: Gas ID
    isoID(ngas) :: Isotope ID
    '''

    ngas = len(gasID)
    gasname = ['']*ngas
    for i in range(ngas):

        if isoID[i]!=0:
            gasname[i] = isochem.dict.gas_dictgas_info[str(gasID[i])]['isotope'][str(isoID[i])]['label']
        else:
            gasname[i] = isochem.dict.gas_dict.gas_info[str(gasID[i])]['label']

    return gasname

######################################################################################

def calc_mmol(gasID,isoID):

    '''
    Routine to calculate the molecular weight of each species (g mol-1)

    Inputs
    ------
    
    gasID(ngas) :: Gas ID
    isoID(ngas) :: Isotopologue ID

    '''

    ngas = len(gasID)
    mmol = np.zeros(ngas)
    for i in range(ngas):

        if isoID[i]!=0:
            mmol1 = isochem.dict.gas_dict.gas_info[str(gasID[i])]['isotope'][str(isoID[i])]['mass']
        else:
            mmol1 = isochem.dict.gas_dict.gas_info[str(gasID[i])]['mmw']

        mmol[i] = mmol1

    return mmol

######################################################################################

def calc_mmean(vmr,mmol):

    '''
    Routine to calculate the mean molecular weight of the atmosphere at each level (g mol-1)

    Inputs
    ------
    
    vmr(nh,ngas) :: volume mixing ratio of each species at each altitude
    mmol(ngas) :: Molecular weight of each of the species included

    '''

    mmean = np.sum(vmr * mmol,axis=1)/np.sum(vmr,axis=1)

    return mmean

######################################################################################

def read_moldiff_params(gasID,isoID,planet):
    '''
    Function to read the parameters defining the molecular diffusion coefficient
    from the python dictionary (A,s).

    The molecular diffusion coefficient is then calculated using:

        D_i = A_i * temp**s_i / numdens

    The thermal diffusion coefficient B is also read in this function.

    '''

    ngas = len(gasID)

    A = np.zeros(ngas) 
    s = np.zeros(ngas)
    B = np.zeros(ngas)

    for i in range(ngas):

        if isochem.dict.planet_dict.diffusion_coeff[planet].get(str(gasID[i])) is not None:
            A[i] = isochem.dict.planet_dict.diffusion_coeff[planet][str(gasID[i])]['A']
            s[i] = isochem.dict.planet_dict.diffusion_coeff[planet][str(gasID[i])]['s']
            B[i] = isochem.dict.planet_dict.diffusion_coeff[planet][str(gasID[i])]['Btherm']
        else:
            A[i] = 1.0e17
            s[i] = 0.75
            B[i] = 0.0

    return A,s,B


######################################################################################

def calc_upper_bc(gasID,isoID,planet):

    '''
    Routine to read the upper boundary conditions from the dictionary.

    If a given species is not present in the dictionary then it is assumed that
    the upper boundary condition is given by a fixed flux of 0.0

    Inputs
    ------

    gasID :: Gas ID
    isoID :: Isotope ID
    planet :: Planet name (e.g., 'Mars','Venus')

    '''

    ngas = len(gasID)

    type = np.zeros(ngas,dtype='int32')
    value = np.zeros(ngas)

    for i in range(ngas):

        if isochem.dict.planet_dict.upper_bc[planet].get(str(gasID[i])) is not None:
            type[i] = isochem.dict.planet_dict.upper_bc[planet][str(gasID[i])]['type']
            value[i] = isochem.dict.planet_dict.upper_bc[planet][str(gasID[i])]['value']
        else:
            type[i] = 2
            value[i] = 0.0

    return type,value

######################################################################################

def calc_lower_bc(gasID,isoID,planet):

    '''
    Routine to read the lower boundary conditions from the dictionary.

    If a given species is not present in the dictionary then it is assumed that
    the lower boundary condition is given by a fixed flux of 0.0

    Inputs
    ------

    gasID :: Gas ID
    isoID :: Isotope ID
    planet :: Planet name (e.g., 'Mars','Venus')

    '''

    ngas = len(gasID)

    type = np.zeros(ngas,dtype='int32')
    value = np.zeros(ngas)

    for i in range(ngas):

        if isochem.dict.planet_dict.lower_bc[planet].get(str(gasID[i])) is not None:
            type[i] = isochem.dict.planet_dict.lower_bc[planet][str(gasID[i])]['type']
            value[i] = isochem.dict.planet_dict.lower_bc[planet][str(gasID[i])]['value']
        else:
            type[i] = 2
            value[i] = 0.0

    return type,value

###############################################################################################

def ini_layer(h,temp,numdens,vmr,hlay):
    '''

    Function to initialise the atmospheric profiles

    Outputs
    --------

    zh(nh) :: Altitude of the boundaries of each layer (m)
    temp(nh) :: Temperature at the boundaries of each layer (K)
    press(nh) :: Pressure at the boundaries of each layer (Pa)
    numdens(nh) :: Number density at the boundaries of each layer (m-3)
    vmr(nh,ngas) :: Volume mixing ratio of each species at the boundaries of each layer

    '''

    from scipy.interpolate import interp1d

    nh = vmr.shape[0]
    ngas = vmr.shape[1]

    #Calculating the properties of each layer, assuming our profiles are the boundaries of the layer
    nlay = nh - 1
    Tlay = np.zeros(nlay)
    N0lay = np.zeros(nlay)
    VMRlay = np.zeros((nlay,ngas))

    #Interpolating temperatures
    f = interp1d(h,temp)
    Tlay = f(hlay)

    #Interpolating densities
    f = interp1d(h,np.log(numdens))
    N0lay = np.exp(f(hlay))

    #Interpolating VMRs
    f = interp1d(h,vmr,axis=0)
    VMRlay = f(hlay)

    Play = N0lay * phys_const['k_B'] * Tlay

    return hlay,Tlay,Play,N0lay,VMRlay

###############################################################################################

def adjust_vmr(vmr):

    '''
    Routine to adjust the VMR values so that they add up to 1.0 at all altitudes

    Inputs
    ------

    vmr(nh,ngas) :: volume mixing ratio of each species at each altitude

    '''

    nh = vmr.shape[0]
    ngas = vmr.shape[1]

    #Making sure that the sum of the volume mixing ratios add up to 1.0
    #dominent species = 1.0 - sum(all other species)
    for i in range(nh):

        vmrs = vmr[i,:]
        jmax = np.argmax(vmrs)

        vmr_rest = 0.0
        vmr_dom = 0.0
        for j in range(ngas):
            if j!=jmax:
                vmr_rest = vmr_rest + vmr[i,j]
            else:
                vmr_dom = vmr[i,j]

        vmr[i,jmax] = 1.0 - vmr_rest

    return vmr

###############################################################################################

def find_reactions_atmosphere(gasID,isoID):

    """
    FUNCTION NAME : file_lines()

    DESCRIPTION : Returns the index of the reactions involving the gases in the atmosphere
                  If a reaction involves species not in the atmosphere, it is ignored

    INPUTS : 
 
        gasID(ngas) :: Gas IDs in the atmosphere
        isoID(ngas) :: Isotope IDs in the atmosphere

    OPTIONAL INPUTS: none
            
    OUTPUTS : 
 
        reaction_ids :: List of reaction indices involving only species in the atmosphere

    CALLING SEQUENCE:

        reaction_ids = find_reactions_atmosphere(gasID,isoID)

    MODIFICATION HISTORY : Juan Alday (11/12/2025)

    """

    #Getting information about the available reactions
    reactions_all = isochem.chemistry.get_reaction_ids()
    gasIDx = np.array([2,7,22,45],dtype='int32')
    isoIDx = np.zeros(4,dtype='int32')
    h = np.zeros(3) ; p = np.ones(3) ; t = np.ones(3)
    n = np.ones((3,4),dtype='float64')

    rtype, ns, sf, sID, sISO, npr, pf, pID, pISO, rrates = isochem.chemistry.reaction_rate_coefficients(reactions_all, gasIDx, isoIDx, h, p, t, n)
    
    # Build a set for the membership tests
    allowed = set(zip(gasID, isoID))

    reaction_ids = []
    for i in range(len(reactions_all)):

        # Check sources
        sources = set(zip(sID[:ns[i], i], sISO[:ns[i], i]))
        if not sources.issubset(allowed):
            continue

        # Check products
        products = set(zip(pID[:npr[i], i], pISO[:npr[i], i]))
        if not products.issubset(allowed):
            continue

        reaction_ids.append(reactions_all[i])
    
    return reaction_ids
    
###############################################################################################

def file_lines(fname):

    """
    FUNCTION NAME : file_lines()

    DESCRIPTION : Returns the number of lines in a given file

    INPUTS : 
 
        fname :: Name of the file

    OPTIONAL INPUTS: none
            
    OUTPUTS : 
 
        nlines :: Number of lines in file

    CALLING SEQUENCE:

        nlines = file_lines(fname)

    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


    