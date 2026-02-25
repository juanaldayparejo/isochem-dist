import isochem
import numpy as np
import matplotlib.pyplot as plt
import os,sys,shutil
import h5py
import time
os.environ["ISOCHEM_USE_JIT"] = "1"

start_time_simulation = time.time()  # Record start time

# Reading the input files
#############################################################################

atm_file = 'mars_mcd_aphelion_ini'
xs_file = 'reference_xs'
sol_file = '/exomars/projects/ja22256/isochem-dist/data/Solar_Spectrum/atlas3_thuillier_tuv'

#Copying the atmospheric file so that it is not overwritten
new_atm_file = 'mars_mcd_aphelion'
shutil.copyfile(os.path.join(os.path.dirname(__file__), atm_file+'.h5'),
                os.path.join(os.path.dirname(__file__), new_atm_file+'.h5'))
print('Reading input files...')

#Reading the inputs
gasID, isoID, hlay, Play, Tlay, Nlay,\
wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,\
mmol,A,s,B,\
typelbc,valuelbc,typeubc,valueubc = isochem.converge.initialise_run(atm_file,xs_file,sol_file,planet='Mars')

#Fixing species
##############################################################################

print('Fixing species profiles...')

#Fixing the H2O profile from 0 to 50 km
gasID_fix = np.array([1])   #H2O
isoID_fix = np.array([0])   #H2O

hmin_fix = np.array([0.])   #Altitude boundaries at which to fix the profile
hmax_fix = np.array([70.0]) 

nlay = len(hlay)
ngas = len(gasID)
fix_species = np.zeros((nlay,ngas),dtype='int32')
for igasx in range(len(gasID_fix)):

    il = np.where( (hlay>=hmin_fix[igasx]*1000.) & (hlay<=hmax_fix[igasx]*1000.) )
    il = il[0]

    ifix = 0
    for igas in range(ngas):
        if ( (gasID_fix[igasx]==gasID[igas]) & (isoID_fix[igasx]==isoID[igas]) ):

            fix_species[il,igas] = 1
            ifix = 1
        
    if ifix==0:
        sys.exit('error :: The gas species to be fixed cannot be found in gas list')


#Defining the chemical network
##############################################################################

print('Defining chemical network...')

reaction_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 40, 42, 45])

#Running the model with increasing steps
##############################################################################

#Defining the time steps
niter_per_timestep = 1000
dt_steps = np.logspace(-3,3,7)  #in seconds
nsteps = len(dt_steps)

#Defining arrays to hold current profiles
Pcurr = np.zeros(nlay)
Pcurr[:] = Play[:]
Ncurr = np.zeros((nlay,ngas))
Ncurr[:,:] = Nlay[:,:]

timex = 0.0
for step in range(nsteps):

    dt = dt_steps[step]

    print('Running time step {0}/{1} with dt={2:.2e} seconds for {3} iterations'.format(step+1,nsteps,dt,niter_per_timestep))

    Nnew, tnew = isochem.converge.run_model_implicit(gasID, isoID, hlay, Pcurr, Tlay, Ncurr,                              #Atmospheric profiles
                        reaction_ids,                                                                                   #Chemical network
                        wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,  #Photolysis
                        mmol,A,s,B,                                                                                     #Diffusion
                        typelbc,valuelbc,typeubc,valueubc,                                                              #Boundary conditions
                        dt,                                                                                             #Timestep parameters
                        fix_species=fix_species, 
                        planet='Mars',
                        max_iter=niter_per_timestep,
                        time=timex,
                        include_chemistry=True,
                        include_diffusion=True,
                        include_13c=False)
    
    #Updating current profiles
    if np.where(np.isnan(Nnew))[0].size > 0:
        raise ValueError('error :: NaN values encountered during the model run')

    Ncurr[:,:] = Nnew[:,:]
    Pcurr = np.sum(Nnew,axis=1)*isochem.dict.const_dict.phys_const["k_B"]*Tlay
    timex = tnew
    
    #Writing output files
    hf = h5py.File(new_atm_file+'.h5','r+')
    hf['N'].resize((hf['N'].shape[2] + 1), axis=2)
    hf['N'][:,:,hf['N'].shape[2]-1] = Nnew[:,:]
    hf['time'].resize((hf['time'].shape[1] + 1), axis=1)
    hf['time'][:,hf['time'].shape[1] - 1] = np.array([timex])[:,np.newaxis]
    hf.close()
    
    
#Running the model with a fixed step for more iterations
##############################################################################

#Defining the time steps
niter_per_timestep = 1000
dt_step = 2500.   #Maximum time step in seconds that I have seen it behaving stably. 1000 iterations correspond to a month.
nsteps = 120      #Running for 10 Martian years

for step in range(nsteps):

    dt = dt_step

    print('Running time step {0}/{1} with dt={2:.2e} seconds'.format(step+1,nsteps,dt))

    Nnew, tnew = isochem.converge.run_model_implicit(gasID, isoID, hlay, Pcurr, Tlay, Ncurr,                              #Atmospheric profiles
                        reaction_ids,                                                                                   #Chemical network
                        wl,wu,wc,sID_xs,sISO_xs,xs,sID_phot,sISO_phot,npr_phot,pID_phot,pISO_phot,pf_phot,xsr,solflux,  #Photolysis
                        mmol,A,s,B,                                                                                     #Diffusion
                        typelbc,valuelbc,typeubc,valueubc,                                                              #Boundary conditions
                        dt,                                                                                             #Timestep parameters
                        fix_species=fix_species, 
                        planet='Mars',
                        max_iter=niter_per_timestep,
                        time=timex,
                        include_chemistry=True,
                        include_diffusion=True,
                        include_13c=False)
    
    #Updating current profiles
    Ncurr[:,:] = Nnew[:,:]
    Pcurr = np.sum(Nnew,axis=1)*isochem.dict.const_dict.phys_const["k_B"]*Tlay
    timex = tnew
    
    #Writing output files
    hf = h5py.File(new_atm_file+'.h5','r+')
    hf['N'].resize((hf['N'].shape[2] + 1), axis=2)
    hf['N'][:,:,hf['N'].shape[2]-1] = Nnew[:,:]
    hf['time'].resize((hf['time'].shape[1] + 1), axis=1)
    hf['time'][:,hf['time'].shape[1] - 1] = np.array([timex])[:,np.newaxis]
    hf.close()
    
end_time_simulation = time.time()  # Record end time
elapsed_time = end_time_simulation - start_time_simulation
print('Simulation completed in {:.2f} seconds.'.format(elapsed_time))