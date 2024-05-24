from dust_sputtering import *
import numpy as np
if __name__ == '__main__':
    
    ion_atomic_masses = np.array([1.00784,4.002602,12.011,12.011,14.006,14.006,15.999,15.999])
    ion_atomic_numbers = np.array([1,2,6,6,7,7,8,8])
    ion_charges = np.array([1,1,1,2,1,2,1,2])
    ion_abundances = np.array([1,8.04e-2,0.8*3.62e-8,0.2*3.62e-8,0.8*1.12e-8,0.2*1.12e-8,0.6*2.70e-7,0.4*2.70e-7])
    
    Tmin = 1e4
    Tmax = 1e10
    nT = 200
    nbins_v = 300
    
    export_rates(Tmin,Tmax,ion_atomic_masses,
                 ion_atomic_numbers,ion_charges,
                 ion_abundances,nT=nT,nbins_v=nbins_v)