from models.dust_gas_collisions.dust_sputtering import *
import numpy as np
if __name__ == '__main__':
    
    ion_atomic_masses = np.array([1.00784,4.002602,12.011,12.011,14.006,14.006,15.999,15.999])
    ion_atomic_numbers = np.array([1,2,6,6,7,7,8,8])
    ion_charges = np.array([1,1,1,2,1,2,1,2])
    ion_abundances = np.array([1,8.04e-2,0.8*3.62e-8,0.2*3.62e-8,0.8*1.12e-8,0.2*1.12e-8,0.6*2.70e-7,0.4*2.70e-7])
    # ion_abundances = np.array([1,8.04e-2])
    # ion_atomic_masses = np.array([1.00784,4.002602])
    # ion_atomic_numbers = np.array([1,2])
    # ion_charges = np.array([1,1])

    ion_atomic_masses = np.array([15.999])
    ion_atomic_numbers = np.array([8])
    ion_charges = np.array([6])
    ion_abundances = np.array([1])

    
    Tmin = 1e4
    Tmax = 1e10
    nT = 100
    nbins_v = 300
    fig = compare_sputtering_rates(Tmin,Tmax,ion_atomic_masses,
                             ion_atomic_numbers,ion_charges,
                             ion_abundances,nT=nT,nbins_v=nbins_v,label='-4Z')
    
    fig.savefig('./thermal_sputtering_data/dust_sputtering_rate-4Z.pdf',format='pdf',dpi=300)
    
    # ion_abundances = np.array([1,8.04e-2,0.8*3.62e-7,0.2*3.62e-7,0.8*1.12e-7,0.2*1.12e-7,0.6*2.70e-6,0.4*2.70e-6])
    # fig = compare_sputtering_rates(Tmin,Tmax,ion_atomic_masses,
    #                          ion_atomic_numbers,ion_charges,
    #                          ion_abundances,nT=nT,nbins_v=nbins_v,label='-3Z')
    
    # fig.savefig('./thermal_sputtering_data/dust_sputtering_rate-3Z.pdf',format='pdf',dpi=300)

    # ion_abundances = np.array([1,8.04e-2,0.8*3.62e-6,0.2*3.62e-6,0.8*1.12e-6,0.2*1.12e-6,0.6*2.70e-5,0.4*2.70e-5])    
    # fig = compare_sputtering_rates(Tmin,Tmax,ion_atomic_masses,
    #                          ion_atomic_numbers,ion_charges,
    #                          ion_abundances,nT=nT,nbins_v=nbins_v,label='-2Z')
    
    # fig.savefig('./thermal_sputtering_data/dust_sputtering_rate-2Z.pdf',format='pdf',dpi=300)
    
    # ion_abundances = np.array([1,8.04e-2,0.8*3.62e-5,0.2*3.62e-5,0.8*1.12e-5,0.2*1.12e-5,0.6*2.70e-4,0.4*2.70e-4])        
    # fig = compare_sputtering_rates(Tmin,Tmax,ion_atomic_masses,
    #                          ion_atomic_numbers,ion_charges,
    #                          ion_abundances,nT=nT,nbins_v=nbins_v,label='-1Z')
    
    # fig.savefig('./thermal_sputtering_data/dust_sputtering_rate-1Z.pdf',format='pdf',dpi=300)
    
    # ion_abundances = np.array([1,8.04e-2,0.8*3.62e-4,0.2*3.62e-4,0.8*1.12e-4,0.2*1.12e-4,0.6*2.70e-3,0.4*2.70e-3])        
    # fig = compare_sputtering_rates(Tmin,Tmax,ion_atomic_masses,
    #                          ion_atomic_numbers,ion_charges,
    #                          ion_abundances,nT=nT,nbins_v=nbins_v,label='0Z')
    
    # fig.savefig('./thermal_sputtering_data/dust_sputtering_rate0Z.pdf',format='pdf',dpi=300)
    # ion_abundances = np.array([1,8.04e-2,0.8*3.62e-3,0.2*3.62e-3,0.8*1.12e-3,0.2*1.12e-3,0.6*2.70e-2,0.4*2.70e-2])        
    # fig = compare_sputtering_rates(Tmin,Tmax,ion_atomic_masses,
    #                          ion_atomic_numbers,ion_charges,
    #                          ion_abundances,nT=nT,nbins_v=nbins_v,label='1Z')
    
    # fig.savefig('./thermal_sputtering_data/dust_sputtering_rate1Z.pdf',format='pdf',dpi=300)