"""
DUST SUBLIMATION
"""
# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.constants import *
from unyt import mh,kb

# FUNCTIONS
def photo_sublimation(Umin,Umax,nU=100):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    U = np.logspace(np.log10(Umin),np.log10(Umax),nU)
    
    # Compute the corresponding dust temperatures based on the
    # approximations of Draine (2011) (Eqs. 24.19, 24.20)
    
    T_sil = [16.4*(basic_a0[5]/0.1)**(-1./15.)*U**(1./6.),
             16.4*(basic_a0[6]/0.1)**(-1./15.)*U**(1./6.)]
    
    T_car = [22.3*(basic_a0[2]/0.1)**(-1./40.)*U**(1./6.),
             22.3*(basic_a0[3]/0.1)**(-1./40.)*U**(1./6.)]
        
    # Now compute the sublimation timescales as obtained from 
    # Guhathakurta & Draine (1989) and Waxman and Draine (2000)
    tau_sil = [6.36e3 * (basic_a0[5]/0.1) * np.exp(68100. * (1./T_sil[0] - 1./1800.)),
               6.36e3 * (basic_a0[6]/0.1) * np.exp(68100. * (1./T_sil[1] - 1./1800.))]

    tau_car = [1.36 * (basic_a0[2]/0.1) * np.exp(81200. * (1./T_car[0] - 1./3000.)),
               1.36 * (basic_a0[3]/0.1) * np.exp(81200. * (1./T_car[1] - 1./3000.))]
    print(tau_sil,tau_car)
    
    # Build the figure
    fig, ax = plt.subplots(1,1, figsize=(6,5),dpi=300,facecolor='w',edgecolor='k')
    
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # Add resulting data
    ax.plot(U,tau_sil[0]/sec2Myr,linestyle='-',color='saddlebrown',label='SmallSil')
    ax.plot(U,tau_sil[1]/sec2Myr,linestyle='--',color='sandybrown',label='LargeSil')
    ax.plot(U,tau_car[0]/sec2Myr,linestyle='-',color='steelblue',label='SmallC')
    ax.plot(U,tau_car[1]/sec2Myr,linestyle='--',color='cornflowerblue',label='LargeC')
    
    ax.set_ylabel(r'$t_{\rm sub}$ [Myr]', fontsize=20)
    ax.set_xlabel(r'Draine Field $U$', fontsize=20)
    ax.legend(loc='best', frameon=False, fontsize=14, ncol=2)
    fig.subplots_adjust(top=0.98,bottom=0.1,left=0.15,right=0.99,hspace=0,wspace=0)
    fig.savefig('dust_sublimation.png',format='png',dpi=300)
    plt.close(fig)