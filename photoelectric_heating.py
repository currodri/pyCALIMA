"""
PHOTOELECTRIC HEATING BY DUST AND PAHS

The scripts below are intended for the comparison of different computations
of the very important photoelectric heating on dust grains and PAHs.

By: F. Rodriguez Montero (currodri@gmail.com)
"""

# Import libraries
import numpy as np
import pandas as pd
from dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,LogNormal_Distribution
from dust_model import relative_velocity,grain_charge_dist,cmp_D_WD99
from unyt import mh,kb

def plot_charge_evolution(agrain,grain_type='carbonaceous',nGamma=100):
    """Make a plot for the evolution of the grain charges with the
    charging parameter gamma.

    Args:
        agrain (float): Grain radius in Angstrom
        grain_type (str): Grain type (either 'carbonaceous' or 'silicate'). Defaults to 'carbonaceous'.
        nGamma (int, optional): Number of gamma parameters to obtain. Defaults to 100.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    fig, ax = plt.subplots(1,1, figsize=(6,5),dpi=300,facecolor='w',edgecolor='k')

    
    gamma = np.logspace(-1,5,nGamma)
    minZ = 0
    maxZ = 0
    for i in range(0, nGamma):
        f,Z = grain_charge_dist(0,0,0,grain_type,str(agrain)+'A',gamma[i])
        minZ = min(min(Z),minZ)
        maxZ = max(max(Z),maxZ)
    Zrange = np.arange(minZ,maxZ+1)
    data = np.zeros((len(Zrange),nGamma))
    for i in range(0, nGamma):
        f,Z = grain_charge_dist(0,0,0,grain_type,str(agrain)+'A',gamma[i])
        for j in range(0, len(Z)):
            data[np.where(Zrange==Z[j])[0][0],i] = f[j]
            
    for i in range(len(Zrange)):
        ax.plot(gamma,data[i,:], label=rf'$Z={Zrange[i]}$')
        
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.text(0.02, 0.20, r'$a_{\rm grain} = %.2f \AA$'%(agrain),
                transform=ax.transAxes, fontsize=16,verticalalignment='top',
                color='black', weight='bold')
    ax.set_ylabel(r'$f(Z)$',fontsize=16)
    ax.set_xlabel(r'$\gamma(G_0 \sqrt{T}/n_e)$')
    ax.legend(loc='best',fontsize=12,frameon=False,ncol=2)
    fig.subplots_adjust(top=0.98,bottom=0.1,left=0.12,right=0.99)
    fig.savefig('charge_evolution_%s_angstrom.png'%(str(agrain)),format='png',dpi=300)
    
    
    