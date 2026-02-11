"""
DUST COOLING AND HEATING MODELS

These functions allow the comparison of different
cooling and heating models for dust and in checking in which 
environments they dominate.

By: Curro Rodriguez (currodri@gmail.com)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
import argparse

BERNEPATH = '/home/currodri/Codes/photoelectric-heating'
sys.path.append(BERNEPATH)
from four_levels_model import HeatingGas
from radiation_fields import radiation_field

def Berne22_efficiency(args,ax,fig):
    n_ne = 20
    n_G0 = 20
    G0_list = np.logspace(np.log10(args.G0min),np.log10(args.G0max),n_G0)
    ne_list = np.logspace(np.log10(args.ne_min),np.log10(args.ne_max),n_ne)
    colors = pl.cm.jet(np.linspace(0,1,n_ne))
    Z = [[0,0],[0,0]]
    CS3 = ax.contourf(Z, ne_list, cmap='jet',norm=mpl.colors.LogNorm())
    # fig.clf()
    for j in range(0, n_ne):
        eff = np.zeros(n_G0)
        gamma = np.zeros(n_G0)
        PEH = HeatingGas('ISRF/habing1968.txt',1,args.T,ne_list[j],
                            54,1,ISRF=True)
        for k in range(0,n_G0):
            result = PEH.parameters(G0_list[k])
            eff[k] = result[1]
            gamma[k] = result[3]
        ax.plot(gamma,eff,color=colors[j])
    
    axcb = fig.colorbar(CS3, orientation="vertical", pad=0.0)
    axcb.set_label(r'$n_e$ [cm$^{-3}$]',fontsize=16)

def Tielens2001_efficiency(args,ax):
    n_gamma = 20
    gamma = np.logspace(np.log10(1),np.log10(1e+15),n_gamma)
    eff = 0.06 / (1.0 + 7e-5*gamma)
    ax.plot(gamma,eff,color='k',linestyle=':',label='Tielens 2001')
    
def Wolfire2003_efficiency(args,ax):
    n_gamma = 20
    gamma = np.logspace(np.log10(1),np.log10(1e+15),n_gamma)
    eff = 4.9e-2/(1.0+2.411e-3*gamma**0.73) + 3.7e-2*(args.T/1e4)**0.7/(1.0+1e-4*gamma)
    ax.plot(gamma,eff,color='k',linestyle='-.',label='Wolfire et al. 2003')

def HollenbachMcKee89_cooling(args,ax):
    
    kb = 1.380649e-16
    mp = 1.67262192e-24
    T = np.logspace(np.log10(args.T_min),np.log10(args.T_max),100)

    H = np.pi*(2*kb)**(3/2)*np.sqrt(T/mp)*args.ne*(T-args.Tgr)*(args.a*1e-3)**2
    ax.plot(T,H,color='red',linestyle='--',label='Hollenbach & McKee 1979')

def DwekWerner81_cooling(args,ax):

    T = np.logspace(np.log10(args.T_min),np.log10(args.T_max),100)

    x = 2.71e8*(args.a)**(2/3)/T

    H = np.zeros(100)

    for i in range(0, len(H)):
        if x[i] > 4.5:
            H[i] = 5.38e-18*args.ne*(args.a)**2*T[i]**(3/2)
        elif 4.5>x[i]>1.5:
            H[i] = 3.37e-13*args.ne*(args.a)**2.41*T[i]**(0.88)
        else:
            H[i] = 6.48e-6*args.ne*(args.a)**3
    
    ax.plot(T,H,color='blue',linestyle='-',label='Dwek & Werner 1981')



if __name__ == '__main__':

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description='Plotting cooling and heating functions for dust')
    parser.add_argument('model', type=str, nargs='+', help='Model name to be included in plot.')
    parser.add_argument('--G0min', type=float, default=1e-4, help='Minimum Habing intensity of the UV field.')
    parser.add_argument('--G0max', type=float, default=1e+6, help='Maximum Habing intensity of the UV field.')
    parser.add_argument('--T_min', type=float, default=2, help='Minimum temperature of the gas in K.')
    parser.add_argument('--T', type=float, default=500, help='Temperature of the gas in K.')
    parser.add_argument('--Tgr', type=float, default=25, help='Grain temperature in K.')
    parser.add_argument('--a', type=float, default=0.1, help='Grain size in micrometre.')
    parser.add_argument('--T_max', type=float, default=1e+9, help='Maximum temperature of the gas in K.')
    parser.add_argument('--ne_min', type=float, default=3e-6, help='Minimum electron number density.')
    parser.add_argument('--ne', type=float, default=3e-3, help='Electron number density.')
    parser.add_argument('--ne_max', type=float, default=1e+3, help='Minimum electron number density.')

    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$\epsilon_{\Gamma},\epsilon_{\rm PAH}$', fontsize=16)
    ax.set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xlim([10,1e+6])
    ax.set_ylim([1e-4,1])

    mydir = os.getcwd()

    for i in range(0,len(args.model)):
        if args.model[i] == 'Berne+22':
            os.chdir(BERNEPATH)
            Berne22_efficiency(args,ax,fig)
            os.chdir(mydir)
        elif args.model[i] == 'Tielens2001':
            Tielens2001_efficiency(args,ax)
        elif args.model[i] == 'Wolfire+2003':
            Wolfire2003_efficiency(args,ax)
            
    ax.text(0.65, 0.9, r'$T=$%i K'%int(args.T),
                        transform=ax.transAxes, fontsize=16,verticalalignment='top',
                        color='black')
    ax.legend(loc='lower left',fontsize=14,frameon=False)

    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.95)
    fig.savefig('dust_heating_efficiency_'+str(int(args.T))+'K.png', format='png', dpi=300)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$H_{\rm coll}$ [erg/s]', fontsize=16)
    ax.set_xlabel(r'$T$ [K]',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    #ax.set_xlim([10,1e+6])
    #ax.set_ylim([1e-4,1])

    mydir = os.getcwd()

    for i in range(0,len(args.model)):
        if args.model[i] == 'Dwek81':
            DwekWerner81_cooling(args,ax)
        elif args.model[i] == 'Hollenbach79':
            HollenbachMcKee89_cooling(args,ax)
    ax.text(0.65, 0.9, r'$n_{e}=$%.4f cm$^{-3}$'%args.ne,
                        transform=ax.transAxes, fontsize=16,verticalalignment='top',
                        color='black')
    ax.legend(loc='lower left',fontsize=14,frameon=False)

    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.95)
    fig.savefig('dust_cooling_'+str(args.ne)+'cm-3.png', format='png', dpi=300)


