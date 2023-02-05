"""
MODELLING DUST DISTRIBUTION AND EVOLUTION

These functions allow for the computation of intrinstic dust
size distributions and how different processes affect their
formation, size and properties.

By: Curro Rodriguez (currodri@gmail.com)
"""

# Import libraries
import numpy as np
import pandas as pd

# Model parameters
basic_a0 = np.array([5e-4,5e-3,1e-1])
basic_amin = np.array([1e-4,1e-4,1e-4])
basic_amax = np.array([2e-3,1,1])
basic_sigma = np.array([0.4,0.75,0.75])
basic_s = np.array([2,2.24,2.24])

# Tielens et al. (1994) - Thermal sputtering rates for silicate 
# and carbonaceous grains. See https://ui.adsabs.harvard.edu/abs/1994ApJ...431..321T/abstract

thermal_spu = {'Sil':{'a0':-2.7446,'a1':1.5439,'a2':-0.37046,'a3':0.21641,'a4':-0.34755,'a5':0.10114},
               'Car':{'a0':-2.8605,'a1':1.0572,'a2':-0.27545,'a3':0.23735,'a4':-0.31820,'a5':0.087376}}

class LogNormal_Distribution(object):

    def __init__(self,a0,amin,amax,sigma,grain_density):
        self.a0 = a0
        self.amin = amin
        self.amax = amax
        self.sigma = sigma
        self.a = np.logspace(np.log10(amin),np.log10(amax),1000)
        self.grain_density = grain_density
        self.sintegral = self._init_integral()

    def _init_integral(self):
        y = (1.0/self.a) * np.exp(-(np.log10(self.a/self.a0))**2/(2*self.sigma**2))
        return (3/(4*np.pi*self.grain_density))*np.trapz(y,self.a)

    def n_density(self,mass_density,sizes):
        C = mass_density*self.sintegral
        dist = (C/sizes**4)*np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        dist[sizes<self.amin] = 0.0
        dist[sizes>self.amax] = 0.0
        return dist
    
    def averaged_over(self,X,sizes):
        y = (1.0/(sizes**4)) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        N = np.trapz(y,sizes)
        
        x = (X/(sizes**4)) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        x[sizes<self.amin] = 0.0
        x[sizes>self.amax] = 0.0
        
        avg = (1/N) * np.trapz(x,sizes)
        
        return avg
    
def Tielens_rate(fit,T):
    R = fit['a0'] + fit['a1']*T + fit['a2']*T**2 + fit['a3']*T**3 + fit['a4']*T**4 + fit['a5']*T**5
    R  = 10**R
    return R

def plot_dust_sputtering():
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,4), dpi=300, facecolor='w', edgecolor='k')
    
    T = np.logspace(4,9,1000)
    
    linestyles = ['-','--','-.',':']
    colours = ['b','r','m']
    grain_types = ['Sil','Car']
    
    for i,n_key in enumerate(thermal_spu.keys()):
        data = thermal_spu[n_key]
        J = Tielens_rate(data,np.log10(T/1e6))
        label = n_key+' (Tielens et al. 1994)'
        ax.plot(T,J,linestyle=linestyles[i],color=colours[i],label=label)
            
    # PAHs = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    # for t, t_key in enumerate(type_collision):
    #     J_avg = np.zeros(len(T))
    #     for i in range(0, len(T)):
    #         values = np.zeros(len(thermal_spu.keys()))
    #         sizes = np.zeros(len(thermal_spu.keys()))
    #         for j,n_key in enumerate(thermal_spu.keys()):
    #             Nc = int(n_key)
    #             sizes[j] = size_from_Nc(Nc)
    #             fits = thermal_spu[n_key][t_key]
    #             values[j] = Micellotta_rate(fits,np.log10(T[i]))
    #         J_avg[i] = PAHs.averaged_over(values,sizes*1e-4)
    #     ax.plot(T,J_avg,linestyle=linestyles[t],color='k')
    #     popt,pcov = curve_fit(polynomial_rate,np.log10(T),np.log10(J_avg))
    #     print(t_key,popt)
    #     print('Timescale [in s]: '+str(1/(10**polynomial_rate(np.log10(300),*popt)*0.1)))
            
            
    ax.set_ylabel(r'Sputtering rate $(1/n_{\rm H})da/dt$ [cm$^3\AA$yr$^{-1}$]', fontsize=13)
    ax.set_xlabel(r'$T$ [K]',fontsize=16)
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.legend(loc='best',fontsize=10,frameon=False)
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('dust_thermal_sputtering.png',format='png',dpi=300)
    plt.close(fig)
    