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
from unyt import mh,kb

sec2Myr = 3.1536e13

# Model parameters

basic_a0 = np.array([5e-4,1e-3,1e-2,1e-1,5e-4,1e-2,1e-1])
basic_amin = np.array([1e-4,1e-4,1e-4,4e-4,4e-4,4e-4])
basic_amax = np.array([3e-3,9e-3,1,1,2e-3,1,1])
basic_sigma = np.array([0.3,0.4,0.7,0.8,0.4,0.7,0.8])
basic_s = np.array([2,2,2.2,2.2,3.3,3.3,3.3])

# Tielens et al. (1994) - Thermal sputtering rates for silicate 
# and carbonaceous grains. See https://ui.adsabs.harvard.edu/abs/1994ApJ...431..321T/abstract

thermal_spu_tielens94 = {'Sil':{'a0':-2.7446,'a1':1.5439,'a2':-0.37046,'a3':0.21641,'a4':-0.34755,'a5':0.10114},
                        'Car':{'a0':-2.8605,'a1':1.0572,'a2':-0.27545,'a3':0.23735,'a4':-0.31820,'a5':0.087376}}

thermal_spu_nozawa06 = {'Sil':{'a0':-2.34790500e+02,'a1':1.33208637e+02,'a2':-3.13027448e+01,'a3':3.71345730,'a4':-2.21823668e-01,'a5':5.31746427e-03},
                        'Car':{'a0':-2.34333937e+02,'a1':1.38485732e+02,'a2':-3.39021615e+01,'a3':4.17705353,'a4':-2.58281473e-1,'a5':6.38827523e-03}}

class LogNormal_Distribution(object):

    def __init__(self,a0,amin,amax,sigma,grain_density):
        self.a0 = a0
        self.amin = amin
        self.amax = amax
        self.sigma = sigma
        self.a = np.logspace(np.log(amin),np.log10(amax),1000)
        self.grain_density = grain_density
        self.sintegral = self._init_integral()

    def _init_integral(self):
        y = (1.0/self.a) * np.exp(-(np.log(self.a/self.a0))**2/(2*self.sigma**2))
        return (3/(4*np.pi*self.grain_density))*np.trapz(y,self.a)

    def n_density(self,mass_density,sizes):
        C = mass_density*self.sintegral
        dist = (C/sizes**4)*np.exp(-(np.log(sizes/self.a0))**2/(2*self.sigma**2))
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
    def averaged_over_mass(self,X,sizes):
        y = (1.0/sizes) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        N = np.trapz(y,sizes)
        
        x = (X/sizes) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        x[sizes<self.amin] = 0.0
        x[sizes>self.amax] = 0.0
        
        avg = (1/N) * np.trapz(x,sizes)
        
        return avg
    def averaged_over_column(self,X,sizes):
        y = (1.0/(sizes**2)) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        N = np.trapz(y,sizes)
        
        x = (X/(sizes**2)) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
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
    sns.set_theme(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,4), dpi=300, facecolor='w', edgecolor='k')
    
    T = np.logspace(3,10,1000)
    
    linestyles = ['-','--','-.',':']
    colours = ['b','r','m','g']
    grain_types = ['Sil','Car']
    
    for i,n_key in enumerate(thermal_spu_tielens94.keys()):
        data = thermal_spu_tielens94[n_key]
        J = Tielens_rate(data,np.log10(T/1e6))
        label = n_key+' (Tielens et al. 1994)'
        ax.plot(T,J,linestyle=linestyles[i],color=colours[i],label=label)
    idiff = len(thermal_spu_tielens94.keys())
    for i,n_key in enumerate(thermal_spu_nozawa06.keys()):
        data = thermal_spu_nozawa06[n_key]
        J = Tielens_rate(data,np.log10(T*0.6))
        label = n_key+' (Nozawa et al. 2006)'
        ax.plot(T,J*1e4,linestyle=linestyles[i+idiff],color=colours[i+idiff],label=label)
        
        
            
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
    ax.set_ylim([1e-7,1])
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

def grain_charge_dist(Gtot,T,ne,grain_type,grain_radius,gamma=None):
    from scipy.stats import norm
    # This uses the fitting function from Ibanez-Mejias et al. (2019)
    # (https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.1220I/abstract)
    # which are detailed in the Eq. 17-19.
    # We assume a discrete Gaussian distribution correcting for
    # integer dust charges
    
    # Fitting parameters form their Table 1
    fit_params = {'silicates':{'3.5A':{'alpha':0.3263,'k':0.0149,'b':-0.1212,'hz':57,
                                        'c+':0.4123,'eta+':0.2513,'d':0.1891,'c-':0.4845,
                                        'eta-':0.3532},
                                '5A':{'alpha':0.3141,'k':0.0372,'b':-0.3043,'hz':86,
                                        'c+':0.2734,'eta+':0.2925,'d':0.3233,'c-':0.3615,
                                        'eta-':0.6532},
                                '10A':{'alpha':0.3535,'k':0.0494,'b':-0.4865,'hz':73,
                                        'c+':0.4353,'eta+':0.7459,'d':0.4451,'c-':0.1053,
                                        'eta-':0.5803},
                                '50A':{'alpha':0.5115,'k':0.0717,'b':-0.4106,'hz':107,
                                        'c+':1.0758,'eta+':1.7832,'d':0.5860,'c-':-1.0379e3,
                                        'eta-':7.7069e3},
                                '100A':{'alpha':0.3525,'k':0.6591,'b':-0.1649,'hz':384,
                                        'c+':1.6245,'eta+':2.8390,'d':0.6346,'c-':-4.2075e2,
                                        'eta-':1.9840e3},
                                '500A':{'alpha':0.3643,'k':2.6283,'b':0.5217,'hz':345,
                                        'c+':4.0732,'eta+':11.0200,'d':0.6797,'c-':-0.2418,
                                        'eta-':0.5910},
                                '1000A':{'alpha':0.3927,'k':3.6493,'b':0.8389,'hz':372,
                                        'c+':5.9813,'eta+':20.6410,'d':0.6961,'c-':-0.1885,
                                        'eta-':0.4237}},

                'carbonaceous':{'3.5A':{'alpha':0.4699,'k':0.0085,'b':-0.1162,'hz':48,
                                        'c+':0.3103,'eta+':0.2744,'d':0.2551,'c-':0.3766,
                                        'eta-':0.5241},
                                '5A':{'alpha':0.4386,'k':0.0195,'b':-0.3084,'hz':95,
                                        'c+':0.3699,'eta+':0.5654,'d':0.4158,'c-':0.2890,
                                        'eta-':1.6241},
                                '10A':{'alpha':0.4994,'k':0.0199,'b':-0.4959,'hz':78,
                                        'c+':0.6511,'eta+':0.9839,'d':0.5275,'c-':0.0213,
                                        'eta-':0.0977},
                                '50A':{'alpha':0.6009,'k':0.0523,'b':-0.4092,'hz':218,
                                        'c+':1.6536,'eta+':2.6688,'d':0.6671,'c-':-9.5138,
                                        'eta-':35.3519},
                                '100A':{'alpha':0.2900,'k':2.2310,'b':-0.2061,'hz':1063,
                                        'c+':2.5445,'eta+':4.3352,'d':0.7010,'c-':-2.5341e3,
                                        'eta-':8.1962e3},
                                '500A':{'alpha':0.3400,'k':5.8944,'b':0.1727,'hz':1034,
                                        'c+':5.9455,'eta+':18.3186,'d':0.8377,'c-':-2.4189e3,
                                        'eta-':4.9424e3},
                                '1000A':{'alpha':0.3500,'k':9.6536,'b':0.4183,'hz':1273,
                                        'c+':8.7003,'eta+':36.1014,'d':0.9094,'c-':-2.6009e3,
                                        'eta-':4.7029e3}}}

    sfit = fit_params[grain_type][grain_radius]
    if gamma != None:
        charPar = gamma
    else:
        charPar = Gtot *np.sqrt(T) / ne
    Z = sfit['k'] * (1.0 - np.exp(-charPar/sfit['hz'])) * (charPar**sfit['alpha']) + sfit['b']
    if Z>0:
        sigma = sfit['c+'] * (1.0 - np.exp(-Z/sfit['eta+'])) + sfit['d']
    else:
        sigma = sfit['c-'] * (1.0 - np.exp(-abs(Z)/sfit['eta-'])) + sfit['d']
        
    Zmin = round(Z - 3.*sigma)
    Zmax = round(Z + 3.*sigma)
    x = np.arange(Zmin,Zmax+1)
    dist = np.zeros(len(x))
    for i in range(0,len(x)):
        dist[i] = (1. / (sigma * np.sqrt(2.*np.pi))) * np.exp(-0.5*((float(x[i]) - Z) / sigma)**2.)
    dist = dist / np.sum(dist)
    return dist,x

def grain_mean_charge(Gtot,T,ne,grain_type,grain_radius,gamma=None):
    from scipy.stats import norm
    # This uses the fitting function from Ibanez-Mejias et al. (2019)
    # (https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.1220I/abstract)
    # which are detailed in the Eq. 17-19.
    
    # Fitting parameters form their Table 1
    fit_params = {'silicates':{'3.5A':{'alpha':0.3263,'k':0.0149,'b':-0.1212,'hz':57,
                                        'c+':0.4123,'eta+':0.2513,'d':0.1891,'c-':0.4845,
                                        'eta-':0.3532},
                                '5A':{'alpha':0.3141,'k':0.0372,'b':-0.3043,'hz':86,
                                        'c+':0.2734,'eta+':0.2925,'d':0.3233,'c-':0.3615,
                                        'eta-':0.6532},
                                '10A':{'alpha':0.3535,'k':0.0494,'b':-0.4865,'hz':73,
                                        'c+':0.4353,'eta+':0.7459,'d':0.4451,'c-':0.1053,
                                        'eta-':0.5803},
                                '50A':{'alpha':0.5115,'k':0.0717,'b':-0.4106,'hz':107,
                                        'c+':1.0758,'eta+':1.7832,'d':0.5860,'c-':-1.0379e3,
                                        'eta-':7.7069e3},
                                '100A':{'alpha':0.3525,'k':0.6591,'b':-0.1649,'hz':384,
                                        'c+':1.6245,'eta+':2.8390,'d':0.6346,'c-':-4.2075e2,
                                        'eta-':1.9840e3},
                                '500A':{'alpha':0.3643,'k':2.6283,'b':0.5217,'hz':345,
                                        'c+':4.0732,'eta+':11.0200,'d':0.6797,'c-':-0.2418,
                                        'eta-':0.5910},
                                '1000A':{'alpha':0.3927,'k':3.6493,'b':0.8389,'hz':372,
                                        'c+':5.9813,'eta+':20.6410,'d':0.6961,'c-':-0.1885,
                                        'eta-':0.4237}},

                'carbonaceous':{'3.5A':{'alpha':0.4699,'k':0.0085,'b':-0.1162,'hz':48,
                                        'c+':0.3103,'eta+':0.2744,'d':0.2551,'c-':0.3766,
                                        'eta-':0.5241},
                                '5A':{'alpha':0.4386,'k':0.0195,'b':-0.3084,'hz':95,
                                        'c+':0.3699,'eta+':0.5654,'d':0.4158,'c-':0.2890,
                                        'eta-':1.6241},
                                '10A':{'alpha':0.4994,'k':0.0199,'b':-0.4959,'hz':78,
                                        'c+':0.6511,'eta+':0.9839,'d':0.5275,'c-':0.0213,
                                        'eta-':0.0977},
                                '50A':{'alpha':0.6009,'k':0.0523,'b':-0.4092,'hz':218,
                                        'c+':1.6536,'eta+':2.6688,'d':0.6671,'c-':-9.5138,
                                        'eta-':35.3519},
                                '100A':{'alpha':0.2900,'k':2.2310,'b':-0.2061,'hz':1063,
                                        'c+':2.5445,'eta+':4.3352,'d':0.7010,'c-':-2.5341e3,
                                        'eta-':8.1962e3},
                                '500A':{'alpha':0.3400,'k':5.8944,'b':0.1727,'hz':1034,
                                        'c+':5.9455,'eta+':18.3186,'d':0.8377,'c-':-2.4189e3,
                                        'eta-':4.9424e3},
                                '1000A':{'alpha':0.3500,'k':9.6536,'b':0.4183,'hz':1273,
                                        'c+':8.7003,'eta+':36.1014,'d':0.9094,'c-':-2.6009e3,
                                        'eta-':4.7029e3}}}

    sfit = fit_params[grain_type][grain_radius]
    if gamma != None:
        charPar = gamma
    else:
        charPar = Gtot *np.sqrt(T) / ne
    Z = sfit['k'] * (1.0 - np.exp(-charPar/sfit['hz'])) * (charPar**sfit['alpha']) + sfit['b']
    return Z

def cmp_D_WD99(charge_dist,x,Zi,T,a):
    # This is based on Eq. 6-7 in Weingartner & Draine (1999) which allows
    # the computation of the Coulomb enhancement factor from the charge
    # distribution
    # (https://iopscience.iop.org/article/10.1086/307197)

    e = 4.8032047e-10 # statC
    kB = 1.380649e-16   # erg/K
    D = 0.0
    if Zi != 0:
        for i in range(0, len(charge_dist)):
            Zg = x[i]
            if Zg*Zi>0:
                B = np.exp(-Zg*Zi*e**2/(kB*T*a))
            elif Zg*Zi<0:
                B = 1.0 - Zg*Zi*e**2/(kB*T*a)
            elif Zg==0:
                B = 1.0 + np.sqrt(np.pi*Zi**2*e**2/(2.0*kB*T*a))
            D = D + charge_dist[i] * B
    else:
        D = 1.0
    D = max(D,1e-10)
    return D
    
def plot_coulomb_enhancement(Gtot,Zi):
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,6), dpi=300, facecolor='w', edgecolor='k')
    
    # Coulomb enhancement factor from Weingartner & Draine (1999)
    # This is given for graphitic and silicate grains in 
    # CNM: nH=30 Hcc, T=100K, xe=0.0015
    # WNM: nH=0.4 Hcc, T=6000K, xe=0.1
    # WIM: nH=0.1 Hcc, T=8000K, xe=0.99
    names = ['CNM,gra_x','CNM,gra_y','CNM,sil_x','CNM,sil_y','WNM,gra_x','WNM,gra_y',
            'WNM,sil_x','WNM,sil_y','WIM,gra_x','WIM,gra_y','WIM,sil_x','WIM,sil_y']
    D = pd.read_csv('weingartner_draine_1999_coulomb_enhancement.csv',header=1,names=names)
    
    linestyles = ['-','--','-.',':',(0, (1, 10)),(0, (3, 5, 1, 5))]
    colours = ['b','r','m','g']

    for i in range(0, int(len(names)/2)):
        x = np.asarray(D[names[i*2]])
        y = np.asarray(D[names[i*2+1]])
        label = names[i*2].split('_')[0]
        ax.plot(x,y,label=label,linestyle=linestyles[i],color='k')

    IM19_sizes = ['3.5A','5A','10A','50A','100A','500A','1000A']
    sizes_incm = [3.5e-8,5e-8,1e-7,5e-7,1e-6,5e-6,1e-5]
    
    # CNM: nH=30 Hcc, T=100K, xe=0.0015
    D_CNM = np.zeros(len(IM19_sizes))
    for i in range(0,len(IM19_sizes)):
        #ch_dist,x = grain_charge_dist(Gtot,100,30*0.0015,'silicates',IM19_sizes[i])
        ch_dist,x = grain_charge_dist(Gtot,94.9113984403983,0.194638139893389,'silicates',IM19_sizes[i])
        a = sizes_incm[i]
        D = cmp_D_WD99(ch_dist,x,Zi,100,a)
        D_CNM[i] = D
    ax.plot(sizes_incm,D_CNM,color='b',linestyle='-')
    f = interp1d(np.log10(sizes_incm),np.log10(D_CNM), fill_value='extrapolate',kind='linear')
    xnew = np.logspace(np.log10(3.5e-8),np.log10(2.5e-5),100)
    ax.plot(xnew,10**f(np.log10(xnew)),color='b',linestyle='-',alpha=0.6)
    D_CNM = np.zeros(len(IM19_sizes))
    for i in range(0,len(IM19_sizes)):
        # ch_dist,x = grain_charge_dist(Gtot,100,30*0.0015,'carbonaceous',IM19_sizes[i])
        ch_dist,x = grain_charge_dist(Gtot,99.7592216672557,0.194638139893389,'carbonaceous',IM19_sizes[i])
        a = sizes_incm[i]
        D = cmp_D_WD99(ch_dist,x,Zi,100,a)
        D_CNM[i] = D
    ax.plot(sizes_incm,D_CNM,color='r',linestyle='-')
    f = interp1d(np.log10(sizes_incm),np.log10(D_CNM), fill_value='extrapolate',kind='linear')
    xnew = np.logspace(np.log10(3.5e-8),np.log10(2.5e-5),100)
    ax.plot(xnew,10**f(np.log10(xnew)),color='r',linestyle='-',alpha=0.6)
    
    # WNM: nH=0.4 Hcc, T=6000K, xe=0.1
    D_WNM = np.zeros(len(IM19_sizes))
    for i in range(0,len(IM19_sizes)):
        ch_dist,x = grain_charge_dist(Gtot,6000,0.4*0.1,'silicates',IM19_sizes[i])
        a = sizes_incm[i]
        D = cmp_D_WD99(ch_dist,x,Zi,6000,a)
        D_WNM[i] = D
    ax.plot(sizes_incm,D_WNM,color='b',linestyle='--')
    f = interp1d(np.log10(sizes_incm),np.log10(D_WNM), fill_value='extrapolate',kind='linear')
    xnew = np.logspace(np.log10(3.5e-8),np.log10(2.5e-5),100)
    ax.plot(xnew,10**f(np.log10(xnew)),color='b',linestyle='--',alpha=0.6)
    D_WNM = np.zeros(len(IM19_sizes))
    for i in range(0,len(IM19_sizes)):
        ch_dist,x = grain_charge_dist(Gtot,6000,0.4*0.1,'carbonaceous',IM19_sizes[i])
        a = sizes_incm[i]
        D = cmp_D_WD99(ch_dist,x,Zi,6000,a)
        D_WNM[i] = D
    ax.plot(sizes_incm,D_WNM,color='r',linestyle='--')
    f = interp1d(np.log10(sizes_incm),np.log10(D_WNM), fill_value='extrapolate',kind='linear')
    xnew = np.logspace(np.log10(3.5e-8),np.log10(2.5e-5),100)
    ax.plot(xnew,10**f(np.log10(xnew)),color='r',linestyle='--',alpha=0.6)
    
    # WIM: nH=0.1 Hcc, T=8000K, xe=0.99
    D_WIM = np.zeros(len(IM19_sizes))
    for i in range(0,len(IM19_sizes)):
        ch_dist,x = grain_charge_dist(Gtot,8000,0.1*0.99,'silicates',IM19_sizes[i])
        a = sizes_incm[i]
        D = cmp_D_WD99(ch_dist,x,Zi,8000,a)
        D_WIM[i] = D
    ax.plot(sizes_incm,D_WIM,color='b',linestyle=':')
    f = interp1d(np.log10(sizes_incm),np.log10(D_WIM), fill_value='extrapolate',kind='linear')
    xnew = np.logspace(np.log10(3.5e-8),np.log10(2.5e-5),100)
    ax.plot(xnew,10**f(np.log10(xnew)),color='b',linestyle=':',alpha=0.6)
    D_WIM = np.zeros(len(IM19_sizes))
    for i in range(0,len(IM19_sizes)):
        ch_dist,x = grain_charge_dist(Gtot,8000,0.1*0.99,'carbonaceous',IM19_sizes[i])
        a = sizes_incm[i]
        D = cmp_D_WD99(ch_dist,x,Zi,8000,a)
        D_WIM[i] = D
    ax.plot(sizes_incm,D_WIM,color='r',linestyle=':')
    f = interp1d(np.log10(sizes_incm),np.log10(D_WIM), fill_value='extrapolate',kind='linear')
    xnew = np.logspace(np.log10(3.5e-8),np.log10(2.5e-5),100)
    ax.plot(xnew,10**f(np.log10(xnew)),color='r',linestyle=':',alpha=0.6)
    
    init_legend = ax.legend(loc='upper right',fontsize=10,frameon=False,ncol=2)
    ax.add_artist(init_legend)
    
    dummy_lines = [ax.plot([],[],color='b',linestyle='-',label='Silicates')[0],
                   ax.plot([],[],color='r',linestyle='-',label='Carbonaceous')[0]]
    first_legend = ax.legend(handles=dummy_lines, loc='lower left', frameon=False, fontsize=10)
    ax.add_artist(first_legend)
    dummy_lines = [ax.plot([],[],color='k',linestyle='-',label='CNM')[0],
                   ax.plot([],[],color='k',linestyle='--',label='WNM')[0],
                   ax.plot([],[],color='k',linestyle=':',label='WIM')[0]]
    second_legends = ax.legend(handles=dummy_lines, loc='lower right', frameon=False, fontsize=10)
    ax.add_artist(second_legends)


    ax.set_ylabel(r'Coulomb enhancement $D(a)$', fontsize=13)
    ax.set_xlabel(r'$a$ [cm]',fontsize=16)
    ax.set_ylim([1e-3,300])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('dust_coulomb_enhancement.png',format='png',dpi=300)
    plt.close(fig)


def disruption_size(U,gamma,lmean,Smax,asize,nH,Tgas):
    # When the grain rotation induced by radiative torques is sufficiently high that
    # the tensile stress is larger than the tensile strength, the grain is
    # disrupted into smaller fragments
    
    # For an arbitrary radiation field and the grain size being below the transition
    # size between power-law to constant behaviour with size of the radiative torque,
    # the minimum size of the grain for disruption is given by Eq. 45 in Hoang (2020)
    # (https://ui.adsabs.harvard.edu/abs/2020Galax...8...52H/abstract)
    
    # IR photons emitted by grain carry away part of the grain's angular momentum,
    # resulting in the damping of the grain rotation. The timescale of IR damping is
    # proportional to the timescale of damping with gas species.
    # Given by Draine & Lazarian (1998)
    # (https://ui.adsabs.harvard.edu/abs/1998ApJ...508..157D/abstract)
    
    FIR = (0.4*U**(2/3)/(asize/1e-5))*(30/nH)*np.sqrt(100/Tgas)

    #y = 5.1 * (0.1/gamma) * (U**(-1/3)) * (lmean / (5e-5))**1.7 * np.sqrt(Smax/1e7)
    #a_disr = 1e-5 * (y**(1/2.7))
    nH = nH / 10
    Tgas = Tgas / 100
    y = 3.8 * (0.1/gamma) * (lmean / (5e-5))**1.7 * np.sqrt(Smax/1e7) * (1+FIR) * (nH*np.sqrt(Tgas)) / U
    
    a_disr = 1e-5 * (y**(1/1.7))

    return a_disr

def disruption_timescale(U,gamma,lmean,Smax,sx,a):
    
    a_trans = (lmean / 1.8)
    if a<=a_trans:
        t_disr = 1e5 / (gamma*U) * (lmean / (0.5e-6))**1.7 * np.sqrt(sx / 3.0) * np.sqrt(Smax/1e7) * (a/1e-5)**(-0.7)
    else:
        t_disr = 7.4 / (gamma*U) * (0.5e-6 / lmean) * np.sqrt(sx / 3.0) * np.sqrt(Smax/1e7) * (a/1e-5)**2
        
    return t_disr/1e6
def plot_ratd_timescale(gamma,lmean,Smax,nH,Tgas):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,6), dpi=300, facecolor='w', edgecolor='k')
    
    U = np.logspace(-8,7,100)
    
    t_disr = {'silicates':{'small':np.zeros(len(U)),'large':np.zeros(len(U))},
              'carbonaceous':{'small':np.zeros(len(U)),'large':np.zeros(len(U))}}
    
    for i in range(0, len(U)):
        a_disr = disruption_size(U[i],gamma,lmean,Smax,basic_a0[1]*1e-4,nH,Tgas)
        # Do small grains
        if basic_a0[1]*1e-4>= a_disr:
            
            t_disr['silicates']['small'][i] = disruption_timescale(U[i],gamma,lmean,Smax,basic_s[2],basic_a0[1]*1e-4)
            t_disr['carbonaceous']['small'][i] = disruption_timescale(U[i],gamma,lmean,Smax,basic_s[1],basic_a0[1]*1e-4)
        else:
            t_disr['silicates']['small'][i] = 1e6
            t_disr['carbonaceous']['small'][i] = 1e6
        a_disr = disruption_size(U[i],gamma,lmean,Smax,basic_a0[2]*1e-4,nH,Tgas)
        # Now do large grains
        if basic_a0[2]*1e-4>= a_disr:
            t_disr['silicates']['large'][i] = disruption_timescale(U[i],gamma,lmean,Smax,basic_s[2],basic_a0[2]*1e-4)
            t_disr['carbonaceous']['large'][i] = disruption_timescale(U[i],gamma,lmean,Smax,basic_s[1],basic_a0[2]*1e-4)
        else:
            t_disr['silicates']['large'][i] = 1e6
            t_disr['carbonaceous']['large'][i] = 1e6
    
    ax.plot(U,t_disr['silicates']['small'],linestyle='-',color='b')
    ax.plot(U,t_disr['silicates']['large'],linestyle='--',color='b',label='Large SilDust')
    ax.plot(U,t_disr['carbonaceous']['small'],linestyle='-',color='r')
    ax.plot(U,t_disr['carbonaceous']['large'],linestyle='--',color='r',label='Large CDust')
    
    ax.set_ylabel(r'$t_{\rm RATD}$ [Myr]', fontsize=13)
    ax.set_xlabel(r'$U$',fontsize=16)
    #ax.set_ylim([1e-4,1e2])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.legend(loc='best',fontsize=12,frameon=False)
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('dust_ratd_timescales.png',format='png',dpi=300)
    plt.close(fig)

def t_shattering(Dbig,nH,a,s):
    
    local_mu = 1.4
    
    t_sha = 75.73 * (0.01/Dbig) * (a/1.e-5) * (s/3.)
    if nH < 1:
        t_sha = t_sha * (1./(nH*local_mu))
    elif 1 <= nH <= 1e3:
        t_sha = t_sha * (1./nH)**(1./3.) / local_mu
    else:
        t_sha = 1e9
    return t_sha

def t_coagulation(Dsmall,Mach,nH,T,L,a,s,boost=True):
    
    from scipy.special import erfc
    
    lambda_jeans = 3.8409904e7 * np.sqrt(T/(nH*mh.to('g').d))
    nhmax_coa = 1e20
    sigs = np.log(1.+(0.4*Mach)**2.)
    sigs2 = sigs**2.
    smax = np.log(nhmax_coa/nH)
    if boost:
        boost_coa = 0.5*np.exp(sigs2)*erfc((1.5*sigs2-smax)/(np.sqrt(2.)*sigs))
    else:
        boost_coa = 1.
    L = L * 3.0857e18 # [cm]
    local_mu = 1.4
    
    if T>1e4 or nH<1e2 or lambda_jeans>4*L:
        t_coa = 1e5
    else:
        t_coa = 0.00094663 * (a/5e-7) * (s/3.) * (1e3/(nH*boost_coa)) / Dsmall / local_mu
    
    return t_coa

def relative_velocity(model,T,nH,ne,Mach,L,target_a,projectile_a,target_s,projectile_s):
    
    local_mu = nH / (nH + ne)
    gamma_gas =5./3.
    Lmax = L * 3.0857e18 # [cm] 1 pc
    e = 4.8032047e-10 # statC
    
    if model == 'Ormel and Cuzzi2007':
        def OC07_function(x):
            f = 3.2 - (1.+x) + 2./(1.+x)*(1./2.6+x**3./(1.6+x))
            return f
        # This is based on the formulation presented in Kawasaki & Machida (2023)
        # which is basically the analytical model of Ormel & Cuzzi (2007)
        mass_target = 4./3. * np.pi * target_s * (target_a**3.)
        mass_projectile = 4./3. * np.pi * projectile_s * (projectile_a**3.)
        dV_thermal = np.sqrt(8. * kb.to('cm**2*g/s**2/K').d * T * (mass_projectile+mass_target)/(mass_target*mass_projectile))
        rho_gas = nH * mh.to('g').d * local_mu
        cs_gas = np.sqrt(gamma_gas * kb.to('cm**2*g/s**2/K').d * T / (mh.to('g').d*local_mu))
        v_th = np.sqrt(8/np.pi) * cs_gas
        v_turb = Mach * cs_gas
        
        # Assume that the injection scale is a cell size of Lmax pc and the velocity is given by
        # the largest size eddie velocity
        
        tau_L = Lmax / v_turb
        # Assume the closure equations by Braginskii (1965), based on the Chapman-Enskog scheme
        # which is based in the assumption that the macroscopic scale of the plasma is large
        # compared to the mean free path or the gyro-radii of the electrons and the ions. In this
        # case, the viscosity is dominated by the hydrogen viscosity parallel to the magnetic field
        # (Braginskii 1965). This is because the ions carry the majority of the momemtum
        rc = e**2 / (kb.to('cm**2*g/s**2/K').d * T)
        mfp = 1. / (nH*rc**2.)
        nu = cs_gas * mfp / 3.
        Re = v_turb * Lmax / 8.7e15 #nu
        
        tau_eta = tau_L / np.sqrt(Re)
        
        # Stopping time computation for target and projectile
        # Epstein's law
        ts_target = target_s * target_a / (rho_gas*v_th)            
        ts_projectile = projectile_s * projectile_a / (rho_gas*v_th)        

        # Compute the Stokes' numbers for both particles
        St_target = ts_target / tau_L
        St_projectile = ts_projectile / tau_L
        
        # And finally compute the relative velocity between the particles
        Stmin = tau_eta / tau_L
        if ts_target < tau_eta:
            dV_turb = np.sqrt(3./2.) * v_turb * np.sqrt((St_target-St_projectile)/(St_target+St_projectile)) * \
                np.sqrt((St_target**2./(St_target+Stmin)) - (St_projectile**2./(St_projectile+Stmin)))
        elif tau_eta <= ts_target < tau_L:
            dV_turb = np.sqrt(3./2.) * v_turb * np.sqrt(OC07_function(St_projectile/St_target)*St_target)
        elif ts_target >= tau_L:
            dV_turb = np.sqrt(3./2.) * v_turb * np.sqrt(1./(1.+St_target)+1./(1.+St_projectile))
        v_rel = np.sqrt(dV_thermal**2. + dV_turb**2.)
        
    elif model == 'Hirashita and Aoyama2019':
        v_target = 1.1e5 * (Mach**(3./2.)) * np.sqrt(target_a/1e-5) * ((T/1e4)**(1./4.)) * (nH**(-1./4.)) * np.sqrt(target_s/3.5)
        v_projectile = 1.1e5 * (Mach**(3./2.)) * np.sqrt(projectile_a/1e-5) * ((T/1e4)**(1./4.)) * (nH**(-1./4.)) * np.sqrt(projectile_s/3.5)
        v_rel = 0.5 * (v_target + v_projectile)
    return v_rel
            

def plot_relative_velocity(target_a,projectile_a,target_s,projectile_s,composition,nH,ne,T,nMach=100):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    
    Mach = np.logspace(-1,1,nMach)
    
    Lmax = 10 # [pc]
    
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,6), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$v_{\rm rel}(a,\mathcal{M})$ [km/s]', fontsize=16)
    ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    # Compute velocities for all models
    models = ['Hirashita and Aoyama2019','Ormel and Cuzzi2007']
    for i in range(0, len(models)):
        v_rel = np.zeros(nMach)
        for j in range(0,nMach):
            v_rel[j] = relative_velocity(models[i],T,nH,ne,Mach[j],Lmax,target_a,projectile_a,
                                    target_s,projectile_s)
        ax.plot(Mach,v_rel/1e5,label=models[i])
    

    ax.legend(loc='best',fontsize=10,frameon=False)
    
    ax.text(0.7, 0.2, r'$T_{\rm gas}=%.1f$'%T+'\n'+r'$n_{\rm H}=%.3f$'%nH+'\n'+r'$n_{\rm e}=%.3f$'%ne,
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=ax.transAxes,fontsize=13)
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('relative_velocity_%s_%s.png'%(str(target_a/1e-4),composition),format='png',dpi=300)
    plt.close(fig)
        
    
    
def plot_shattering_frag(target_a,projectile_a,target_s,projectile_s,composition,nH,ne,T,nprojectile,nMach=100):
    # This function plots the shattering fragment distribution for big grains
    # based on the power-law model 
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })

    from utils import as_si
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,6), dpi=300, facecolor='w', edgecolor='k')
    
    a = np.logspace(-4,1,1000)
    n = a**(-3.3)
    
    ax.plot(a,n,'k-')
    
    ax.axvspan(1e-4, 1e-3, alpha=0.5, color='blue')
    ax.axvspan(1e-3, 2e-2, alpha=0.5, color='green')
    ax.axvspan(2e-2, 0.3, alpha=0.5, color='red')
    
    ax.set_ylabel(r'$n_{\rm frag}(a)$', fontsize=13)
    ax.set_xlabel(r'$a$ [$\mu$m]',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('big_shattering_dist.png',format='png',dpi=300)
    plt.close(fig)
    
    # Shattering model quantities (Kobayashi & Tanaka 2010)
    Q_D = [8.9e9,4.3e10]
    alpha_f = 3.3
    
    if composition == 'Gra':
        index = 0
    elif composition == 'Sil':
        index = 1
    
    # Grain quantities
    mass_target = 4./3. * np.pi * target_s * (target_a**3.)
    mass_projectile = 4./3. * np.pi * projectile_s * (projectile_a**3.)
    rho_projectile = nprojectile * mass_projectile
    
    Mach = np.logspace(-1,1,nMach)
    Lmax = 10 # [pc]
    
    # Limits of the distributions
    m_dest_min = 4./3. * np.pi * target_s * ((1e-8)**3.)
    m_vsmall_min = 4./3. * np.pi * target_s * ((1e-8)**3.)
    m_vsmall_max = 4./3. * np.pi * target_s * ((1e-7)**3.)
    m_small_min = 4./3. * np.pi * target_s * ((1e-7)**3.)
    m_small_max = 4./3. * np.pi * target_s * ((2e-6)**3.)
    m_big_min = 4./3. * np.pi * target_s * ((2e-6)**3.)
    m_big_max = 4./3. * np.pi * target_s * ((3e-5)**3.)
    
    exponent1 = (4.-alpha_f)/3.
    exponent2 = 1. + (-alpha_f+1.)/3.
    
    
    M_dest = np.zeros((nMach,3))
    M_vsmall = np.zeros((nMach,3))
    M_small = np.zeros((nMach,3))
    M_big = np.zeros((nMach,3))
    
    t_dest = np.zeros((nMach,3))
    t_vsmall = np.zeros((nMach,3))
    t_small = np.zeros((nMach,3))
    t_big = np.zeros((nMach,3))
    
    for i in range(0, nMach):
        
        # Relative velocity between two grains of the same size (Eq. 18 of Hirashita & Aoyama 2019)
        v_target = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(target_a/1e-5) * ((T/1e4)**(1./4.)) * (nH**(-1./4.)) * np.sqrt(target_s/3.5)
        v_projectile = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(projectile_a/1e-5) * ((T/1e4)**(1./4.)) * (nH**(-1./4.)) * np.sqrt(projectile_s/3.5)
        v_rel_min = min(abs(v_target-v_projectile),v_target,v_projectile)
        v_rel_max = v_target+v_projectile
        
        
        # Average collision velocity as given by the Ormel & Cuzzy (2007) fitting functions
        v_rel_avg = relative_velocity('Ormel and Cuzzi2007',T,nH,ne,Mach[i],Lmax,target_a,projectile_a,
                                      target_s,projectile_s)
        v_rel = np.array([max(v_rel_min,v_projectile),v_rel_max,v_rel_avg])
        
        # Disrupted mass computation (Eqs. 20-22 of Hirashita & Aoyama 2019)
        E_imp = 0.5 * (mass_projectile*mass_target)/(mass_target+mass_projectile) * v_rel**2.
        phi = E_imp / (mass_target*Q_D[index])
        m_ej = phi / (1.+phi) * mass_target
        
        # Now compute the maximum and minimum masses of the fragments
        m_remnant = mass_target - m_ej
        m_max = 0.02*m_ej
        m_min = 1e-6*m_max
        
        # Compute the mass fractions for each size bin
        prefactor = (4.-alpha_f)/(3.*(m_max**exponent1-m_min**exponent1)) / exponent2 * m_ej
        for j in range(0,3):
            # 1. Destruction to the gas phase (1e-4 mum)
            if m_min[j] >= m_dest_min:
                M_dest[i,j] = 0.0
            else:
                M_dest[i,j] = prefactor[j] * (min(m_dest_min,m_max[j])**exponent2-m_min[j]**exponent2)
            
            # 2. Destruction to very small grains
           
            if m_min[j] >= m_vsmall_max:
                M_vsmall[i,j] =  0.0
            else:
                M_vsmall[i,j] = prefactor[j] * (min(m_vsmall_max,m_max[j])**exponent2-max(m_vsmall_min,m_min[j])**exponent2)
                if m_vsmall_min <= m_remnant[j] < m_vsmall_max:
                    M_vsmall[i,j] += m_remnant[j]
            
            # 3. Destruction to small grains
            if m_min[j] >= m_small_max:
                M_small[i,j] = 0.
            else:
                M_small[i,j] = prefactor[j] * (min(m_small_max,m_max[j])**exponent2-max(m_small_min,m_min[j])**exponent2)
                if m_small_min <= m_remnant[j] < m_small_max:
                    M_small[i,j] += m_remnant[j]

            # 4. Destruction to big grains
            if m_min[j] >= m_big_max:
                M_big[i,j] =  0.0
            else:
                M_big[i,j] = prefactor[j] * (min(m_big_max,m_max[j])**exponent2-max(m_big_min,m_min[j])**exponent2)
                if m_big_min <= m_remnant[j] < m_big_max:
                    M_big[i,j] += m_remnant[j]
            
        # 5. Put remnant fragment to its correct bin
        M_tot = (M_dest[i] + M_vsmall[i] + M_small[i] + M_big[i])
        # print('Total final mass vs initial mass: '+str(M_tot/mass_target))
        M_dest[i] = M_dest[i]/M_tot
        M_vsmall[i] = M_vsmall[i]/M_tot
        M_small[i] = M_small[i]/M_tot
        M_big[i] = M_big[i]/M_tot
        
        alpha = np.pi * (target_a + projectile_a)**2. * v_rel / (mass_target*mass_projectile)
        t_dest[i] = 1./(alpha*rho_projectile*mass_target*M_dest[i])/sec2Myr
        t_vsmall[i] = 1./(alpha*rho_projectile*mass_target*M_vsmall[i])/sec2Myr
        t_small[i] = 1./(alpha*rho_projectile*mass_target*M_small[i])/sec2Myr
        t_big[i] = 1./(alpha*rho_projectile*mass_target*M_big[i])/sec2Myr

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,6), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$\chi_{\rm frag}(a,\mathcal{M})$', fontsize=16)
    ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    selected = (M_dest[:,0]>0.) & (M_dest[:,1]>0.)
    ax.fill_between(Mach,M_dest[:,0],M_dest[:,1],label='Destroyed',alpha=0.5,where=(selected))
    ax.fill_between(Mach,M_vsmall[:,0],M_vsmall[:,1],label='Very small grains',alpha=0.5,where=M_vsmall[:,1]>0.)
    ax.fill_between(Mach,M_small[:,0],M_small[:,1],label='Small grains',alpha=0.5,where=M_small[:,1]>0.)
    ax.fill_between(Mach,M_big[:,0],M_big[:,1],label='Large grains',alpha=0.5,where=M_big[:,1]>0.)
    ax.plot(Mach,M_dest[:,2],linestyle='-.')
    ax.plot(Mach,M_vsmall[:,2],linestyle='-.')
    ax.plot(Mach,M_small[:,2],linestyle='-.')
    ax.plot(Mach,M_big[:,2],linestyle='-.')
    ax.legend(loc='best',fontsize=10,frameon=False)
    
    ax.text(0.6, 0.2, r'$T_{\rm gas}$'+r'$={0:s}$'.format(as_si(T,1))+
                        ' K\n'+r'$n_{\rm H}$'+r'$={0:s}$'.format(as_si(nH,1)),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=ax.transAxes,fontsize=13)
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('shattering_efficiency_%s.png'%(composition),format='png',dpi=300)
    plt.close(fig)
    
    
    # Timescale plot
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,6), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$t_{\rm sha}(a,\mathcal{M})$ [Myr]', fontsize=16)
    ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)
    ax.tick_params(labelsize=12)
    #ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    
    ax.fill_between(Mach,t_dest[:,0],t_dest[:,1],label='Destroyed',alpha=0.5,where=M_dest[:,1]>0.)
    ax.fill_between(Mach,t_vsmall[:,0],t_vsmall[:,1],label='Very small grains',alpha=0.5,where=M_vsmall[:,1]>0.)
    ax.fill_between(Mach,t_small[:,0],t_small[:,1],label='Small grains',alpha=0.5,where=M_small[:,1]>0.)
    ax.fill_between(Mach,t_big[:,0],t_big[:,1],label='Large grains',alpha=0.5,where=M_big[:,1]>0.)
    ax.plot(Mach,t_dest[:,2],linestyle='-.')
    ax.plot(Mach,t_vsmall[:,2],linestyle='-.')
    ax.plot(Mach,t_small[:,2],linestyle='-.')
    ax.plot(Mach,t_big[:,2],linestyle='-.')
    ax.legend(loc='best',fontsize=10,frameon=False)
    ax.legend(loc='best',fontsize=10,frameon=False)
    
    ax.text(0.6, 0.6, r'$T_{\rm gas}$'+r'$={0:s}$'.format(as_si(T,1))+
                        ' K\n'+r'$n_{\rm H}$'+r'$={0:s}$'.format(as_si(nH,1))+
                        r' cm$^{-3}$'+'\n'+r'GDR'+r'$={0:s}$'.format(as_si((nH*mh.to('g').d)/rho_projectile,2)),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=ax.transAxes,fontsize=13)
    ax.hlines(t_shattering(rho_projectile/(nH*mh.to('g').d),nH,target_a,target_s),0.1,10,color='k')
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('shattering_timescale_%s.png'%(composition),format='png',dpi=300)
    plt.close(fig)
    
def plot_shattering_frag_full(GDR_small,GDR_big,nMach=100):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })

    from utils import as_si
    fig, axes = plt.subplots(2,3, figsize=(13,8),dpi=300,facecolor='w',edgecolor='k',sharex=True,sharey=True)
    fig2, axes2 = plt.subplots(2,3, figsize=(13,8),dpi=300,facecolor='w',edgecolor='k',sharex=True,sharey=True)
    
    # Shattering model quantities (Kobayashi & Tanaka 2010)
    Q_D = [8.9e9,4.3e10]
    alpha_f = 3.3
    
    phases = {'CNM':{'T':100,'nH':30,'ne':0.03,'L':0.64},
              'WNM':{'T':6000,'nH':0.3,'ne':0.03,'L':100},
              'WIM':{'T':8000,'nH':0.1,'ne':0.0991,'L':100}}
    phase_colors = ['b','orange','r']
    compositions = ['Carbonaceous','Silicates']
    collision_model = {'Big-Big':{'target_a':0.1e-4,'projectile_a':0.1e-4},
                       'Big-Small':{'target_a':0.1e-4,'projectile_a':0.005e-4}}
    colors = [['steelblue','royalblue','cornflowerblue','lightsteelblue'],
              ['saddlebrown','chocolate','sandybrown']]
    
    # Add GDR text
    axes2[0,1].text(0.4, 0.8, r'GDR($a_{\rm L}$)'+r'$={0:s}$'.format(as_si(GDR_big,2))+'\n'+
                                r'GDR($a_{\rm S}$)'+r'$={0:s}$'.format(as_si(GDR_small,2)),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes2[0,1].transAxes,fontsize=14)
    
    # Loop over grain compositions
    for c,comp in enumerate(compositions):
    
        if comp == 'Carbonaceous':
            index = 0
            target_s = 2.2
            projectile_s = 2.2
        elif comp == 'Silicates':
            index = 1
            target_s = 3.3
            projectile_s = 3.3
        
        # Loop over collision models
        for m,model_name in enumerate(collision_model):
            model = collision_model[model_name]
            target_a = model['target_a']
            projectile_a = model['projectile_a']
        
            # Grain quantities
            mass_target = 4./3. * np.pi * target_s * (target_a**3.)
            mass_projectile = 4./3. * np.pi * projectile_s * (projectile_a**3.)
            
            Mach = np.logspace(-1,1,nMach)
            
            # Limits of the distributions
            m_dest_min = 4./3. * np.pi * target_s * ((1e-8)**3.)
            m_vsmall_min = 4./3. * np.pi * target_s * ((1e-8)**3.)
            m_vsmall_max = 4./3. * np.pi * target_s * ((1e-7)**3.)
            m_small_min = 4./3. * np.pi * target_s * ((1e-7)**3.)
            m_small_max = 4./3. * np.pi * target_s * ((2e-6)**3.)
            m_big_min = 4./3. * np.pi * target_s * ((2e-6)**3.)
            m_big_max = 4./3. * np.pi * target_s * ((3e-5)**3.)
            
            exponent1 = (4.-alpha_f)/3.
            
            # Loop over ISM phases
            for p,phase_name in enumerate(phases):
                ax = axes[m,p]
                ax2 = axes2[m,p]
                phase = phases[phase_name]
                nH = phase['nH']
                T = phase['T']
                ne = phase['ne']
                Lmax = phase['L']
                
                if model_name == 'Big-Big':
                    rho_projectile = nH * mh.to('g').d * (1./GDR_big)
                elif model_name == 'Big-Small':
                    rho_projectile = nH * mh.to('g').d * (1./GDR_small)
                
                M_dest = np.zeros((nMach,3))
                M_vsmall = np.zeros((nMach,3))
                M_small = np.zeros((nMach,3))
                M_big = np.zeros((nMach,3))
                
                t_dest = np.zeros((nMach,3))
                t_vsmall = np.zeros((nMach,3))
                t_small = np.zeros((nMach,3))
                t_big = np.zeros((nMach,3))
                
                
                for i in range(0, nMach):
                    
                    # Relative velocity between two grains of the same size (Eq. 18 of Hirashita & Aoyama 2019)
                    v_target = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(target_a/1e-5) * ((T/1e4)**(1./4.)) * (nH**(-1./4.)) * np.sqrt(target_s/3.5)
                    v_projectile = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(projectile_a/1e-5) * ((T/1e4)**(1./4.)) * (nH**(-1./4.)) * np.sqrt(projectile_s/3.5)
                    v_rel_min = min(abs(v_target-v_projectile),v_target,v_projectile)
                    v_rel_max = v_target+v_projectile
                    
                    
                    # Average collision velocity as given by the Ormel & Cuzzy (2007) fitting functions
                    v_rel_avg = relative_velocity('Ormel and Cuzzi2007',T,nH,ne,Mach[i],Lmax,target_a,projectile_a,
                                                target_s,projectile_s)
                    v_rel = np.array([max(v_rel_min,v_projectile),v_rel_max,v_rel_avg])
                    
                    # Disrupted mass computation (Eqs. 20-22 of Hirashita & Aoyama 2019)
                    E_imp = 0.5 * (mass_projectile*mass_target)/(mass_target+mass_projectile) * v_rel**2.
                    phi = E_imp / (mass_target*Q_D[index])
                    m_ej = phi / (1.+phi) * mass_target
                    
                    # Now compute the maximum and minimum masses of the fragments
                    m_remnant = mass_target - m_ej
                    m_max = 0.02*m_ej
                    m_min = 1e-6*m_max
                    
                    # Compute the mass fractions for each size bin
                    prefactor = m_ej /(m_max**exponent1-m_min**exponent1)
                    for j in range(0,3):
                        # 1. Destruction to the gas phase (1e-4 mum)
                        if m_min[j] >= m_dest_min:
                            M_dest[i,j] = 0.0
                            if m_remnant[j] < m_dest_min:
                                 M_dest[i,j] += m_remnant[j]
                        else:
                            M_dest[i,j] = prefactor[j] * (min(m_dest_min,m_max[j])**exponent1-m_min[j]**exponent1)
                            if m_remnant[j] < m_dest_min:
                                 M_dest[i,j] += m_remnant[j]
                        
                        # 2. Destruction to very small grains
                    
                        if m_min[j] >= m_vsmall_max or m_max[j] < m_vsmall_min:
                            M_vsmall[i,j] =  0.0
                            if m_vsmall_min <= m_remnant[j] < m_vsmall_max:
                                M_vsmall[i,j] += m_remnant[j]
                        else:
                            M_vsmall[i,j] = prefactor[j] * (min(m_vsmall_max,m_max[j])**exponent1-max(m_vsmall_min,m_min[j])**exponent1)
                            if m_vsmall_min <= m_remnant[j] < m_vsmall_max:
                                M_vsmall[i,j] += m_remnant[j]
                        
                        # 3. Destruction to small grains
                        if m_min[j] >= m_small_max or m_max[j] < m_small_min:
                            M_small[i,j] = 0.
                            if m_small_min <= m_remnant[j] < m_small_max:
                                M_small[i,j] += m_remnant[j]
                        else:
                            M_small[i,j] = prefactor[j] * (min(m_small_max,m_max[j])**exponent1-max(m_small_min,m_min[j])**exponent1)
                            if m_small_min <= m_remnant[j] < m_small_max:
                                M_small[i,j] += m_remnant[j]

                        # 4. Destruction to big grains
                        if m_min[j] >= m_big_max or m_max[j] < m_big_min:
                            M_big[i,j] =  0.0
                            if m_big_min <= m_remnant[j] < m_big_max:
                                M_big[i,j] += m_remnant[j]
                        else:
                            M_big[i,j] = prefactor[j] * (min(m_big_max,m_max[j])**exponent1-max(m_big_min,m_min[j])**exponent1)
                            if m_big_min <= m_remnant[j] < m_big_max:
                                M_big[i,j] += m_remnant[j]
                        
                    # 5. Put remnant fragment to its correct bin
                    M_tot = (M_dest[i] + M_vsmall[i] + M_small[i] + M_big[i])
                    if comp == 'Silicates':
                        M_dest += M_vsmall
                    M_dest[i] = M_dest[i]/M_tot
                    M_vsmall[i] = M_vsmall[i]/M_tot
                    M_small[i] = M_small[i]/M_tot
                    M_big[i] = M_big[i]/M_tot
                                        
                    alpha = np.pi * (target_a + projectile_a)**2. * v_rel / (mass_target*mass_projectile)
                    t_dest[i] = 1./(alpha*rho_projectile*mass_target*M_dest[i])/sec2Myr
                    t_vsmall[i] = 1./(alpha*rho_projectile*mass_target*M_vsmall[i])/sec2Myr
                    t_small[i] = 1./(alpha*rho_projectile*mass_target*M_small[i])/sec2Myr
                    t_big[i] = 1./(alpha*rho_projectile*mass_target*M_big[i])/sec2Myr
                    
                selected = (M_dest[:,0]>0.) & (M_dest[:,1]>0.)
                
                if p == 1 and m == 0 and comp == 'Silicates':
                    ax.fill_between(Mach,M_dest[:,0],M_dest[:,1],label='Destroyed '+comp,alpha=0.5,where=(selected),color=colors[index][-1])
                    ax.fill_between(Mach,M_small[:,0],M_small[:,1],label='Small '+comp,alpha=0.5,where=M_small[:,1]>0.,color=colors[index][1])
                    ax.fill_between(Mach,M_big[:,0],M_big[:,1],label='Large '+comp,alpha=0.5,where=M_big[:,1]>0.,color=colors[index][0])
                elif p == 2 and m == 0 and comp == 'Carbonaceous':
                    ax.fill_between(Mach,M_dest[:,0],M_dest[:,1],label='Destroyed '+comp,alpha=0.5,where=(selected),color=colors[index][-1])
                    ax.fill_between(Mach,M_vsmall[:,0],M_vsmall[:,1],label='PAHs',alpha=0.5,where=M_vsmall[:,1]>0.,color=colors[index][2])
                    ax.fill_between(Mach,M_small[:,0],M_small[:,1],label='Small '+comp,alpha=0.5,where=M_small[:,1]>0.,color=colors[index][1])
                    ax.fill_between(Mach,M_big[:,0],M_big[:,1],label='Large '+comp,alpha=0.5,where=M_big[:,1]>0.,color=colors[index][0])
                else:
                    ax.fill_between(Mach,M_dest[:,0],M_dest[:,1],alpha=0.5,where=(selected),color=colors[index][-1])
                    if comp == 'Carbonaceous':
                        ax.fill_between(Mach,M_vsmall[:,0],M_vsmall[:,1],alpha=0.5,where=M_vsmall[:,1]>0.,color=colors[index][2])
                    ax.fill_between(Mach,M_small[:,0],M_small[:,1],alpha=0.5,where=M_small[:,1]>0.,color=colors[index][1])
                    ax.fill_between(Mach,M_big[:,0],M_big[:,1],alpha=0.5,where=M_big[:,1]>0.,color=colors[index][0])
                ax.plot(Mach,M_dest[:,2],linestyle='-.',color=colors[index][-1])
                ax.plot(Mach,M_vsmall[:,2],linestyle='-.',color=colors[index][2])
                ax.plot(Mach,M_small[:,2],linestyle='-.',color=colors[index][1])
                ax.plot(Mach,M_big[:,2],linestyle='-.',color=colors[index][0])
                
                if p == 1 and m == 0 and comp == 'Silicates':
                    ax2.fill_between(Mach,t_dest[:,0],t_dest[:,1],label='Destroyed '+comp,alpha=0.5,color=colors[index][-1])
                    selected = ((t_small[:,1]!=np.infty) & (t_small[:,0]!=np.infty))
                    ax2.fill_between(Mach,t_small[:,0],t_small[:,1],label='Small '+comp,alpha=0.5,color=colors[index][1])
                    ax2.fill_between(Mach,t_big[:,0],t_big[:,1],label='Large '+comp,alpha=0.5,color=colors[index][0])
                elif p == 2 and m == 0 and comp == 'Carbonaceous':
                    ax2.fill_between(Mach,t_dest[:,0],t_dest[:,1],label='Destroyed '+comp,alpha=0.5,where=t_dest[:,1]!=np.infty,color=colors[index][-1])
                    ax2.fill_between(Mach,t_vsmall[:,0],t_vsmall[:,1],label='PAHs',alpha=0.5,where=t_vsmall[:,1]!=np.infty,color=colors[index][2])
                    ax2.fill_between(Mach,t_small[:,0],t_small[:,1],label='Small '+comp,alpha=0.5,where=t_small[:,1]!=np.infty,color=colors[index][1])
                    ax2.fill_between(Mach,t_big[:,0],t_big[:,1],label='Large '+comp,alpha=0.5,where=t_big[:,1]!=np.infty,color=colors[index][0])
                else:
                    ax2.fill_between(Mach,t_dest[:,0],t_dest[:,1],alpha=0.5,color=colors[index][-1])
                    if comp == 'Carbonaceous':
                        ax2.fill_between(Mach,t_vsmall[:,0],t_vsmall[:,1],alpha=0.5,color=colors[index][2])
                    ax2.fill_between(Mach,t_small[:,0],t_small[:,1],alpha=0.5,color=colors[index][1])
                    ax2.fill_between(Mach,t_big[:,0],t_big[:,1],alpha=0.5,color=colors[index][0])
                ax2.plot(Mach,t_dest[:,2],linestyle='-.',color=colors[index][-1])
                ax2.plot(Mach,t_vsmall[:,2],linestyle='-.',color=colors[index][2])
                ax2.plot(Mach,t_small[:,2],linestyle='-.',color=colors[index][1])
                ax2.plot(Mach,t_big[:,2],linestyle='-.',color=colors[index][0])
                ax2.hlines(t_shattering(rho_projectile/(nH*mh.to('g').d),nH,target_a,target_s),0.1,10,color=colors[index][1])
    
    # Add legend of composition/size type
    ax = axes[0,1]
    first_legend = ax.legend(loc='lower right', fontsize=14,frameon=False)
    ax.add_artist(first_legend)
    
    ax = axes2[0,1]
    first_legend = ax.legend(loc='lower right', fontsize=14,frameon=False)
    ax.add_artist(first_legend)
    
    ax = axes[0,2]
    first_legend = ax.legend(loc='lower right', fontsize=14,frameon=False)
    ax.add_artist(first_legend)
    
    ax = axes2[0,2]
    first_legend = ax.legend(loc='lower right', fontsize=14,frameon=False)
    ax.add_artist(first_legend)
    
    # Add legend of velocity model type
    ax = axes[1,0]
    dummy_lines = []

    dummy_lines.append(ax.fill_between([],[], color="black",label = 'Hirashita and Aoyama 2019'))
    dummy_lines.append(ax.plot([],[], color="black", ls = '-.',label = 'Ormel and Cuzzi 2007')[0])
    second_legend = ax.legend(handles=dummy_lines, loc='lower left', frameon=False, fontsize=14,ncol=1)         
    ax.add_artist(second_legend)
    
    ax = axes2[1,0]
    dummy_lines = []

    dummy_lines.append(ax.fill_between([],[], color="black",label = 'Hirashita and Aoyama 2019'))
    dummy_lines.append(ax.plot([],[], color="black", ls = '-.',label = 'Ormel and Cuzzi 2007')[0])
    dummy_lines.append(ax.plot([],[], color="black", ls = '-',label = 'Granato et al. 2021')[0])
    second_legend = ax.legend(handles=dummy_lines, loc='center left', frameon=False, fontsize=14,ncol=1)         
    ax.add_artist(second_legend)
    
    # Setup axes
    p = 0
    for i in range(0,2):
        for j,k in enumerate(phases):
            ax = axes[i,j]
            ax.tick_params(labelsize=14)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.minorticks_on()
            ax.tick_params(which='both',axis="both",direction="in")
            ax.set_yscale('log')
            ax.set_xscale('log')
            phase = k
            ax.text(0.08, 0.90, r'\textbf{%s}'%phase,
                    transform=ax.transAxes, fontsize=20,verticalalignment='top',
                    color=phase_colors[j], weight='bold')
            ax.set_ylim([1e-4,1])
            p += 1
            if i==1:
                ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)
    
    p = 0
    for i in range(0,2):
        for j,k in enumerate(phases):
            ax = axes2[i,j]
            ax.tick_params(labelsize=14)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.minorticks_on()
            ax.tick_params(which='both',axis="both",direction="in")
            ax.set_yscale('log')
            ax.set_xscale('log')
            phase = k
            ax.text(0.02, 0.90, r'\textbf{%s}'%phase,
                    transform=ax.transAxes, fontsize=20,verticalalignment='top',
                    color=phase_colors[j], weight='bold')
            ax.set_ylim([5e-3,1e5])            
            p += 1
            if i==1:
                ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)

    # Setup y-labels
    axes[0,0].set_ylabel(r'$\chi_{\rm frag}(a,\mathcal{M};a_{\rm L},a_{\rm L})$', fontsize=20)
    axes[1,0].set_ylabel(r'$\chi_{\rm frag}(a,\mathcal{M};a_{\rm L},a_{\rm S})$', fontsize=20)
    
    axes2[0,0].set_ylabel(r'$t_{\rm sha}(a,\mathcal{M};a_{\rm L},a_{\rm L})$ [Myr]', fontsize=20)
    axes2[1,0].set_ylabel(r'$t_{\rm sha}(a,\mathcal{M};a_{\rm L},a_{\rm S})$ [Myr]', fontsize=20)
    
    fig.subplots_adjust(top=0.98,bottom=0.07,left=0.06,right=0.99,hspace=0,wspace=0)
    fig.savefig('shattering_efficiency_full.pdf',format='pdf',dpi=300)
    plt.close(fig)
    
    fig2.subplots_adjust(top=0.98,bottom=0.07,left=0.06,right=0.99,hspace=0,wspace=0)
    fig2.savefig('shattering_timescale_full.pdf',format='pdf',dpi=300)
    plt.close(fig2)
    
    
def plot_coagulation(target_a,projectile_a,target_s,projectile_s,composition,nH,ne,T,nsmall,nMach=100):
    # This function plots the shattering fragment distribution for big grains
    # based on the power-law model 
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    from utils import as_si,sigmoid_function
    from scipy.special import erfc
    
    # Coagulation model quantities (Hirashita & Yan 2009)
    E = [3.4e10,5.4e11]
    gamma = [12,25]
    
    if composition == 'Gra':
        index = 0
    elif composition == 'Sil':
        index = 1
    
    # Grain quantities
    mass_target = 4./3. * np.pi * target_s * (target_a**3.)
    mass_projectile = 4./3. * np.pi * projectile_s * (projectile_a**3.)
    rho_small = nsmall * mass_projectile
    R = target_a*projectile_a / (projectile_a + target_a)
    v_coag = 21.4 * np.sqrt((mass_target**3.+mass_projectile**3.)/(mass_projectile+mass_target)**3.) * \
            gamma[index]**(5./3.) / (E[index]**(1./3.)*R**(5./6.)*np.sqrt(target_s))
    
    Mach = np.logspace(-1,1,nMach)
    Lmax = 10 # [pc]
    
    t_coag = np.zeros((nMach,4))
    
    t_min = np.pi * (target_a + projectile_a)**2. * v_coag
    t_max = 1e9
    kernel = t_max/(1e12*t_min)
    
    for i in range(0, nMach):
        
        # Boosting of density due to subgrid turbulence
        lambda_jeans = 3.8409904e7 * np.sqrt(T/(nH*mh.to('g').d))
        nhmax_coa = 1e20
        sigs = np.log(1.+(0.4*Mach[i])**2.)
        sigs2 = sigs**2.
        smax = np.log(nhmax_coa/nH)
        boost_coa = 0.5*np.exp(sigs2)*erfc((1.5*sigs2-smax)/(np.sqrt(2.)*sigs))
        L = Lmax * 3.0857e18 # [cm]
        if T>1e4 or nH<1e2 or lambda_jeans>4*L:
            boost_coa = 1.
            
        # Relative velocity between two grains of the same size (Eq. 18 of Hirashita & Aoyama 2019)
        v_target = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(target_a/1e-5) * ((T/1e4)**(1./4.)) * ((nH)**(-1./4.)) * np.sqrt(target_s/3.5)
        v_projectile = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(projectile_a/1e-5) * ((T/1e4)**(1./4.)) * ((nH)**(-1./4.)) * np.sqrt(projectile_s/3.5)
        v_rel_min = min(abs(v_target-v_projectile),v_target,v_projectile)
        v_rel_max = v_target+v_projectile
        
        
        # Average collision velocity as given by the Ormel & Cuzzy (2007) fitting functions
        v_rel_avg = relative_velocity('Ormel and Cuzzi2007',T,nH,ne,Mach[i],Lmax,target_a,projectile_a,
                                      target_s,projectile_s)
        v_rel = np.array([max(v_rel_min,v_projectile),v_rel_max,v_rel_avg])
        
        # Collision rate parameters
        
        alpha = np.pi * (target_a + projectile_a)**2. * v_rel
        for j in range(0,3):
            t = 1./(alpha[j]*nsmall*boost_coa)/sec2Myr
            # if v_rel[j] <= v_coag:
            #     t_coag[i,j] = t
            # else:
            #     t_coag[i,j] = 1e9
            t_coag[i,j] = (1-sigmoid_function(kernel,v_coag,v_rel[j]))*t + sigmoid_function(kernel,v_coag,v_rel[j]) * t_max
        t_coag[i,-1] = t_coagulation(rho_small/(nH*mh.to('g').d),Mach[i],nH,T,Lmax,projectile_a,projectile_s)
    
    # Timescale plot
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,6), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$t_{\rm coa}(a,\mathcal{M})$ [Myr]', fontsize=16)
    ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    
    ax.fill_between(Mach,t_coag[:,0],t_coag[:,1],label='Kobayashi&Tanaka2010-Hirashita&Aoyama2019',alpha=0.5)
    ax.plot(Mach,t_coag[:,2],linestyle='-.',label='Kobayashi&Tanaka2010-Ormel&Cuzzi2007')
    
    ax.text(0.1, 0.2, r'$T_{\rm gas}$'+r'$={0:s}$'.format(as_si(T,1))+
                        ' K\n'+r'$n_{\rm H}$'+r'$={0:s}$'.format(as_si(nH,1))+
                        r' cm$^{-3}$'+'\n'+r'GDR'+r'$={0:s}$'.format(as_si((nH*mh.to('g').d)/rho_small,2)),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=ax.transAxes,fontsize=13)
    
    ax.plot(Mach,t_coag[:,3],linestyle='-',color='k',label='Basic-TurbulentModel')
    ax.legend(loc='best',fontsize=10,frameon=False)
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('coagulation_timescale_%s.png'%(composition),format='png',dpi=300)
    plt.close(fig)
    
def plot_coagulation_full(GDR_small,GDR_big,nMach=100):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })

    from utils import as_si,sigmoid_function
    from scipy.special import erfc
    fig2, axes2 = plt.subplots(2,3, figsize=(13,8),dpi=300,facecolor='w',edgecolor='k',sharex=True,sharey=True)
    
    # Coagulation model quantities (Hirashita & Yan 2009)
    E = [3.4e10,5.4e11]
    gamma = [12,25]
    
    phases = {'DC1':{'T':10,'nH':1e4,'ne':0.01,'L':1},
              'MC':{'T':25,'nH':300,'ne':0.03,'L':1},
              'CNM':{'T':100,'nH':30,'ne':0.0991,'L':0.64}}
    phase_colors = ['indigo','goldenrod','b']
    compositions = ['Carbonaceous','Silicates']
    collision_model = {'Small-Small':{'target_a':0.005e-4,'projectile_a':0.005e-4},
                       'Big-Small':{'target_a':0.1e-4,'projectile_a':0.005e-4}}
    colors = [['steelblue','royalblue','cornflowerblue','lightsteelblue'],
              ['saddlebrown','chocolate','sandybrown']]
    
    # Add GDR text
    axes2[0,1].text(0.4, 0.8,r'GDR($a_{\rm small}$)'+r'$={0:s}$'.format(as_si(GDR_small,2))+'\n'+
                                r'GDR($a_{\rm large}$)'+r'$={0:s}$'.format(as_si(GDR_big,2)),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes2[0,1].transAxes,fontsize=14)
    
    # Loop over grain compositions
    for c,comp in enumerate(compositions):
    
        if comp == 'Carbonaceous':
            index = 0
            target_s = 2.2
            projectile_s = 2.2
        elif comp == 'Silicates':
            index = 1
            target_s = 3.3
            projectile_s = 3.3
        
        # Loop over collision models
        for m,model_name in enumerate(collision_model):
            model = collision_model[model_name]
            target_a = model['target_a']
            projectile_a = model['projectile_a']
        
            # Grain quantities
            mass_target = 4./3. * np.pi * target_s * (target_a**3.)
            mass_projectile = 4./3. * np.pi * projectile_s * (projectile_a**3.)
            R = target_a*projectile_a / (projectile_a + target_a)
            v_coag = 21.4 * np.sqrt((mass_target**3.+mass_projectile**3.)/(mass_projectile+mass_target)**3.) * \
            gamma[index]**(5./3.) / (E[index]**(1./3.)*R**(5./6.)*np.sqrt(target_s))
            
            Mach = np.logspace(-1,1,nMach)
            
            # Loop over ISM phases
            for p,phase_name in enumerate(phases):
                ax2 = axes2[m,p]
                phase = phases[phase_name]
                nH = phase['nH']
                T = phase['T']
                ne = phase['ne']
                Lmax = phase['L']
                
                if model_name == 'Small-Small':
                    rho_projectile = nH * mh.to('g').d * (1./GDR_small)
                elif model_name == 'Big-Small':
                    rho_projectile = nH * mh.to('g').d * (1./GDR_big)
                    mass_projectile = mass_target
                t_coag = np.zeros((nMach,4))
                
                for i in range(0, nMach):
                    
                    # Boosting of density due to subgrid turbulence
                    lambda_jeans = 3.8409904e7 * np.sqrt(T/(nH*mh.to('g').d))
                    nhmax_coa = 1e20
                    sigs = np.log(1.+(0.4*Mach[i])**2.)
                    sigs2 = sigs**2.
                    smax = np.log(nhmax_coa/nH)
                    boost_coa = 0.5*np.exp(sigs2)*erfc((1.5*sigs2-smax)/(np.sqrt(2.)*sigs))
                    L = Lmax * 3.0857e18 # [cm]
                    if T>1e4 or nH<1e2 or lambda_jeans>4*L:
                        boost_coa = 1.
                        
                    # Relative velocity between two grains of the same size (Eq. 18 of Hirashita & Aoyama 2019)
                    v_target = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(target_a/1e-5) * ((T/1e4)**(1./4.)) * ((nH)**(-1./4.)) * np.sqrt(target_s/3.5)
                    v_projectile = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(projectile_a/1e-5) * ((T/1e4)**(1./4.)) * ((nH)**(-1./4.)) * np.sqrt(projectile_s/3.5)
                    v_rel_min = min(abs(v_target-v_projectile),v_target,v_projectile)
                    v_rel_max = v_target+v_projectile
                    
                    
                    # Average collision velocity as given by the Ormel & Cuzzy (2007) fitting functions
                    v_rel_avg = relative_velocity('Ormel and Cuzzi2007',T,nH,ne,Mach[i],Lmax,target_a,projectile_a,
                                                target_s,projectile_s)
                    v_rel = np.array([max(v_rel_min,v_projectile),v_rel_max,v_rel_avg])
                    # Collision rate parameters
                    
                    alpha = np.pi * (target_a + projectile_a)**2. * v_rel
                    t_min = mass_projectile/(np.pi * (target_a + projectile_a)**2. * v_coag*rho_projectile)/sec2Myr
                    t_max = 1e6
                    kernel = t_max/(1e5*t_min)
                    for j in range(0,3):
                        t = mass_projectile/(alpha[j]*rho_projectile*boost_coa)/sec2Myr
                        if v_rel[j] <= v_coag:
                            t_coag[i,j] = t
                        else:
                            t_coag[i,j] = 1e6
                        # t_coag[i,j] = (1-sigmoid_function(kernel,v_coag,v_rel[j]))*t + sigmoid_function(kernel,v_coag,v_rel[j]) * t_max
                    t_coag[i,-1] = t_coagulation(1./GDR_small,Mach[i],nH,T,Lmax,projectile_a,projectile_s)
                
                if p == 1 and m == 0:
                    ax2.fill_between(Mach,t_coag[:,0],t_coag[:,1],label=comp,alpha=0.5,color=colors[index][0])
                else:
                    ax2.fill_between(Mach,t_coag[:,0],t_coag[:,1],alpha=0.5,color=colors[index][0])
                ax2.plot(Mach,t_coag[:,2],linestyle='-.',color=colors[index][1])
                ax2.plot(Mach,t_coag[:,3],linestyle='-',color=colors[index][1])
    
    # Add legend of composition/size type    
    ax = axes2[0,1]
    first_legend = ax.legend(loc='lower right', fontsize=14,frameon=False)
    ax.add_artist(first_legend)
    
    # Add legend of velocity model type    
    ax = axes2[1,0]
    dummy_lines = []

    dummy_lines.append(ax.fill_between([],[], color="black",label = 'Hirashita and Aoyama 2019'))
    dummy_lines.append(ax.plot([],[], color="black", ls = '-.',label = 'Ormel and Cuzzi 2007')[0])
    dummy_lines.append(ax.plot([],[], color="black", ls = '-',label = 'Aoyama et al. 2017')[0])
    second_legend = ax.legend(handles=dummy_lines, loc='lower left', frameon=False, fontsize=14,ncol=1)         
    ax.add_artist(second_legend)
    
    # Setup axes
    
    p = 0
    for i in range(0,2):
        for j,k in enumerate(phases):
            ax = axes2[i,j]
            ax.tick_params(labelsize=14)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.minorticks_on()
            ax.tick_params(which='both',axis="both",direction="in")
            ax.set_yscale('log')
            ax.set_xscale('log')
            phase = k
            ax.text(0.02, 0.90, r'\textbf{%s}'%phase,
                    transform=ax.transAxes, fontsize=20,verticalalignment='top',
                    color=phase_colors[j], weight='bold')
            #ax.set_ylim([5e-1,1e5])            
            p += 1
            if i==1:
                ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)

    # Setup y-labels
    
    axes2[0,0].set_ylabel(r'$t_{\rm coa}(a,\mathcal{M};a_{\rm small},a_{\rm small})$ [Myr]', fontsize=20)
    axes2[1,0].set_ylabel(r'$t_{\rm coa}(a,\mathcal{M};a_{\rm big},a_{\rm small})$ [Myr]', fontsize=20)

    fig2.subplots_adjust(top=0.98,bottom=0.07,left=0.06,right=0.99,hspace=0,wspace=0)
    fig2.savefig('coagulation_timescale_full.pdf',format='pdf',dpi=300)
    plt.close(fig2)
    
def plot_coagulation_single(nH,ne,T,Lmax,phase_name,GDR_small,GDR_big,nMach=100):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })

    from utils import as_si,sigmoid_function
    from scipy.special import erfc
    fig2, axes2 = plt.subplots(2,1, figsize=(5,7),dpi=300,facecolor='w',edgecolor='k',sharex=True,sharey=True)
    
    # Coagulation model quantities (Hirashita & Yan 2009)
    E = [3.4e10,5.4e11]
    gamma = [12,25]
    
    compositions = ['Carbonaceous','Silicates']
    collision_model = {'Small-Small':{'target_a':0.005e-4,'projectile_a':0.005e-4},
                       'Big-Small':{'target_a':0.1e-4,'projectile_a':0.005e-4}}
    colors = [['steelblue','royalblue','cornflowerblue','lightsteelblue'],
              ['saddlebrown','chocolate','sandybrown']]
    
    # Add GDR text
    axes2[0].text(0.4, 0.8,r'GDR($a_{\rm S}$)'+r'$={0:s}$'.format(as_si(GDR_small,2))+'\n'+
                                r'GDR($a_{\rm L}$)'+r'$={0:s}$'.format(as_si(GDR_big,2)),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes2[0].transAxes,fontsize=14)
    
    # Loop over grain compositions
    for c,comp in enumerate(compositions):
    
        if comp == 'Carbonaceous':
            index = 0
            target_s = 2.2
            projectile_s = 2.2
        elif comp == 'Silicates':
            index = 1
            target_s = 3.3
            projectile_s = 3.3
        
        # Loop over collision models
        for m,model_name in enumerate(collision_model):
            model = collision_model[model_name]
            target_a = model['target_a']
            projectile_a = model['projectile_a']
        
            # Grain quantities
            mass_target = 4./3. * np.pi * target_s * (target_a**3.)
            mass_projectile = 4./3. * np.pi * projectile_s * (projectile_a**3.)
            R = target_a*projectile_a / (projectile_a + target_a)
            v_coag = 21.4 * np.sqrt((mass_target**3.+mass_projectile**3.)/(mass_projectile+mass_target)**3.) * \
            gamma[index]**(5./3.) / (E[index]**(1./3.)*R**(5./6.)*np.sqrt(target_s))
            
            Mach = np.logspace(-1,1,nMach)
            
            ax2 = axes2[m]
            
            if model_name == 'Small-Small':
                rho_projectile = nH * mh.to('g').d * (1./GDR_small)
            elif model_name == 'Big-Small':
                rho_projectile = nH * mh.to('g').d * (1./GDR_big)
                mass_projectile = mass_target
            t_coag = np.zeros((nMach,4))
            
            t_min = np.log10(mass_projectile/(np.pi * (target_a + projectile_a)**2. * v_coag*rho_projectile)/sec2Myr)
            t_max = np.log10(1e5)
            kernel = t_max
            for i in range(0, nMach):
                
                # Boosting of density due to subgrid turbulence
                lambda_jeans = 3.8409904e7 * np.sqrt(T/(nH*mh.to('g').d))
                nhmax_coa = 1e20
                sigs = np.log(1.+(0.4*Mach[i])**2.)
                sigs2 = sigs**2.
                smax = np.log(nhmax_coa/nH)
                boost_coa = 0.5*np.exp(sigs2)*erfc((1.5*sigs2-smax)/(np.sqrt(2.)*sigs))
                L = Lmax * 3.0857e18 # [cm]
                if T>1e4 or nH<1e2 or lambda_jeans>4*L:
                    boost_coa = 1.
                    
                # Relative velocity between two grains of the same size (Eq. 18 of Hirashita & Aoyama 2019)
                v_target = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(target_a/1e-5) * ((T/1e4)**(1./4.)) * ((nH)**(-1./4.)) * np.sqrt(target_s/3.5)
                v_projectile = 1.1e5 * (Mach[i]**(3./2.)) * np.sqrt(projectile_a/1e-5) * ((T/1e4)**(1./4.)) * ((nH)**(-1./4.)) * np.sqrt(projectile_s/3.5)
                v_rel_min = min(abs(v_target-v_projectile),v_target,v_projectile)
                v_rel_max = v_target+v_projectile
                
                
                # Average collision velocity as given by the Ormel & Cuzzy (2007) fitting functions
                v_rel_avg = relative_velocity('Ormel and Cuzzi2007',T,nH,ne,Mach[i],Lmax,target_a,projectile_a,
                                            target_s,projectile_s)
                v_rel = np.array([max(v_rel_min,v_projectile),v_rel_max,v_rel_avg])
                # Collision rate parameters
                
                alpha = np.pi * (target_a + projectile_a)**2. * v_rel
                
                for j in range(0,3):
                    # t = mass_projectile/(alpha[j]*rho_projectile*boost_coa)/sec2Myr
                    t = np.log10(mass_projectile/(alpha[j]*rho_projectile*boost_coa)/sec2Myr)
                    # if v_rel[j] <= v_coag:
                    #     t_coag[i,j] = t
                    # else:
                    #     t_coag[i,j] = 1e6
                    t_coag[i,j] = (1-sigmoid_function(kernel,v_coag,v_rel[j]))*t + sigmoid_function(kernel,v_coag,v_rel[j]) * t_max
                    t_coag[i,j] = 10**(t_coag[i,j])
                t_coag[i,-1] = t_coagulation(1./GDR_small,Mach[i],nH,T,Lmax,projectile_a,projectile_s)
            
            if m == 0:
                ax2.fill_between(Mach,t_coag[:,0],t_coag[:,1],label=comp,alpha=0.5,color=colors[index][0])
            else:
                ax2.fill_between(Mach,t_coag[:,0],t_coag[:,1],alpha=0.5,color=colors[index][0])
            ax2.plot(Mach,t_coag[:,2],linestyle='-.',color=colors[index][1])
            ax2.plot(Mach,t_coag[:,3],linestyle='-',color=colors[index][1])
    
    # Add legend of composition/size type    
    ax = axes2[0]
    first_legend = ax.legend(loc='lower left', fontsize=14,frameon=False)
    ax.add_artist(first_legend)
    
    # Add legend of velocity model type    
    ax = axes2[1]
    dummy_lines = []

    dummy_lines.append(ax.fill_between([],[], color="black",label = 'Hirashita and Aoyama 2019'))
    dummy_lines.append(ax.plot([],[], color="black", ls = '-.',label = 'Ormel and Cuzzi 2007')[0])
    dummy_lines.append(ax.plot([],[], color="black", ls = '-',label = 'Aoyama et al. 2017')[0])
    second_legend = ax.legend(handles=dummy_lines, loc='lower left', frameon=False, fontsize=14,ncol=1)         
    ax.add_artist(second_legend)
    
    # Setup axes
    
    p = 0
    for i in range(0,2):
        ax = axes2[i]
        ax.tick_params(labelsize=14)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.text(0.02, 0.90, r'\textbf{%s}'%phase_name,
                transform=ax.transAxes, fontsize=20,verticalalignment='top',
                color='goldenrod', weight='bold')
        ax.set_ylim([2e-2,2e5])            
        p += 1
        if i==1:
            ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)

    # Setup y-labels
    
    axes2[0].set_ylabel(r'$t_{\rm coa}(a_{\rm L},\mathcal{M};a_{\rm S},a_{\rm S})$ [Myr]', fontsize=20)
    axes2[1].set_ylabel(r'$t_{\rm coa}(a_{\rm L},\mathcal{M};a_{\rm L},a_{\rm S})$ [Myr]', fontsize=20)

    fig2.subplots_adjust(top=0.98,bottom=0.07,left=0.16,right=0.99,hspace=0,wspace=0)
    fig2.savefig('coagulation_timescale_%s.pdf'%phase_name,format='pdf',dpi=300)
    plt.close(fig2)
    
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