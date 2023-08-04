"""
MODELLING PAHs DISTRIBUTIONS AND EVOLUTION

These functions allow for the computation of the intrinsic PAHs
size distribution and how different processes affect their total mass.

By: Curro Rodriguez (currodri@gmail.com)
"""

# Import libraries
import numpy as np
import pandas as pd
from dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,LogNormal_Distribution

# Galliano data on dust destruction timescales by UV photons
# See https://irfu.cea.fr/Pisp/frederic.galliano/HDR/hdrch6.html#x7-3070004 (Section 4.2.2.1)
names = ['3A_x','3A_y','3.67A_x','3.67A_y','4.48A_x','4.48_y','5.47A_x','5.47A_y']
carbon_subl_time = pd.read_csv('uv_subl_carbonaceous.csv',names=names,header=2)

# Allain et al. (1996) C2H2 dissociation timescales by UV photons
# See https://articles.adsabs.harvard.edu/pdf/1996A%26A...305..602A (Eq. 25 and Table 6)
allain_Nc = [6,14,16,24,32,50]
allain_rates = [1.49e-10,1.89e-10,7.13e-11,4.55e-11,4.85e-12,3.56e-18]

# Murga et al. (2019) - SHIVA model prediction for PAH of 5 Angstrom photo-destruction
murga_subl_time = pd.read_csv('uv_subl_pah_Murga2019.csv',header=1)

# Micelotta et al. (2010) PAH processing in a hot gas
# See 

thermal_spu = {'50':{'electrons': {'a':-2136.83,'b':1632.17,'c':-499.822,'d':76.4347,'e':-5.82964,'f':0.177174},
                     'H':         {'a':-1896.69,'b':1480.8,'c':-462.733,'d':71.8957,'e':-5.54719,'f':0.169996},
                     'He':        {'a':-971.448,'b':770.259,'c':-245.561,'d':38.8995,'e':-3.05787,'f':0.0954303},
                     'C':         {'a':-704.392,'b':551.313,'c':-175.063,'d':27.6506,'e':-2.16871,'f':0.0675643}},
               
               '100':{'electrons': {'a':-2255.38,'b':1681.45,'c':-503.451,'d':75.4339,'e':-5.64956,'f':0.168959},
                     'H':         {'a':-1645.64,'b':1257.06,'c':-384.309,'d':58.4103,'e':-4.41007,'f':0.132356},
                     'He':        {'a':-945.901,'b':747.921,'c':-237.984,'d':37.6808,'e':-2.96613,'f':0.0928852},
                     'C':         {'a':-711.244,'b':558.48,'c':-177.983,'d':28.2547,'e':-2.23145,'f':0.0701374}},
               
               '200':{'electrons': {'a':-2234.37,'b':1597.71,'c':-459.647,'d':66.332,'e':-4.79841,'f':0.139019},
                     'H':         {'a':-1473.64,'b':1109.01,'c':-334.292,'d':50.1417,'e':-3.74133,'f':0.111164},
                     'He':        {'a':-963.188,'b':765.639,'c':-245.054,'d':39.0745,'e':-3.10123,'f':0.0980047},
                     'C':         {'a':-738.791,'b':584.928,'c':-187.884,'d':30.0819,'e':-2.39748,'f':0.0760639}}}

# H2 formation onto PAHs by Le Page et al. (2009) -
# (https://ui.adsabs.harvard.edu/abs/2009ApJ...704..274L/abstract)
# The mechanism involves the chemical trapping of H atoms on the periphery of the PAH
# carbon skeleton and the subsequent release of H2 through dissociative recombination
# of the hydrogenated ion with an electron.
names = ['C32_x','C32_y','C40_x','C40_y','C50_x','C50_y','C80_x',
         'C80_y','C100_x','C100_y','C120_x','C120_y','C150_x','C150_y']
H2_rate_LePage = pd.read_csv('H2_formation_rate_PAH_LePage2009.csv',header=1,names=names)

def Nc_from_size(a):
    """This function returns the effective number of Carbon atoms
    for a PAH molecule, following Eq. 8 in Draine et al. (2021).

    Args:
        a (float): grain radius in Angstrom
    """

    return int(418*(a/10)**3)

def size_from_Nc(Nc):
    """This function returns the PAH radius (in Angstrom) from the number of Carbon atoms,
    following Eq. 8 in Draine et al. (2021).

    Args:
        Nc (int): number of Carbon atoms
    """

    return 10*((float(Nc)/418))**(1/3)

def plot_distribution(rho_gas,D_PAHs,D_small,D_large):
    """Create figure for the plotting of the full dust distribution.

    Args:
        rho_gas (float): Gas density in g/cm^3
        D_PAHs (float): PAHs mass fraction
        D_small (float): Small carbonaceous fraction
        D_large (float): Large carbonaceous fraction
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    sizes = np.logspace(np.log10(1e-4),np.log10(1),1000)
    PAHs = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    small = LogNormal_Distribution(basic_a0[1],basic_amin[1],basic_amax[1],basic_sigma[1],basic_s[1])
    large = LogNormal_Distribution(basic_a0[2],basic_amin[2],basic_amax[2],basic_sigma[2],basic_s[2])

    n_PAHs = PAHs.n_density(rho_gas*D_PAHs,sizes)
    n_small = small.n_density(rho_gas*D_small,sizes)
    n_large = large.n_density(rho_gas*D_large,sizes)

    n_tot = n_PAHs + n_small + n_large

    n_tot = (sizes**4)*n_tot

    ax.plot(sizes,n_tot,'k-',label='Total')
    ax.plot(sizes,(sizes**4)*n_PAHs,'--',color='blue',label='PAHs')
    ax.plot(sizes,(sizes**4)*n_small,'-.',color='green',label='Small CDust')
    ax.plot(sizes,(sizes**4)*n_large,':',color='red',label='Large CDust')
    ax.set_ylabel(r'$a^4 n(a)$', fontsize=16)
    ax.set_xlabel(r'$a$ [$\mu$m]',fontsize=16)
    ax.set_ylim([1e-30,3e-27])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.legend(loc='best',fontsize=14,frameon=False)

    ax.plot(sizes,1e-27*sizes**(.5),':',color='gray')
    ax.text(0.4, 0.6, r'MRN ($n\propto a^{-3.5}$)',
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=ax.transAxes,fontsize=10)

    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.15,right=0.99)

    return fig

def subl_func1(x,a,b,c):
    # Power law with exponential cutoff
    return a - b*np.log10(x) - c*x

def subl_func2(x,a,b):
    # Power law
    return a - b*np.log10(x)
def subl_func3(x,a,b,c,d):
    # Broken power law with smooth transition
    return a + b*np.log10(x) + np.log10(1+c*x**d)

def Allain_time(chi,k):
    return (chi*k)**(-1)

def plot_UV_sublimation():
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import seaborn as sns
    sns.set(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8,7), dpi=300, facecolor='w', edgecolor='k')
    x_names = ['3A_x','3.67A_x','4.48A_x','5.47A_x']
    y_names = ['3A_y','3.67A_y','4.48_y','5.47A_y']
    labels = ['3.0 $\AA$','3.67 $\AA$','4.48 $\AA$','5.47 $\AA$']
    
    fittings = []
    U = np.logspace(-1,7,100)
    for i in range(0, len(x_names)):
        x = np.asarray(carbon_subl_time[x_names[i]])
        y = np.asarray(carbon_subl_time[y_names[i]])
        ax.plot(x,y,label=labels[i]+' Galliano')
        popt,pcov = curve_fit(subl_func1,x[~np.isnan(x)],np.log10(y[~np.isnan(x)]))
        fittings.append(popt)
        ax.plot(U,10**subl_func1(U,*popt),'--',color='k')
        
    
    PAHs = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    sizes = np.array([3e-4,3.67e-4,4.48e-4,5.47e-4])
    
    tau_avg = np.zeros(100)
    for i in range(0,len(tau_avg)):
        X = np.zeros(len(sizes))
        for j in range(0, len(sizes)):
            X[j] = 10**subl_func1(U[i],*fittings[j])
        tau_avg[i] = PAHs.averaged_over(X,sizes)
    
    popt,pcov = curve_fit(subl_func1,U,np.log10(tau_avg))
    print('Fitting for sublimation (Galliano): ',popt)
    ax.plot(U,10**subl_func1(U,popt[0],popt[1],popt[2]),'-',color='k',label='Average over PAHs distribution (Galliano)')
    
    allain_sizes = np.zeros(len(allain_Nc))
    for i in range(0, len(allain_Nc)):
        a = size_from_Nc(allain_Nc[i])
        allain_sizes[i] = a
        tau_subl = Allain_time(U,allain_rates[i])/(3.156e+7)
        ax.plot(U,tau_subl/1e+6,'-.',label='{a:.2f}'.format(a=a)+' $\AA$ Allain+1996')

    tau_avg = np.zeros(100)
    
    for i in range(0,len(tau_avg)):
        X = np.zeros(len(allain_Nc))
        for j in range(0, len(allain_Nc)):
            X[j] =  Allain_time(U[i],allain_rates[j])/(3.156e+7)/1e+6
        tau_avg[i] = PAHs.averaged_over(X,allain_sizes*1e-4)
    popt,pcov = curve_fit(subl_func2,U,np.log10(tau_avg))
    print('Fitting for sublimation (Allain+1996): ',popt)
    ax.plot(U,10**subl_func2(U,popt[0],popt[1]),'-.',color='k',label='Average over PAHs distribution (Allain+1996)')
    
    ax.plot(murga_subl_time['X'],murga_subl_time['Y']/1e6, '-', color='purple', label='5 $\AA$ (Murga+2019)')
    popt,pcov = curve_fit(subl_func3,murga_subl_time['X'],np.log10(murga_subl_time['Y']/1e6))
    print('Fitting for sublimation (Murga+2019): ',popt)
    ax.plot(U,10**subl_func3(U,popt[0],popt[1],popt[2],popt[3]),':',color='purple')

    ax.set_ylabel(r'Sublimation timescale $\tau_{\rm subl}$ [Myr]', fontsize=16)
    ax.set_xlabel(r'Starlight intensity $U$ [$2.2\times 10^{-5}$ W/m$^2$]',fontsize=16)
    ax.set_ylim([1e-10,1e6])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.legend(loc='best',fontsize=8,frameon=False,ncol=2)
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    
    fig.savefig('UV_sublimation_time.png',format='png',dpi=300)
    plt.close(fig)
    
def Granato2021_shattering(a,s,D,n):
    t_sha = np.zeros(len(n))
    for i in range(0, len(n)):
        if n[i]<1:
            p = 1
            t = 54*(a/0.1)*(s/3)*(0.01/D)*n[i]**(-p)
        elif 1<=n[i]<=1e3:
            p = 1/3
            t = 54*(a/0.1)*(s/3)*(0.01/D)*n[i]**(-p)
        else:
            t = 1e9
        t_sha[i] = t
    return t_sha
def plot_shattering_time(D_small,D_large):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')
    
    n = np.logspace(np.log10(1e-4),np.log10(1e3),1000)
    
    t_small = Granato2021_shattering(basic_a0[1],basic_s[1],D_small,n)
    t_large = Granato2021_shattering(basic_a0[2],basic_s[2],D_large,n)
    ax.plot(n,t_small,'-',color='blue',label='Small CDust')
    ax.plot(n,t_large,'-',color='red',label='Large CDust')
    
    ax.set_ylabel(r'Shattering timescale $\tau_{\rm sha}$ [Myr]', fontsize=16)
    ax.set_xlabel(r'$n_{\rm H}$ [cm$^{-3}$]',fontsize=16)
    #ax.set_ylim([1e-10,1e6])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.legend(loc='best',fontsize=12,frameon=False)
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    
    return fig

def Micellotta_rate(fit,T):
    # Fitting function from Micellotta et al. (2010) - Eq 24
    J = fit['a'] + fit['b']*T + fit['c']*T**2 + fit['d']*T**3 + fit['e']*T**4 + fit['f']*T**5
    J = 10**J
    return J
def polynomial_rate(T,a,b,c,d,e,f):
    y = a + b*T + c*T**2 + d*T**3 + e*T**4 + f*T**5
    return y
def polynomial_rate_fix(T,a,b):
    y = a + b*T
    return y
def plot_PAH_sputtering():
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')
    
    T = np.logspace(3,8,1000)
    
    linestyles = ['-','--','-.',':']
    type_collision = ['electrons','H','He','C']
    colours = ['b','r','m']
    
    for i,n_key in enumerate(thermal_spu.keys()):
        data1 = thermal_spu[n_key]
        Nc = int(n_key)
        for j,t_key in enumerate(data1.keys()):
            data2 = data1[t_key]
            J = Micellotta_rate(data2,np.log10(T))
            label = r'N$_C$ = '+str(Nc)+' coll. w/ '+t_key
            ax.plot(T,J,linestyle=linestyles[j],color=colours[i],label=label)
            
    PAHs = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    T_fit = np.logspace(4,np.log10(2e7),1000)
    T_fix = np.logspace(np.log10(2e7),np.log10(1e8),1000)
    T = np.logspace(3,10,1000)
    for t, t_key in enumerate(type_collision):
        J_avg = np.zeros(len(T))
        for i in range(0, len(T)):
            values = np.zeros(len(thermal_spu.keys()))
            sizes = np.zeros(len(thermal_spu.keys()))
            for j,n_key in enumerate(thermal_spu.keys()):
                Nc = int(n_key)
                sizes[j] = size_from_Nc(Nc)
                fits = thermal_spu[n_key][t_key]
                values[j] = Micellotta_rate(fits,np.log10(T_fit[i]))
            J_avg[i] = PAHs.averaged_over(values,sizes*1e-4)
        #ax.plot(T_fit,J_avg,linestyle=linestyles[t],color='k')
        popt,pcov = curve_fit(polynomial_rate,np.log10(T_fit),np.log10(J_avg),
                              bounds=([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
                                      [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
        ax.plot(T[T<2e7],10**polynomial_rate(np.log10(T[T<2e7]),*popt),linestyle=linestyles[t],color='k')
        print(t_key,popt)
        print('Timescale [in s]: '+str(1/(10**polynomial_rate(np.log10(3e8),*popt)*0.1)))
        for i in range(0, len(T_fix)):
            values = np.zeros(len(thermal_spu.keys()))
            sizes = np.zeros(len(thermal_spu.keys()))
            for j,n_key in enumerate(thermal_spu.keys()):
                Nc = int(n_key)
                sizes[j] = size_from_Nc(Nc)
                fits = thermal_spu[n_key][t_key]
                values[j] = Micellotta_rate(fits,np.log10(T_fix[i]))
            J_avg[i] = PAHs.averaged_over(values,sizes*1e-4)
        popt,pcov = curve_fit(polynomial_rate_fix,np.log10(T_fix),np.log10(J_avg))
        print(t_key,popt)
        print('Timescale [in s]: '+str(1/(10**polynomial_rate_fix(np.log10(3e8),*popt)*0.1)))
        ax.plot(T[T>2e7],10**polynomial_rate_fix(np.log10(T[T>2e7]),*popt),linestyle=linestyles[t],color='k')
            
            
    ax.set_ylabel(r'Rate constant $J$ [cm$^3$s$^{-1}$]', fontsize=16)
    ax.set_xlabel(r'$T$ [K]',fontsize=16)
    ax.set_ylim([1e-20,1e-2])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.legend(loc='best',fontsize=10,frameon=False)
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('PAH_thermal_sputtering.png',format='png',dpi=300)
    plt.close(fig)
    
    
def H2_func1(x,a,b,c,d):
    # Power law with flatten core
    return a - b*np.log10(x**d+c)

def plot_H2rate():
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    X = np.logspace(-2,4,1000)
    
    xnames = ['C32_x','C40_x','C50_x','C80_x','C100_x','C120_x','C150_x']
    ynames = ['C32_y','C40_y','C50_y','C80_y','C100_y','C120_y','C150_y']
    Nc = np.array([32,40,50,80,100,120,150])
    fittings = []
    for i in range(0, len(Nc)):
        x = np.asarray(H2_rate_LePage[xnames[i]])
        y = np.asarray(H2_rate_LePage[ynames[i]])
        ax.plot(x[~np.isnan(x)],y[~np.isnan(x)],label=r'$N_{\rm C}=$ '+str(Nc[i]))
        bounds = ([-np.inf,-np.inf,0,-np.inf],np.inf)
        popt,pcov = curve_fit(H2_func1,x[~np.isnan(x)],np.log10(y[~np.isnan(x)]),bounds=bounds)
        fittings.append(popt)

    PAHs = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    sizes = np.array([size_from_Nc(n) for n in Nc])
    R_avg = np.zeros(1000)
    for i in range(0, len(R_avg)):
        r = np.zeros(len(Nc))
        for j in range(0, len(Nc)):
            r[j] = H2_func1(X[i],*fittings[j])
        R_avg[i] = PAHs.averaged_over(r,sizes*1e-4)
    popt,pcov = curve_fit(H2_func1,X,R_avg)
    print('Fitting for H2 formation (LePage+2009): ',popt)
    print(X,10**H2_func1(X,*popt))
    ax.plot(X,10**H2_func1(X,*popt),'k-',label='Average over PAHs distribution')
    
    ax.set_ylabel(r'Rate Coefficient $R_{\rm H_2}$ [cm$^3$/s]', fontsize=16)
    ax.set_xlabel(r'$n_t/\chi$ [$n_t$ in cm$^{-3}$ and $\chi$ in Draine units]',fontsize=16)
    ax.set_ylim([1e-20,1e-16])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.legend(loc='best',fontsize=8,frameon=False,ncol=2)
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    
    fig.savefig('H2_rate_PAHs.png',format='png',dpi=300)
    plt.close(fig)