"""
PAH COALESCENCE
"""
# LIBRARIES
import numpy as np
from models.constants import *


# FUNCTIONS
def pah_coalescence(GDR_PAHs,nMach=100):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    from models.tools.utils import as_si,sigmoid_function
    from scipy.special import erfc
    
    fig2, axes2 = plt.subplots(1,3, figsize=(10,5),dpi=300,facecolor='w',edgecolor='k',sharex=True,sharey=True)
    
    
    phases = {'DC1':{'T':10,'nH':1e4,'ne':0.01,'L':1,'G0':0.01},
              'MC':{'T':25,'nH':300,'ne':0.03,'L':1,'G0':0.1},
              'CNM':{'T':100,'nH':30,'ne':0.0991,'L':0.64,'G0':1.}}
    
    phase_colors = ['indigo','goldenrod','b']
    
    # Add GDR text
    axes2[0].text(0.4, 0.8,r'GDR($a_{\rm PAH}$)'+r'$={0:s}$'.format(as_si(GDR_PAHs,2)),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes2[0].transAxes,fontsize=14)
    
    
    mass_pah = (4./3.) * np.pi * (basic_a0[0]*1e-4)**3. * basic_s[0]
    
    Mach = np.logspace(-1,1,nMach)
    e = 4.8032047e-10 # statC
    kB = 1.380649e-16
    
    # Loop over ISM phases
    for p,phase_name in enumerate(phases):
        ax2 = axes2[p]
        phase = phases[phase_name]
        nH = phase['nH']
        T = phase['T']
        ne = phase['ne']
        Lmax = phase['L']
        G0 = phase['G0']
        
        rho_PAH = nH * mh.to('g').d * (1./GDR_PAHs)
        n_PAH = rho_PAH / mass_pah
        
        t_coal = np.zeros(nMach)
        
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
                
            # Relative velocity given by Brownian motion
            v_brownian = np.sqrt(16. * kb.to('cm**2*g/s**2/K').d * T / mass_pah)
        
            # Compute the PAH charge distribution and Coulomb enhancement
            f,Z = grain_charge_dist(G0,T,ne*boost_coa,'carbonaceous','5A')
            D = 0.
            for j in range(0, len(Z)):
                Zj = Z[j]
                B = 0.
                if Zj != 0:
                    for k in range(0, len(Z)):
                        Zk = Z[k]
                        if Zj*Zk>0:
                            B += f[k] * np.exp(-Zj*Zk*e**2./(kB*T*(basic_a0[0]*1e-4)))
                        elif Zj*Zk<0:
                            B += f[k] * (1.0 - Zj*Zk*e**2./(kB*T*(basic_a0[0]*1e-4)))
                        else:
                            B += f[k] * (1.0 + np.sqrt(np.pi*(Zj**2.)*e**2./(2.0*kB*T*(basic_a0[0]*1e-4))))
                else:
                    B = 1.0
                D += f[j] * B
            D = max(D,1e-10)
            print(n_PAH,v_brownian/1e5)
            t_coal[i] = 1. / (4.*np.pi*(basic_a0[0]*1e-4)**2.*n_PAH*v_brownian*D*boost_coa) / sec2Myr
            
        ax2.plot(Mach,t_coal,linestyle='-',color=phase_colors[p])
        
    for j,k in enumerate(phases):
        ax = axes2[j]
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
        ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)
    
    axes2[0].set_ylabel(r'$t_{\rm coa}(a,\mathcal{M};a_{\rm small},a_{\rm small})$ [Myr]', fontsize=20)
    fig2.subplots_adjust(top=0.98,bottom=0.15,left=0.1,right=0.99,hspace=0,wspace=0)
    fig2.savefig('PAH_coalescence.pdf',format='pdf',dpi=300)
    plt.close(fig2)
    
def compute_coalescence_timescale_Totton12(n_pah,Tk,Nc,a):
    """This function computes the coalescence timescale for PAHs based on the results of Totton et al. (2012).

    Args:
        n_pah (float): PAH number density [cm^-3]
        Tk (float): gas temperature [K]
        Nc (int): number of carbon atoms in the PAH
        a (float): PAH radius [cm]

    Returns:
        float: coalescence timescale [s]
    """
    
    # 1. Compute the reduced mass
    reduced_mass = 0.5 * mass_from_Nc(Nc)
    
    # 2. Compute the relative velocity
    dV_thermal = np.sqrt(8. * kb.to('cm**2*g/s**2/K').d * Tk / reduced_mass)
    
    # 3. Compute the sticking probability based on the fitting to the results of Totton et al. (2012)
    C_eff = 1. / (1. + 9.92807181e-7 * np.log10(Tk)**1.37933821e1)
    
    # 4. Compute the coalescence timescale
    coll_section = 4. * np.pi * a**2.
    t_coal = 1. / (coll_section * dV_thermal * C_eff * n_pah)
    
    return t_coal

def compute_coalescence_timescale_Tielens21(n_pah,Tk,Nc,ionised=False):
    """This function computes the coalescence timescale for PAHs based on the results of Tielens et al. (2021).

    Args:
        n_pah (float): PAH number density [cm^-3]
        Tk (float): gas temperature [K]
        Nc (int): number of carbon atoms in the PAH
        ionised (bool, optional): whether the PAH is ionised. Defaults to False.

    Returns:
        float: coalescence timescale [s]
    """
    
    if not ionised:
        # 1. If neutral grains (taken from Tielens 2021)
        k_coal = 4e-11 * np.sqrt(Tk/10.) * np.sqrt(float(Nc)/50.) # [cm^3/s]
        t_coal = 1. / (k_coal * n_pah) # [s]
    else:
        # 2. If ionised grains (taken from Tielens 2021) we use the Langevin rate
        reduced_mass = 0.5 * mass_from_Nc(Nc)
        k_coal = 6e-9 * np.sqrt(float(Nc)/50.) * np.sqrt((12. * mh.to('g').d)/reduced_mass) # [cm^3/s]
        t_coal = 1. / (k_coal * n_pah) # [s]
        
    return t_coal

def plot_coalescence_timescale(rho_pah,Tmin,Tmax,nT=100):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    # 1. Setup the figure
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(5,4), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$t_{\rm coal}$ [Myr]', fontsize=16)
    ax.set_xlabel(r'$T$ [K]',fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    # 2. Compute the range of temperatures we want
    T = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    Nc = 54 # circumcoronene
    a = basic_a0[0]*1e-4
    
    # 3. Compute the coalescence timescale for Totton et al. (2012)
    n_pah = rho_pah / mass_from_Nc(Nc)
    t_coal_Totton = np.zeros(nT)
    for i in range(0, nT):
        t_coal_Totton[i] = compute_coalescence_timescale_Totton12(n_pah,T[i],Nc,a) / sec2Myr
    ax.plot(T,t_coal_Totton,'-',label='Totton+2012')
    
    # 4. Compute the coalescence timescale for Tielens et al. (2021)
    t_coal_Tielens = np.zeros(nT)
    for i in range(0, nT):
        t_coal_Tielens[i] = compute_coalescence_timescale_Tielens21(n_pah,T[i],Nc) / sec2Myr
    ax.plot(T,t_coal_Tielens,'-',label='Tielens+2021 (Neutral)')
    
    # 5. Compute the coalescence timescale for Tielens et al. (2021) for ionised grains
    t_coal_Tielens_ionised = np.zeros(nT)
    for i in range(0, nT):
        t_coal_Tielens_ionised[i] = compute_coalescence_timescale_Tielens21(n_pah,T[i],Nc,ionised=True) / sec2Myr
    ax.plot(T,t_coal_Tielens_ionised,'-',label='Tielens+2021 (Ionised)')
    
    ax.legend(loc='best',fontsize=12,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.12,left=0.13,right=0.99)
    fig.savefig('PAHSmall_t_coalescence.png',format='png',dpi=300)
    plt.close(fig)

def Totton_efficiency(mu):
    T_values = np.array([500, 750, 1000, 1250, 1500])
    a_values = np.array([0.5074, 0.6822, 0.8032, 0.8425, 0.8858])
    b_values = np.array([54.13, 190.0, 441.3, 714.2, 1322])
    
    C = 1. + mu / (a_values*mu+b_values) - 1. / a_values
    
    return T_values, C

def logistic_curve(x,a,b):
    
    return 1./(1.+a*x**b)
def plot_pah_coalescence(Tmin,Tmax,nT):
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(6,5),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$T$ [K]', fontsize=20)
    ax.set_ylabel(r'$k_{\rm coal}/n_{\rm smallPAHs}^2$ [cm$^{3}$ s$^{-1}$]', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([Tmin,Tmax])
    
    T = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    smallPAHs = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    # 2. Neutral PAH coalescence rate by Tielens 2021
    k = 4e-11 * np.sqrt(T/10.) * np.sqrt(54./50.)
    ax.plot(T,k,linestyle='--',color='royalblue',linewidth=2.5,label=r'Neutral smallPAHs (Tielens 2021)')
    
    # 3. Ionised PAH coalescence rate by Tielens 2021
    k = 6e-9 * np.sqrt(54./50.) * np.sqrt(12./(12.*54+18.))
    ax.hlines(k,Tmin,Tmax,linestyles=':',color='cornflowerblue',linewidth=2.5,label=r'Ionised smallPAHs (Tielens 2021)')
    
    # 4. Neutral PAH coalescence rate as given by kinetic theory with 
    # the sticking probability by Totton et al. (2012)
    reduced_mass = 0.5 * (12.011*54. + 1.00784*18)
    T_vals,C_vals = Totton_efficiency(reduced_mass)
    params_C, _ = curve_fit(logistic_curve, np.log10(T_vals), C_vals, maxfev=10000)
    print('Logistic curve parameters: ',params_C)
    C = logistic_curve(np.log10(T),*params_C)
    reduced_mass = reduced_mass * 1.660538921e-24
    dV_thermal = np.sqrt(8. * kb.to('cm**2*g/s**2/K').d * T / reduced_mass)
    sigma = np.pi * (2.*smallPAHs.a0*1e-4)**2.
    k = sigma * dV_thermal * C
    ax.plot(T,k,linestyle='-',color='steelblue',linewidth=2.5,label=r'Neutral small PAHs (Totton et al. 2012)')
    
    ax.legend(loc='best', frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.98,bottom=0.112,left=0.15,right=0.98,hspace=0,wspace=0)
    fig.savefig('small_pah_coalescence.pdf',format='pdf',dpi=300)
    plt.close(fig)