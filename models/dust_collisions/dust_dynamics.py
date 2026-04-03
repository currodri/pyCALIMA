"""
DUST DYNAMICS
"""

# LIBRARIES
import numpy as np
from astropy.constants import k_B as kb, m_p as mh


# FUNCTIONS
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
    sns.set_theme(style="white")
    
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
  