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
        # This is based on the formulation presented in Kawasaki & Machida (2023)
        # which is basically the analytical model of Ormel & Cuzzi (2007)
        mass_target = 4./3. * np.pi * target_s * (target_a**3.)
        mass_projectile = 4./3. * np.pi * projectile_s * (projectile_a**3.)
        dV_thermal = np.sqrt(8. * kb.to('cm**2*g/s**2/K').d * T * (mass_projectile+mass_target)/(np.pi*mass_target*mass_projectile))
        rho_gas = nH * mh.to('g').d * local_mu
        cs_gas = np.sqrt(gamma_gas * kb.to('cm**2*g/s**2/K').d * T / (mh.to('g').d*local_mu))
        # Maxwellian mean speed of the gas particles (no adiabatic index)
        v_th = np.sqrt(8. * kb.to('cm**2*g/s**2/K').d * T / (np.pi * mh.to('g').d * local_mu))
        v_turb = Mach * cs_gas
        
        # Assume that the injection scale is a cell size of Lmax pc and the velocity is given by
        # the largest size eddie velocity
        
        tau_L = Lmax / v_turb
        # Kinematic viscosity of the neutral + ionised (fraction ne / nH) mixture,
        # each weighted by its mass fraction: nu = sum_s f_s v_th mfp_s / 3, with
        # neutral collisions at 2e-15 cm^2 and ion-ion Coulomb collisions
        # (Braginskii 1965, ln Lambda = 20)
        x_ion = min(ne / nH, 1.0)
        rc = e**2 / (kb.to('cm**2*g/s**2/K').d * T)
        sigma_C = np.pi * rc**2. * 20.
        nu = v_th / (3. * nH) * ((1. - x_ion) / 2e-15 + x_ion / (x_ion * sigma_C + (1. - x_ion) * 2e-15))
        Re = max(v_turb * Lmax / nu, 1.)
        
        tau_eta = tau_L / np.sqrt(Re)
        
        # Stopping time computation for target and projectile
        # Epstein's law
        ts_target = target_s * target_a / (rho_gas*v_th)            
        ts_projectile = projectile_s * projectile_a / (rho_gas*v_th)        

        # Compute the Stokes' numbers for both particles
        St_target = ts_target / tau_L
        St_projectile = ts_projectile / tau_L
        
        # And finally the turbulent relative velocity: the full closed form of
        # Ormel & Cuzzi (2007, Eqs. 16-18), continuous across the tightly
        # coupled, intermediate and heavy regimes, with the class I / II
        # boundary t* = 1.6 t_s (t* = t_L once t_s >= t_L, t_eta <= t* <= t_L)
        St1, St2 = max(St_target, St_projectile), min(St_target, St_projectile)
        Stmin = tau_eta / tau_L
        St_star = min(max(1.6 * St1, Stmin), 1.) if St1 < 1. else 1.
        dV_I2 = (St1 - St2) / (St1 + St2) * (St1**2. / (St_star + St1) - St1**2. / (1. + St1)
                                             - St2**2. / (St_star + St2) + St2**2. / (1. + St2))
        dV_II2 = (2. * (St_star - Stmin) + St1**2. / (St1 + St_star) - St1**2. / (St1 + Stmin)
                  + St2**2. / (St2 + St_star) - St2**2. / (St2 + Stmin))
        dV_turb = np.sqrt(3./2.) * v_turb * np.sqrt(max(dV_I2 + dV_II2, 0.))
        v_rel = np.sqrt(dV_thermal**2. + dV_turb**2.)
        
    elif model == 'Hirashita and Aoyama2019':
        v_target = 1.1e5 * (Mach**(3./2.)) * np.sqrt(target_a/1e-5) * ((T/1e4)**(1./4.)) * (nH**(-1./4.)) * np.sqrt(target_s/3.5)
        v_projectile = 1.1e5 * (Mach**(3./2.)) * np.sqrt(projectile_a/1e-5) * ((T/1e4)**(1./4.)) * (nH**(-1./4.)) * np.sqrt(projectile_s/3.5)
        # RMS over a random collision angle, (v_t^2 + v_p^2)^1/2, as the solver
        # (solvers.grain_dynamics) and RAMSES (random cos theta) do
        v_rel = np.sqrt(v_target**2. + v_projectile**2.)
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
  