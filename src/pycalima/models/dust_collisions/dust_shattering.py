"""
DUST SHATTERING

"""
# LIBRARIES
import numpy as np
from pycalima.models.constants import *
from unyt import mh,kb


def t_shattering_Dubois2024(Dbig,nH,a,s):
    
    local_mu = 1.4
    
    t_sha = 75.73 * (0.01/Dbig) * (a/1.e-5) * (s/3.)
    if nH < 1:
        t_sha = t_sha * (1./(nH*local_mu))
    elif 1 <= nH <= 1e3:
        t_sha = t_sha * (1./nH)**(1./3.) / local_mu
    else:
        t_sha = 1e9
    return t_sha

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

    from pycalima.models.tools.utils import as_si
    from pycalima.models.dust_collisions.dust_dynamics import relative_velocity
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
    ax.hlines(t_shattering_Dubois2024(rho_projectile/(nH*mh.to('g').d),nH,target_a,target_s),0.1,10,color='k')
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('shattering_timescale_%s.png'%(composition),format='png',dpi=300)
    plt.close(fig)
    
def plot_shattering_frag_full(GDR_small,GDR_big,nMach=100):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })

    from pycalima.models.tools.utils import as_si
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
                        M_dest[i] += M_vsmall[i]
                        M_vsmall[i] = 0.0
                        
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
                ax2.hlines(t_shattering_Dubois2024(rho_projectile/(nH*mh.to('g').d),nH,target_a,target_s),0.1,10,color=colors[index][1])
    
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

def plot_shattering_frag_simple(nMach=100):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })

    from pycalima.models.tools.utils import as_si
    from pycalima.models.dust_collisions.dust_dynamics import relative_velocity
    fig, axes = plt.subplots(1,2, figsize=(10,4),dpi=300,facecolor='w',edgecolor='k',sharex=True,sharey=True)
    
    # Shattering model quantities (Kobayashi & Tanaka 2010)
    Q_D = [8.9e9,4.3e10]
    alpha_f = 3.3

    phase_details = {'T':8000,'nH':0.1,'ne':0.0991,'L':100}
    compositions = ['Carbonaceous']
    comp_simple_name = ['C','Sil']
    collision_model = {'Carbonaceous':{'Big-Big':{'target_a':0.1e-4,'projectile_a':0.1e-4},
                       'Big-Small':{'target_a':0.1e-4,'projectile_a':0.01e-4}},
                       'Silicates':{'Big-Big':{'target_a':0.1e-4,'projectile_a':0.1e-4},
                                    'Big-Small':{'target_a':0.1e-4,'projectile_a':0.005e-4}}}
    colors = [['lightsteelblue','blue','royalblue','steelblue','cornflowerblue'],
              ['chocolate','saddlebrown','sandybrown']]
    
    
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
        for m,model_name in enumerate(collision_model[comp]):
            model = collision_model[comp][model_name]
            target_a = model['target_a']
            projectile_a = model['projectile_a']
        
            # Grain quantities
            mass_target = 4./3. * np.pi * target_s * (target_a**3.)
            mass_projectile = 4./3. * np.pi * projectile_s * (projectile_a**3.)
            
            Mach = np.logspace(-1,1,nMach)
            
            # Limits of the distributions
            if comp == 'Carbonaceous':
                m_dest_min = 4./3. * np.pi * target_s * ((shattering_a0[0]*1e-4)**3.)
                m_vvsmall_min = 4./3. * np.pi * target_s * ((shattering_amin[0]*1e-4)**3.)
                m_vvsmall_max = 4./3. * np.pi * target_s * ((shattering_amax[0]*1e-4)**3.)
                m_vsmall_min = 4./3. * np.pi * target_s * ((shattering_amin[1]*1e-4)**3.)
                m_vsmall_max = 4./3. * np.pi * target_s * ((shattering_amax[1]*1e-4)**3.)
                m_small_min = 4./3. * np.pi * target_s * ((shattering_amin[2]*1e-4)**3.)
                m_small_max = 4./3. * np.pi * target_s * ((shattering_amax[2]*1e-4)**3.)
                m_big_min = 4./3. * np.pi * target_s * ((shattering_amin[3]*1e-4)**3.)
                m_big_max = 4./3. * np.pi * target_s * ((shattering_amax[3]*1e-4)**3.)
            elif comp == 'Silicates':
                m_dest_min = 4./3. * np.pi * target_s * ((shattering_a0[4]*1e-4)**3.)
                m_small_min = 4./3. * np.pi * target_s * ((shattering_amin[5]*1e-4)**3.)
                m_small_max = 4./3. * np.pi * target_s * ((shattering_amax[5]*1e-4)**3.)
                m_big_min = 4./3. * np.pi * target_s * ((shattering_amin[6]*1e-4)**3.)
                m_big_max = 4./3. * np.pi * target_s * ((shattering_amax[6]*1e-4)**3.)

            exponent1 = (4.-alpha_f)/3.

            ax = axes[m]
            nH = phase_details['nH']
            T = phase_details['T']
            ne = phase_details['ne']
            Lmax = phase_details['L']
            
            
            M_dest = np.zeros((nMach,3))
            M_vvsmall = np.zeros((nMach,3))
            M_vsmall = np.zeros((nMach,3))
            M_small = np.zeros((nMach,3))
            M_big = np.zeros((nMach,3))
            
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

                    # 2. Destruction to very very small grains (only for Carbonaceous)
                    if comp == 'Carbonaceous':
                        if m_min[j] >= m_vvsmall_max or m_max[j] < m_vvsmall_min:
                            M_vvsmall[i,j] =  0.0
                            if m_vvsmall_min <= m_remnant[j] < m_vvsmall_max:
                                M_vvsmall[i,j] += m_remnant[j]
                        else:
                            M_vvsmall[i,j] = prefactor[j] * (min(m_vvsmall_max,m_max[j])**exponent1-max(m_vvsmall_min,m_min[j])**exponent1)
                            if m_vvsmall_min <= m_remnant[j] < m_vvsmall_max:
                                M_vvsmall[i,j] += m_remnant[j]
                    
                    # 2. Destruction to very small grains (only for Carbonaceous)
                    if comp == 'Carbonaceous':
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
                M_tot = (M_dest[i] + M_vvsmall[i] + M_vsmall[i] + M_small[i] + M_big[i])
                
                M_dest[i] = M_dest[i]/M_tot
                M_vvsmall[i] = M_vvsmall[i]/M_tot
                M_vsmall[i] = M_vsmall[i]/M_tot
                M_small[i] = M_small[i]/M_tot
                M_big[i] = M_big[i]/M_tot
                                      
            selected = (M_dest[:,0]>0.) & (M_dest[:,1]>0.)
            if m == 1 and comp == 'Silicates':
                ax.fill_between(Mach,M_dest[:,0],M_dest[:,1],label='Destroyed '+comp_simple_name[index],alpha=0.5,where=(selected),color=colors[index][0])
                ax.fill_between(Mach,M_small[:,0],M_small[:,1],label='small'+comp_simple_name[index],alpha=0.5,where=M_small[:,1]>0.,color=colors[index][1])
                ax.fill_between(Mach,M_big[:,0],M_big[:,1],label='large'+comp_simple_name[index],alpha=0.5,where=M_big[:,1]>0.,color=colors[index][2])
            elif m == 0 and comp == 'Carbonaceous':
                ax.fill_between(Mach,M_dest[:,0],M_dest[:,1],label='Destroyed '+comp_simple_name[index],alpha=0.5,where=(selected),color=colors[index][0])
                ax.fill_between(Mach,M_vvsmall[:,0],M_vvsmall[:,1],label='smallPAH',alpha=0.5,where=M_vvsmall[:,1]>0.,color=colors[index][1])
                ax.fill_between(Mach,M_vsmall[:,0],M_vsmall[:,1],label='largePAH',alpha=0.5,where=M_vsmall[:,1]>0.,color=colors[index][2])
                ax.fill_between(Mach,M_small[:,0],M_small[:,1],label='small'+comp_simple_name[index],alpha=0.5,where=M_small[:,1]>0.,color=colors[index][3])
                ax.fill_between(Mach,M_big[:,0],M_big[:,1],label='large'+comp_simple_name[index],alpha=0.5,where=M_big[:,1]>0.,color=colors[index][4])
            else:
                ax.fill_between(Mach,M_dest[:,0],M_dest[:,1],alpha=0.5,where=(selected),color=colors[index][0])
                if comp == 'Carbonaceous':
                    ax.fill_between(Mach,M_vvsmall[:,0],M_vvsmall[:,1],alpha=0.5,where=M_vvsmall[:,1]>0.,color=colors[index][1])
                    ax.fill_between(Mach,M_vsmall[:,0],M_vsmall[:,1],alpha=0.5,where=M_vsmall[:,1]>0.,color=colors[index][2])
                    ax.fill_between(Mach,M_small[:,0],M_small[:,1],alpha=0.5,where=M_small[:,1]>0.,color=colors[index][3])
                    ax.fill_between(Mach,M_big[:,0],M_big[:,1],alpha=0.5,where=M_big[:,1]>0.,color=colors[index][4])
                else:
                    ax.fill_between(Mach,M_small[:,0],M_small[:,1],alpha=0.5,where=M_small[:,1]>0.,color=colors[index][1])
                    ax.fill_between(Mach,M_big[:,0],M_big[:,1],alpha=0.5,where=M_big[:,1]>0.,color=colors[index][2])         
                
            ax.plot(Mach,M_dest[:,2],linestyle='-',color=colors[index][0],lw=2)
            if comp == 'Carbonaceous':
                ax.plot(Mach,M_vvsmall[:,2],linestyle='-',color=colors[index][1],lw=2)
                ax.plot(Mach,M_vsmall[:,2],linestyle='-',color=colors[index][2],lw=2)
                ax.plot(Mach,M_small[:,2],linestyle='-',color=colors[index][3],lw=2)
                ax.plot(Mach,M_big[:,2],linestyle='-',color=colors[index][4],lw=2)
            else:
                ax.plot(Mach,M_small[:,2],linestyle='-',color=colors[index][1],lw=2)
                ax.plot(Mach,M_big[:,2],linestyle='-',color=colors[index][2],lw=2)
                    
    # Add legend of composition/size type
    ax = axes[0]
    first_legend = ax.legend(loc='lower right', fontsize=12,frameon=False)
    ax.add_artist(first_legend)
    
    # Add legend of velocity model type
    ax = axes[1]
    first_legend = ax.legend(loc='upper left', fontsize=12,frameon=False)
    ax.add_artist(first_legend)
    dummy_lines = []

    dummy_lines.append(ax.fill_between([],[], color="black",label = 'Hirashita and Aoyama 2019'))
    dummy_lines.append(ax.plot([],[], color="black", ls = '-',label = 'Ormel and Cuzzi 2007')[0])
    second_legend = ax.legend(handles=dummy_lines, loc='upper left', frameon=False, fontsize=12,ncol=1)         
    ax.add_artist(second_legend)
    
    # Setup axes
    for i in range(0,2):
        ax = axes[i]
        ax.tick_params(labelsize=14)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-4,1])
        ax.set_xlabel(r'$\mathcal{M}$',fontsize=16)

    # Setup y-labels
    axes[0].set_ylabel(r'$\chi_{\rm frag}(a,\mathcal{M};a_{\rm L},a_{\rm L})$', fontsize=20)
    # Set this y label on the right
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_ticks_position('right')
    axes[1].set_ylabel(r'$\chi_{\rm frag}(a,\mathcal{M};a_{\rm L},a_{\rm S})$', fontsize=20)

    fig.subplots_adjust(top=0.97,bottom=0.12,left=0.08,right=0.92,hspace=0,wspace=0)
    fig.savefig('shattering_efficiency_WIM.pdf',format='pdf',dpi=300)
    plt.close(fig)
    