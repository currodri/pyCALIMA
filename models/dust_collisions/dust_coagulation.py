"""
DUST COAGULATION
"""
# LIBRARIES
import numpy as np
from models.constants import *
from unyt import mh,kb
from models.dust_collisions.dust_dynamics import relative_velocity

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


def plot_coagulation(target_a,projectile_a,target_s,projectile_s,composition,nH,ne,T,nsmall,nMach=100):
    # This function plots the shattering fragment distribution for big grains
    # based on the power-law model 
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    from models.tools.utils import as_si,sigmoid_function
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

    from models.tools.utils import as_si,sigmoid_function
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

    from models.tools.utils import as_si,sigmoid_function
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
    