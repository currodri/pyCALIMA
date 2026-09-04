"""
PAH FREEZING

"""
# LIBRARIES
import numpy as np
from models.constants import *


# FUNCTIONS
def pah_freezing(GDR_PAHs,GDR_small,GDR_large,nMach=100):
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
    
    fig2, axes2 = plt.subplots(1,3, figsize=(10,5),dpi=300,facecolor='w',edgecolor='k',sharex=True,sharey=True)
    
    
    phases = {'DC1':{'T':10,'nH':1e4,'ne':0.01,'L':1,'G0':0.01},
              'MC':{'T':25,'nH':300,'ne':0.03,'L':1,'G0':0.1},
              'CNM':{'T':100,'nH':30,'ne':0.0991,'L':0.64,'G0':1.}}
    
    phase_colors = ['indigo','goldenrod','b']
    
    # Add GDR text
    axes2[0].text(0.4, 0.8,r'GDR($a_{\rm PAH}$)'+r'$={0:s}$'.format(as_si(GDR_PAHs,0))+'\n'+
                            r'GDR($a_{\rm small}$)'+r'$={0:s}$'.format(as_si(GDR_small,0))+'\n'+
                            r'GDR($a_{\rm large}$)'+r'$={0:s}$'.format(as_si(GDR_small,0)),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes2[0].transAxes,fontsize=14)
    
    
    mass_pah = (4./3.) * np.pi * (basic_a0[0]*1e-4)**3. * basic_s[0]
    mass_small = (4./3.) * np.pi * (basic_a0[1]*1e-4)**3. * basic_s[1]
    mass_large = (4./3.) * np.pi * (basic_a0[2]*1e-4)**3. * basic_s[2]
    
    Mach = np.logspace(-1,1,nMach)
    e = 4.8032047e-10 # statC
    kB = 1.380649e-16
    eV = 1.602176634e-12 # erg
    
    t_max = np.log10(1e5)
    
    # Loop over ISM phases
    for p,phase_name in enumerate(phases):
        ax2 = axes2[p]
        phase = phases[phase_name]
        nH = phase['nH']
        T = phase['T']
        ne = phase['ne']
        Lmax = phase['L']
        G0 = phase['G0']
        
        rho_small = nH * mh.to('g').d * (1./GDR_small)
        n_small = rho_small / mass_small
        rho_large = nH * mh.to('g').d * (1./GDR_large)
        n_large = rho_large / mass_large

        
        t_coal = np.zeros((nMach,4))
        
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
                
            # Average relative velocity
            v_rel_avg = relative_velocity('Ormel and Cuzzi2007',T,nH,ne,Mach[i],Lmax,basic_a0[1]*1e-4,basic_a0[0]*1e-4,
                                                basic_s[1],basic_s[0])
        
            # Compute the PAH charge distribution and Coulomb enhancement
            f_PAH,Z_PAH = grain_charge_dist(G0,T,ne*boost_coa,'carbonaceous','5A')
            f_small,Z_small = grain_charge_dist(G0,T,ne*boost_coa,'carbonaceous','50A')
            D = 0.
            for j in range(0, len(Z_PAH)):
                Zj = Z_PAH[j]
                B = 0.
                if Zj != 0:
                    for k in range(0, len(Z_small)):
                        Zk = Z_small[k]
                        if Zj*Zk>0:
                            B += f_small[k] * np.exp(-Zj*Zk*e**2./(kB*T*(basic_a0[1]*1e-4)))
                        elif Zj*Zk<0:
                            B += f_small[k] * (1.0 - Zj*Zk*e**2./(kB*T*(basic_a0[1]*1e-4)))
                        else:
                            B += f_small[k] * (1.0 + np.sqrt(np.pi*(Zj**2.)*e**2./(2.0*kB*T*(basic_a0[1]*1e-4))))
                else:
                    B = 1.0
                D += f_PAH[j] * B
            D = max(D,1e-10)
            E_col = 0.5 * (mass_pah*mass_small)/(mass_pah+mass_small)*v_rel_avg**2./eV
            t = np.log10(1. / (np.pi*((basic_a0[0]+basic_a0[1])*1e-4)**2.*n_small*v_rel_avg*D*boost_coa) / sec2Myr)
            t_coal[i,0] = (1-sigmoid_function(t_max,1.0,E_col))*t + sigmoid_function(t_max,1.,E_col) * t_max
            t_coal[i,0] = 10**(t_coal[i,0])
            t = np.log10(1. / (np.pi*((basic_a0[0]+basic_a0[1])*1e-4)**2.*n_small*v_rel_avg*boost_coa) / sec2Myr)
            t_coal[i,1] = (1-sigmoid_function(t_max,1.0,E_col))*t + sigmoid_function(t_max,1.,E_col) * t_max
            t_coal[i,1] = 10**(t_coal[i,1])
            
            # Average relative velocity
            v_rel_avg = relative_velocity('Ormel and Cuzzi2007',T,nH,ne,Mach[i],Lmax,basic_a0[2]*1e-4,basic_a0[0]*1e-4,
                                                basic_s[2],basic_s[0])
            f_large,Z_large = grain_charge_dist(G0,T,ne*boost_coa,'carbonaceous','1000A')
            D = 0.
            for j in range(0, len(Z_PAH)):
                Zj = Z_PAH[j]
                B = 0.
                if Zj != 0:
                    for k in range(0, len(Z_large)):
                        Zk = Z_large[k]
                        if Zj*Zk>0:
                            B += f_large[k] * np.exp(-Zj*Zk*e**2./(kB*T*(basic_a0[2]*1e-4)))
                        elif Zj*Zk<0:
                            B += f_large[k] * (1.0 - Zj*Zk*e**2./(kB*T*(basic_a0[2]*1e-4)))
                        else:
                            B += f_large[k] * (1.0 + np.sqrt(np.pi*(Zj**2.)*e**2./(2.0*kB*T*(basic_a0[2]*1e-4))))
                else:
                    B = 1.0
                D += f_PAH[j] * B
            D = max(D,1e-10)
            E_col = 0.5 * (mass_pah*mass_large)/(mass_pah+mass_large)*v_rel_avg**2./eV
            t = np.log10(1. / (np.pi*((basic_a0[0]+basic_a0[2])*1e-4)**2.*n_large*v_rel_avg*D*boost_coa) / sec2Myr)
            t_coal[i,2] = (1-sigmoid_function(t_max,1.0,E_col))*t + sigmoid_function(t_max,1.,E_col) * t_max
            t_coal[i,2] = 10**(t_coal[i,2])
            t = np.log10(1. / (np.pi*((basic_a0[0]+basic_a0[2])*1e-4)**2.*n_large*v_rel_avg*boost_coa) / sec2Myr)
            t_coal[i,3] = (1-sigmoid_function(t_max,1.0,E_col))*t + sigmoid_function(t_max,1.,E_col) * t_max
            t_coal[i,3] = 10**(t_coal[i,3])
            
        ax2.plot(Mach,t_coal[:,0],linestyle='-',color=phase_colors[p],label = 'Small+Charge')
        ax2.plot(Mach,t_coal[:,1],linestyle='--',color=phase_colors[p],label = 'Small')
        ax2.plot(Mach,t_coal[:,2],linestyle=':',color=phase_colors[p],label = 'Large+Charge')
        ax2.plot(Mach,t_coal[:,3],linestyle='-.',color=phase_colors[p],label = 'Large')
        
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
    
    axes2[1].legend(loc='best',fontsize=10,frameon=False)
    
    axes2[0].set_ylabel(r'$t_{\rm free}(a,\mathcal{M};a_{\rm small},a_{\rm small})$ [Myr]', fontsize=20)
    fig2.subplots_adjust(top=0.98,bottom=0.15,left=0.1,right=0.99,hspace=0,wspace=0)
    fig2.savefig('PAH_freezing.pdf',format='pdf',dpi=300)
    plt.close(fig2)