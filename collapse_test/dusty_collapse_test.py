""""
DUSTY COLLAPSE - TEST FOR RAMSES-RTZDust

The routines here presented are of interest for the running of a
dusty stellar collapse phase.

By: Curro Rodriguez Montero (currodri@gmail.com)
"""

# Import libraries
import numpy as np
import sys
from unyt import unyt_array
from unyt import G,g,cm,s,mh,K,kb,amu,yr
import matplotlib.pyplot as plt

init_fdust = 1e-8

# Functions

class omukai_model(object):
    """
    Class that holds a generalised version of the Okumai+05 model
    """
    def __init__(self,nH,T,gamma):
        self.nH = nH / (cm**3)
        self.T = T * K
        self.gamma = gamma
        self.f = self.get_f()
        self.tcol_0 = self.get_tcol_0()
        self.tcol = self.tcol_0 / np.sqrt(1.-self.f)

    def get_f(self):
        if self.gamma < 0.83:
            f = 0.0
        elif 0.83 < self.gamma < 1:
            f = 0.6+2.5*(self.gamma-1)-6.0*(self.gamma-1)**2
        elif self.gamma > 1:
            f = 1.0 + 0.2*(self.gamma-4./3.) - 2.9*(self.gamma-4./3.)**2
        return f
    def get_tcol_0(self):
        tcol_0 = 3.*np.pi / (32.*G*self.nH*mh)
        tcol_0 = np.sqrt(tcol_0)
        return tcol_0

    def get_ramses_params(self,boxsize):
        scale_d = mh / cm**3
        scale_l = boxsize*cm
        scale_t = self.tcol
        d_region = self.nH * mh / scale_d
        p_region = self.T * self.nH * (kb*scale_t**2.) / (scale_d * scale_l**2.)

        print('')
        print('YOU NEED THESE PARAMETERS FOR YOUR NAMELIST:')
        print('unit_density=%.5e ! gram per proton'%scale_d.to('g/cm**3'))
        print('unit_time=%.5e ! collapse time in seconds'%scale_t.to('s'))
        print('unit_length=%.5e ! boxsize in cm'%scale_l.to('cm'))
        print('')
        print('d_region=%.5e ! H atom per cm3'%d_region.d)
        print('p_region=%.5e ! %.1f K'%(p_region.d,self.T))

def read_log(logpath, plot=True):
    
    read_in = False
    raw_data = []
    with open(logpath) as file:
        for line in file:
            if "seconds" in line:
                read_in = False
            if read_in:
                raw_data.append(line)
            if "Starting Omukai" in line:
                read_in = True
            

    variables = ['time','nH','density','nCO','H2','PAHs','smallC','largeC','smallSil','largeSil']
    labels = {'PAHs': 'PAHs (5 $\AA$)','smallC':r'Small Carbonaceous (5 nm)','largeC':r'Large Carbonaceous (0.1 $\mu$m)',
              'smallSil':r'Small Silicates (5 nm)','largeSil':r'Large Silicates (0.1 $\mu$m)'}
    collapse_data = np.zeros((len(raw_data)-2,len(variables)))

    for i in range(1, len(raw_data)-1):
        line = raw_data[i][1:-1].split(" ")
        if len(line) > 2:
            for j in range(0, len(variables)):
                collapse_data[i - 1, j] = float(line[j])

    if plot:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8,5), dpi=100, facecolor='w', edgecolor='k')

        ax = axes[0]
        ax.set_ylabel(r'$x_{\rm dust}$', fontsize=16)
        ax.set_yscale('log')
        ax.set_xlim([600,collapse_data[-1,0]])
        ax.tick_params(labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")

        for i in range(4,len(variables)):
            ax.plot(collapse_data[:,0], collapse_data[:,i]*init_fdust, label=variables[i])
        ax.legend(loc='best', fontsize=14,frameon=False,ncol=2)

        ax = axes[1]
        ax.set_xlabel(r'$t$ [Myr]', fontsize=16)
        ax.set_ylabel(r'$x_{{\rm H}_2}$', fontsize=16)
        ax.set_yscale('log')
        ax.set_xlim([600,collapse_data[-1,0]])
        ax.tick_params(labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.plot(collapse_data[:,0], collapse_data[:,4]*0.1, label=variables[i])

        fig.savefig('cloud_time_evo_'+str(logpath.split('.log')[0])+'.png', format='png', dpi=200)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6,5), dpi=100, facecolor='w', edgecolor='k')

        ax = axes[0]
        ax.set_ylabel(r'$x_{\rm dust}$', fontsize=16)
        ax.set_yscale('log')
        # ax.set_ylim([0,1])
        ax.set_xscale('log')
        ax.tick_params(labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")

        for i in range(5,len(variables)):
            ax.plot(collapse_data[:,1]*0.1, collapse_data[:,i]*init_fdust, label=labels[variables[i]])
        ax.legend(loc='best', fontsize=10,frameon=False)

        ax = axes[1]
        
        ax.set_ylabel(r'$x_{{\rm H}_2}$', fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.plot(collapse_data[:,1]*0.1, collapse_data[:,4],color='olive',linestyle='--')

        ax = axes[2]
        ax.set_ylabel(r'$X_{\rm CO}$', fontsize=16)
        ax.set_xlabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=16)
        ax.set_ylim([1e-8,1e-2])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        # ax.plot(collapse_data[:,1]*0.1, collapse_data[:,2],label='nCarbon')
        # ax.plot(collapse_data[:,1]*0.1, collapse_data[:,3],label='nOxygen')
        ax.plot(collapse_data[:,1]*0.1, collapse_data[:,3],color='m',linestyle='-.')
        
        fig.subplots_adjust(top=0.99,bottom=0.11,left=0.12,right=0.99,hspace=0)
        fig.savefig('cloud_collapse_'+str(logpath.split('.log')[0])+'.png', format='png', dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=100, facecolor='w', edgecolor='k')

        ax.set_ylabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=16)
        ax.set_xlabel(r'$t$ [Myr]', fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.plot(collapse_data[:,0], collapse_data[:,1]*0.1)
        fig.savefig('density_collapse_'+str(logpath.split('.log')[0])+'.png', format='png', dpi=200)
        plt.close(fig)
        
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=100, facecolor='w', edgecolor='k')

        ax.set_xlabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=16)
        ax.set_ylabel(r'$n_{\rm C}^{\rm depl}/n_{\rm H}$', fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        total_depletion = np.zeros(len(collapse_data[:,1]))
        for i in range(5,len(variables)-2):
            depletion = (collapse_data[:,i]*init_fdust)*collapse_data[:,2]/(12*mh.to('g').d)/(collapse_data[:,1]*0.1)
            total_depletion = total_depletion + depletion
            ax.plot(collapse_data[:,1]*0.1, depletion, label=labels[variables[i]])
        ax.plot(collapse_data[:,1]*0.1, total_depletion,'k--',label='Total dust')
        total_carbon = 0.5*1.096750699913960e-2
        ax.plot([collapse_data[0,1]*0.1,collapse_data[-1,1]*0.1], [total_carbon,total_carbon],'k-',label='Half Total carbon',alpha=0.6)
        ax.legend(loc='best', fontsize=10,frameon=False)
        fig.subplots_adjust(top=0.99,bottom=0.11,left=0.14,right=0.99,hspace=0)
        fig.savefig('Carbon_depletion_'+str(logpath.split('.log')[0])+'.png', format='png', dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=100, facecolor='w', edgecolor='k')

        ax.set_xlabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=16)
        ax.set_ylabel(r'GDR', fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.set_ylim([10,1e5])
        ax.tick_params(which='both',axis="both",direction="in")
        total_GDR = np.zeros(len(collapse_data[:,1]))
        for i in range(5,len(variables)):
            dust_density = (collapse_data[:,i]*init_fdust)*collapse_data[:,2]
            total_GDR = total_GDR + dust_density
            ax.plot(collapse_data[:,1]*0.1, collapse_data[:,2]/dust_density, label=labels[variables[i]])
        ax.plot(collapse_data[:,1]*0.1, collapse_data[:,2]/total_GDR,'k--',label='Total dust')
        ax.legend(loc='best', fontsize=10,frameon=False)
        fig.subplots_adjust(top=0.99,bottom=0.11,left=0.14,right=0.99,hspace=0)
        fig.savefig('GDR_'+str(logpath.split('.log')[0])+'.png', format='png', dpi=300)
        plt.close(fig)
    return collapse_data

def compare_log(logpath1, logpath2, plot=True):
    
    read_in = False
    raw_data1 = []
    with open(logpath1) as file:
        for line in file:
            if "seconds" in line:
                read_in = False
            if read_in:
                raw_data1.append(line)
            if "Starting Omukai" in line:
                read_in = True
            

    variables = ['time','density','H2','smallC','largeC','smallSil','largeSil']
    
    collapse_data1 = np.zeros((len(raw_data1)-2,len(variables)))

    for i in range(1, len(raw_data1)-1):
        line = raw_data1[i][1:-1].split(" ")
        if len(line) > 2:
            for j in range(0, len(variables)):
                collapse_data1[i - 1, j] = float(line[j])

    raw_data2 = []
    with open(logpath2) as file:
        for line in file:
            if "seconds" in line:
                read_in = False
            if read_in:
                raw_data2.append(line)
            if "Starting Omukai" in line:
                read_in = True
            

    variables = ['time','density','H2','smallC','largeC','smallSil','largeSil']
    
    collapse_data2 = np.zeros((len(raw_data2)-2,len(variables)))

    for i in range(1, len(raw_data2)-1):
        line = raw_data2[i][1:-1].split(" ")
        if len(line) > 2:
            for j in range(0, len(variables)):
                collapse_data2[i - 1, j] = float(line[j])

    if plot:

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=100, facecolor='w', edgecolor='k')

        ax.set_ylabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=16)
        ax.set_xlabel(r'$t$ [Myr]', fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.plot(collapse_data1[:,0], collapse_data1[:,1],label=logpath1.split('.log')[0])
        ax.plot(collapse_data2[:,0], collapse_data2[:,1],label=logpath2.split('.log')[0],linestyle=':')
        ax.legend(loc='best', fontsize=12,frameon=False)
        fig.savefig('density_collapse_compare.png', format='png', dpi=200)
        plt.close()
    return collapse_data1,collapse_data2


def run_collapse(zinit,start_nH=0.1,end_nH=1e+6,start_fdust=1e-9):
    """
    Routine that evolves the Omukai+05 model to a final density
    and the simultaneous growth of metal mass to keep the same
    metallicity.
    """
    ndust = 2
    mf_asplund09 = {'Fe':0.0012917570,
                'O':0.0057326442,
                'Si':0.00066494756,
                'Mg':0.00070797884,
                'C':0.0023647147,
                'Ca':6.4145074e-5,
                'Al':5.5625916e-5,
                'Eu':3.6810875e-10,
                'Ne':0.0012564824,
                'S':0.00030926749}

    amu_el = {'Fe':55.854,
                'O':15.9994,
                'Si':28.0855,
                'Mg':24.305,
                'C':12.0107}

    asize = np.array([0.005,0.1])
    sgrain = np.array([2.2,2.2])
    t0_acc = 0.36945e+6*yr*(asize/0.005)*(sgrain/3)
    t0_coa = 5.68e+6*yr*(asize/0.005)*(sgrain/3)
    nH_coa = 10
    def get_tacc(Tk,rhoZ0):
        """
        Compute accretion timescale in seconds
        Taken from Le Bourlot+2012 (https://ui.adsabs.harvard.edu/abs/2012A%26A...541A..76L/abstract)
        """

        tacc = np.zeros(ndust)
        tacc = t0_acc.to('s').d * np.sqrt(50/Tk.d)*(mh.to('g').d/rhoZ0)*np.sqrt(amu_el['C']*amu/mh)*(1.0+1e-4*Tk.d**1.5)
        return tacc
    
    def get_tcoa(rho,rhoS):
        """
        Compute coagulation timescale of small to large grains in seconds
        """
        tcoa = np.zeros(ndust)
        if rho/mh.to('g').d > nH_coa:
            tcoa = t0_coa.to('s').d * (mh.to('g').d/(rhoS))
        else:
            temp = 1e15*yr
            tcoa[:] = temp.to('s').d
        return tcoa


    def dust_update(dt,Tk,ddust,rho,nH,nCarbon):
        """
        This function performs the update of the dust densities in a given time step
        using a Runge-Kutta 4th order integration method.
        """

        dtremain = dt
        countmax = 10000
        errmax = 0.1
        icount = np.zeros(ndust)
        dtloc_bin = np.zeros(ndust)
        
        rhoZ0 = np.zeros(ndust)
        dust_locked = np.sum(ddust)
        # print(nCarbon * amu_el['C'] * amu,dust_locked)
        rhoZ0 = nCarbon * amu_el['C'] * amu.to('g').d + dust_locked
        rhoZ0 = np.full(ndust,rhoZ0)
        rhoD0 = ddust

        # print('Ratios:',(nCarbon * amu_el['C'] * amu)/rhoZ0)

        tacc = get_tacc(Tk,rhoZ0)
        oneovertacc = 1.0 / tacc
        tcoa = get_tcoa(rho,ddust[0])
        oneovertcoa = 1.0 / tcoa
        
        t0 = np.full((ndust,4),3.15e19)
        t0[:,0] = 0.1 * tacc
        t0[:,3] = 0.1 * tcoa


        if rhoZ0[0]/rho>1e-10 or rhoD0[0]/rho >1e-10:
            ok_dust = False
        else:
            ok_dust = True

        while not ok_dust:
            rhoD0 = ddust
            rhoDT0 = np.sum(rhoD0)
            for i in range(0, ndust):
                if icount[i] == 0:
                    dtloc_bin[i] = np.min(t0[i,:])
            rhoGZ0 = rhoZ0 - rhoDT0
            # print(rhoGZ0,rhoZ0,rhoDT0)
            if any(rhoGZ0 < 0.0):
                print('Failed in dust_update_RK4 with negative gas metallicity!')
                print(rhoGZ0,rhoDT0)
                sys.exit()
            # print(rhoGZ0[0]/rhoZ0[0])
            if rhoGZ0[0]/rhoZ0[0]<1e-10: break
            # Initialise local timestep
            dtloc = np.min(dtloc_bin)
            dtloc = min(dtloc,dtremain)
            halfdtloc = 0.5*dtloc
            tot_Gvar = 0.0

            # Begin RK4
            k1,rhoD0k1,rhoGZ0k1,Gvar = np.zeros(ndust),np.zeros(ndust),np.zeros(ndust),0.0
            # Accretion
            dd = (rhoGZ0/rhoZ0)*rhoD0*oneovertacc
            k1 = k1 + dd
            Gvar = - np.sum(dd)
            # Coagulation
            dd = rhoD0[0]*oneovertcoa[0]
            k1[0] = k1[0] - dd
            k1[1] = k1[1] + dd
            tot_Gvar += Gvar*halfdtloc
            rhoD0k1 = rhoD0 + halfdtloc*k1
            rhoGZ0k1 = rhoGZ0 + halfdtloc*Gvar


            k2,rhoD0k2,rhoGZ0k2,Gvar = np.zeros(ndust),np.zeros(ndust),np.zeros(ndust),0.0
            # Accretion
            dd = (rhoGZ0k1/rhoZ0)*rhoD0k1*oneovertacc
            k2 = k2 + dd
            Gvar = - np.sum(dd)
            # Coagulation
            dd = rhoD0k1[0]*oneovertcoa[0]
            k2[0] = k2[0] - dd
            k2[1] = k2[1] + dd
            tot_Gvar += Gvar*halfdtloc
            rhoD0k2 = rhoD0 + halfdtloc*k2
            rhoGZ0k2 = rhoGZ0 + halfdtloc*Gvar

            k3,rhoD0k3,rhoGZ0k3,Gvar = np.zeros(ndust),np.zeros(ndust),np.zeros(ndust),0.0
            # Accretion
            dd = (rhoGZ0k2/rhoZ0)*rhoD0k2*oneovertacc
            k3 = k3 + dd
            Gvar = - np.sum(dd)
            # Coagulation
            dd = rhoD0k2[0]*oneovertcoa[0]
            k3[0] = k3[0] - dd
            k3[1] = k3[1] + dd
            tot_Gvar += Gvar*halfdtloc
            rhoD0k3 = rhoD0 + dtloc*k3
            rhoGZ0k3 = rhoGZ0 + dtloc*Gvar

            k4,drhoD,rhoD = np.zeros(ndust),np.zeros(ndust),np.zeros(ndust)
            # Accretion
            dd = (rhoGZ0k3/rhoZ0)*rhoD0k3*oneovertacc
            k4 = k4 + dd
            # Coagulation
            dd = rhoD0k3[0]*oneovertcoa[0]
            k4[0] = k4[0] - dd
            k4[1] = k4[1] + dd
            drhoD = dtloc/6.0 * (k1+2.0*k2+2.0*k3+k4)
            rhoD = rhoD0 + drhoD
            
            # Check now for integration errors
            okdt_bin = np.full((ndust), False)
            for i in range(0,ndust):
                if rhoD0[i] >0.0:
                    error_rel1 = abs(drhoD[i]) / min(rhoD0[i],rhoD[i])
                    den0 = (1.0-rhoD0[i]/rhoZ0[i])*rhoD0[i]
                    den = (1.0-rhoD[i]/rhoZ0[i])*rhoD[i]
                    if min(den0,den)<1e-10:
                        error_rel = error_rel1
                    else:
                        error_rel2 = abs(drhoD[i]) / min(den0,den)
                        error_rel = max(error_rel1,error_rel2)
                else:
                    ok_dust = True
                
                if not ok_dust:
                    if error_rel<=errmax and error_rel >= 0.0:
                        okdt_bin[i] = True
                        if error_rel<=0.5*errmax:
                            dtloc_bin[i] = dtloc*2.0
                    elif error_rel>=errmax or error_rel < 0.0:
                        dtloc_bin[i] = 0.5*dtloc
                    icount[i] += 1
                
                if icount[i] > countmax:
                    print('stopping in dust processing icount> %i'%countmax)
                    sys.exit()
            
            if not ok_dust:
                if all(okdt_bin):
                    dtremain = dtremain - dtloc
                    for i in range(0, ndust):
                        ddust[i] = rhoD[i]
                if dtremain<=0.0:
                    ok_dust = True
            
        return ddust

    start_rho = mh * start_nH *cm**-3 / (1-zinit*sum(mf_asplund09.values()))
    rhoCarbon = [start_rho.to('g/cm**3').d*(mf_asplund09['C'] - 2*start_fdust)]
    rho_dust = [[start_rho.to('g/cm**3').d*start_fdust,start_rho.to('g/cm**3').d*start_fdust]]
    density = [start_rho.to('g/cm**3').d]
    time = [0.0]
    nH = start_nH *cm**-3
    full_nH = [nH.to('cm**-3').d]

    while nH.to('cm**-3').d < end_nH:

        # Update densities
        omu_model = omukai_model(nH.to('cm**-3').d,300,5/3)
        dt = 1e-3 * omu_model.tcol
        time.append(time[-1]+dt.to('Myr').d)
        fdust = rho_dust[-1] / density[-1]
        new_rho = density[-1] + (density[-1]/omu_model.tcol.to('s').d) * dt.to('s').d
        density.append(new_rho)
        nH = new_rho*(1.0 - zinit*sum(mf_asplund09.values())) / mh.to('g').d
        nH = nH *cm**-3
        full_nH.append(nH.to('cm**-3').d)
        print(time[-1]+dt.to('Myr').d, nH.to('cm**-3'))

        # Update dust        
        temp_dust = fdust * new_rho
        # print(temp_dust, density[-2])
        nCarbon_gas = new_rho*mf_asplund09['C']/(amu_el['C']*amu.to('g').d) - np.sum(temp_dust)/(amu_el['C']*amu.to('g').d)
        # print('orig:',new_rho*mf_asplund09['C'])
        # print(dt.to('s').d,omu_model.T,temp_dust,new_rho,
                                # nH.to('cm**-3').d,nCarbon_gas)
        new_dust = dust_update(dt.to('s').d,omu_model.T,temp_dust,new_rho,
                                nH.to('cm**-3').d,nCarbon_gas)
        new_Carbon = new_rho*mf_asplund09['C'] - np.sum(temp_dust)
        # print('final:',new_Carbon/(new_rho*mf_asplund09['C']),new_Carbon)
        rhoCarbon.append(new_Carbon)
        # print(new_dust)
        rho_dust.append(np.copy(new_dust))

    # print(rho_dust[0])

    density = np.asarray(density)
    full_nH = np.asarray(full_nH)
    time = np.asarray(time)
    rhoCarbon = np.asarray(rhoCarbon)
    rho_dust = np.asarray(rho_dust)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=100, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$nH$ [cm$^{-3}$]', fontsize=16)
    ax.set_xlabel(r'$t$ [Myr]',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.plot(time, full_nH)
    fig.savefig('cloud_collapse_model.png', format='png', dpi=200)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6,5), dpi=100, facecolor='w', edgecolor='k')

    ax = axes[0]
    ax.set_ylabel(r'$X_{\rm dust}$', fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    # print(rho_dust)
    # print(density)
    ax.plot(full_nH, rho_dust[:,0]/density, label='smallC')
    ax.plot(full_nH, rho_dust[:,1]/density, label='largeC')
    ax.legend(loc='best', fontsize=12,frameon=False)

    ax = axes[1]
    ax.set_ylabel(r'$\rho_{\rm dust}/\rho_{\rm C}$', fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xlabel(r'$nH$ [cm$^{-3}$]',fontsize=16)
    ax.plot(full_nH, rho_dust[:,0]/rhoCarbon, label='smallC')
    ax.plot(full_nH, rho_dust[:,1]/rhoCarbon, label='largeC')
    fig.savefig('dust_collapse_model.png', format='png', dpi=200)

    plt.close()
