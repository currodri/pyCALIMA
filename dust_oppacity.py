"""
DUST EFFICIENCIES TABLES

This set of tools have been constructed such that the public
tables from B. Draine and co. can be read, visualised
and reorganised in look-up tables for RAMSES Dust-RTZ

By: Curro Rodriguez (currodri@gmail.com)
"""

# Import some libraries
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
})
import re
from dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,\
                        LogNormal_Distribution,PowerLaw_ExpCutoff_Distribution, \
                        Classical_LogNormal_Distribution
# Functions

def dust_efficiencies(filename):
    """
    This function allows for the construction of a clean and
    nice dataset.
    """
    columns = ['w(micron)','Q_abs', 'Q_sca', 'g=<cos>']
    data = {}

    with open(filename) as f:
        # Begin by reading the header
        for i in range(0,5):
            hd = f.readline()
            if i == 1:
                dust_type = hd
            elif i==3:
                info = list(filter(None, hd.split(' ')))
                nrad = int(info[0])
                amin = float(info[1])
                amax = float(info[2])
            elif i==4:
                info = list(filter(None, hd.split(' ')))
                nwav = int(info[0])
                wmin = float(info[1])
                wmax = float(info[2])
        print(dust_type,nrad,nwav)
        
        while True:
            f.readline() # Blank line
            myarray = np.zeros((nwav,4))
            a = str(f.readline().split(' ')[0])
            if a == '':
                print('End of file')
                break
            f.readline() # Column names
            for i in range(0, nwav):
                line = f.readline()
                myarray[i,:] = np.fromstring(line, dtype=float, sep=' ')
            data[a] = myarray


    return nwav,data,columns,dust_type

def pah_efficiencies(filename,verbose=False):
    """
    This function allows for the construction of a clean and
    nice dataset.
    """
    columns = ['w(micron)','Q_ext','Q_abs', 'Q_sca', 'g=<cos>']
    data = {}

    with open(filename) as f:
        # Begin by reading the header
        for i in range(0,9):
            hd = f.readline()
            if i == 0:
                dust_type = hd
            elif i==7:
                info = list(filter(None, hd.split(' ')))
                nrad = int(info[0])
                amin = float(info[1])
                amax = float(info[2])
            elif i==8:
                info = list(filter(None, hd.split(' ')))
                nwav = int(info[0])
                wmin = float(info[1])
                wmax = float(info[2])
        if verbose: print(dust_type,nrad,nwav)
        
        while True:
            f.readline() # Blank line
            myarray = np.zeros((nwav,5))
            a = str(f.readline().split(' ')[0])
            if a == '':
                if verbose: print('End of file')
                break
            f.readline() # Column names
            
            for i in range(0, nwav):
                line = f.readline()
                dig = np.fromstring(line, dtype=float, sep=' ')
                if len(dig) == 4:
                    last_dig = float(line[-9:])
                    dig = np.concatenate((dig,[last_dig]),axis=0)
                myarray[i,:] = dig
            data[a] = myarray


    return data,columns,dust_type,nwav

def plot_efficiencies(filename,dust_type='grains',
                      do_average=True,
                      output_average=True):

    fig, axes = plt.subplots(3,1, figsize=(6,9),dpi=300,facecolor='w',edgecolor='k',sharey=True)

    if dust_type == 'grains':
        data,columns,name = dust_efficiencies(filename)
    else:
        data,columns,name = pah_efficiencies(filename)
    
    if 'PAH' in name:
        dist = [LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0]),
                LogNormal_Distribution(basic_a0[1],basic_amin[1],basic_amax[1],basic_sigma[1],basic_s[1])]
        ndist = 2
        linestyles = ['-.','-']
        name = ['smallPAHs','largePAHs']
    elif 'Graphite' in name:
        dist = [LogNormal_Distribution(basic_a0[2],basic_amin[2],basic_amax[2],basic_sigma[2],basic_s[2]),
                LogNormal_Distribution(basic_a0[3],basic_amin[3],basic_amax[3],basic_sigma[3],basic_s[3])]
        ndist = 2
        linestyles = ['-.','-']
        name = ['smallC','largeC']
    elif 'silicate' in name:
        dist = [LogNormal_Distribution(basic_a0[5],basic_amin[5],basic_amax[5],basic_sigma[5],basic_s[5]),
                LogNormal_Distribution(basic_a0[6],basic_amin[6],basic_amax[6],basic_sigma[6],basic_s[6])]
        ndist = 2
        linestyles = ['-.','-']
        name = ['smallSil','largeSil']
    for a in data:
        Q_sca = data[a][:,columns.index('Q_sca')]
        Q_abs = data[a][:,columns.index('Q_abs')]
        g     = data[a][:,columns.index('g=<cos>')]
        w     = data[a][:,columns.index('w(micron)')]
        Q_rp  = Q_abs + (1-g)*Q_sca
        
        # if float(a) == 1e-1:
        #     axes[0].plot(w,Q_abs,alpha=0.3,linewidth=0.5,color='k',linestyle=':')
        #     axes[1].plot(w,Q_sca,alpha=0.3,linewidth=0.5,color='r',linestyle=':')
        #     axes[2].plot(w,Q_rp,alpha=0.3,linewidth=0.5,color='b',linestyle=':')
        # elif float(a) == 5.012E-03:
        #     axes[0].plot(w,Q_abs,alpha=0.3,linewidth=0.5,color='k',linestyle='--')
        #     axes[1].plot(w,Q_sca,alpha=0.3,linewidth=0.5,color='r',linestyle='--')
        #     axes[2].plot(w,Q_rp,alpha=0.3,linewidth=0.5,color='b',linestyle='--')
        # else:
        
        axes[0].plot(w,Q_abs*np.pi*float(a)**2.,alpha=0.3,linewidth=0.5,color='k')
        axes[1].plot(w,Q_sca*np.pi*float(a)**2.,alpha=0.3,linewidth=0.5,color='r')
        axes[2].plot(w,Q_rp*np.pi*float(a)**2.,alpha=0.3,linewidth=0.5,color='b')
    if do_average:
        for i in range(0,ndist):
            nwav = len(w)
            Q_sca_eff = np.zeros(nwav)
            Q_abs_eff = np.zeros(nwav)
            Q_rp_eff  = np.zeros(nwav)
            nrad = len(data.keys())
            akeys= list(data.keys())
            for j in range(0, nwav):
                sizes = np.zeros(nrad)
                Q_sca = np.zeros(nrad)
                Q_abs = np.zeros(nrad)
                Q_rp  = np.zeros(nrad)
                for k in range(0,nrad):
                    tmpdt = data[akeys[k]]
                    sizes[k] = float(akeys[k])
                    Q_sca[k] = tmpdt[j,columns.index('Q_sca')]
                    Q_abs[k] = tmpdt[j,columns.index('Q_abs')]
                    g        = tmpdt[j,columns.index('g=<cos>')]
                    w        = tmpdt[:,columns.index('w(micron)')]
                    Q_rp[k]  = Q_abs[k] + (1-g)*Q_sca[k]
                Q_sca_eff[j] = dist[i].averaged_over_number(Q_sca*np.pi*sizes**2.,sizes)
                Q_abs_eff[j] = dist[i].averaged_over_number(Q_abs*np.pi*sizes**2.,sizes)
                Q_rp_eff[j]  = dist[i].averaged_over_number(Q_rp*np.pi*sizes**2.,sizes)
            axes[0].plot(w,Q_abs_eff,linewidth=2,color='k',linestyle=linestyles[i],label=name[i])
            axes[1].plot(w,Q_sca_eff,linewidth=2,color='r',linestyle=linestyles[i])
            axes[2].plot(w,Q_rp_eff,linewidth=2,color='b',linestyle=linestyles[i])
            if output_average:
                # Convert wavelength from micron to angstrom 
                w = w[::-1] * 1e4
                # Convert cross section from micron^2 to cm^2
                Q_abs_eff = Q_abs_eff[::-1] * 1e-8
                Q_sca_eff = Q_sca_eff[::-1] * 1e-8 
                Q_rp_eff = Q_rp_eff[::-1] * 1e-8
                f = open('averaged_cross_section_%.4f_micron_%s'%(dist[i].a0,filename.split('/')[-1]), 'w', encoding="utf-8")
                f.write("{:8d}".format(nwav)+'\n')
                for j in range(0,nwav):
                    f.write("{:14.6e}".format(w[j])+'\n')
                for j in range(0,nwav):
                    f.write("{:14.6e}".format(Q_abs_eff[j])+'\n')
                for j in range(0,nwav):
                    f.write("{:14.6e}".format(Q_sca_eff[j])+'\n')
                for j in range(0,nwav):
                    f.write("{:14.6e}".format(Q_rp_eff[j])+'\n')
                f.close()

    for i in range(0,3):
        ax = axes[i]
        ax.tick_params(labelsize=14)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-10,1e-3])
    axes[0].set_ylabel(r'$C_{\rm abs}$ [cm$^2$]', fontsize=16)
    axes[1].set_ylabel(r'$C_{\rm sca}$ [cm$^2$]', fontsize=16)
    axes[2].set_ylabel(r'$C_{\rm rp}$ [cm$^2$]', fontsize=16)
    axes[2].set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=16)
    axes[0].legend(loc='best',fontsize=14,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.06,left=0.13,right=0.99,hspace=0,wspace=0)
    fig.savefig(folder+'/cross_section_%s.pdf'%filename.split('/')[-1], format='pdf', dpi=300)


def plot_cs_sne(rho_gas,D_smallPAHs,D_largePAHs,D_smallC,D_largeC,D_smallSil,D_largeSil,export=False):

    # 1. Set up the figure
    fig, axes = plt.subplots(2,2, figsize=(10,6),dpi=300,facecolor='w',edgecolor='k',sharey=False,sharex=False)
    axes[0,0].set_ylabel(r'$a^4n(a)$', fontsize=16)
    axes[0,1].set_ylabel(r'$C_{\rm abs}$, $C_{\rm sca}$ [cm$^2$] \& $g$', fontsize=16)
    axes[1,0].set_ylabel(r'$a^4n(a)$', fontsize=16)
    axes[1,1].set_ylabel(r'$C_{\rm abs}$, $C_{\rm sca}$ [cm$^2$] \& $g$', fontsize=16)
    axes[1,0].set_xlabel(r'$a$ [$\mu$m]', fontsize=16)
    axes[1,1].set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=16)

    # 2. Setup the different size distributions
    # Gao et al. 2020 for the empirically derived extinction curve of the supernova SN2012cu
    # (https://www.sciencedirect.com/science/article/pii/S0032063318300321?via%3Dihub)
    Gao_2020_sil = PowerLaw_ExpCutoff_Distribution(5e-3,5,0.04,0.5,3.3)
    Gao_2020_gra = PowerLaw_ExpCutoff_Distribution(5e-3,5,0.03,0.5,2.2)

    # Asano et al. (2013) log-normal distribution parameters (originally used
    # for AGB production) it is also used for SNe ejecta in Hirashita & Aoyama (2019)
    # (https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.2555H/abstract)
    Asano_2013_sil = LogNormal_Distribution(0.1,5e-3,5.,0.47,3.3)
    Asano_2013_gra = LogNormal_Distribution(0.1,5e-3,5.,0.47,2.2)

    # Nozawa et al. (2007) power-law distributions for Mg2SiO4 and C grains
    # after the effect of sputtering and shattering in Pop III ejecta
    # (https://ui.adsabs.harvard.edu/abs/2007ApJ...666..955N/abstract)
    # NOTE: They do not provide the numerical values for this, so I have
    # obtain them by copying their table and fitting the power-law function
    Nozawa_2007_sil = PowerLaw_ExpCutoff_Distribution(1.6e-3,1.0,5.23e-02,1.25,3.3)
    Nozawa_2007_gra = PowerLaw_ExpCutoff_Distribution(1.6e-3,1.0,2.14e-02,1.15,2.2)

    # Marassi et al. (2019) log-normal distribution using the Limongi & Chieffi (2018) SNe
    # yields for the ejecta of massive stars
    # (https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.3109M/abstract)
    Marassi_2019_sil = Classical_LogNormal_Distribution(0.025,1e-3,1,0.1,2.2)
    Marassi_2019_gra = Classical_LogNormal_Distribution(0.075,1e-3,1,0.1,2.2)

    # RAMSES Dust: Using the resulting grain size distribution from the G8 simulation
    # with initial 0.003 Zsun and DTMinit=1d-3 and 18 pc resolution (output 10)
    # fCs  =     0.005     0.010
    # fCl  =     0.464     0.990
    # fSils=     0.010     0.018
    # fSill=     0.522     0.982
    # fs   =     0.014
    # fl   =     0.986
    # fC   =     0.468
    # fSil =     0.532
    ramses_silLarge = LogNormal_Distribution(1e-1,5e-3,1.0,0.75,3.3)
    ramses_silSmall = LogNormal_Distribution(5e-3,5e-4,0.1,0.75,3.3)
    ramses_graLarge = LogNormal_Distribution(1e-1,5e-3,1.0,0.75,2.2)
    ramses_graSmall = LogNormal_Distribution(5e-3,5e-4,0.1,0.75,2.2)
    

    # 3. Plot the size distribution on the first axes
    a = np.logspace(np.log10(5e-3),np.log10(0.8),100)
    axes[0,0].plot(a,a**4*Gao_2020_sil.n_density(rho_gas*D_largeSil,a),
                 label='Gao et al. 2020',color='#8CBA80',linestyle='-')
    axes[1,0].plot(a,a**4*Gao_2020_gra.n_density(rho_gas*D_largeC,a),
                    color='#8CBA80',linestyle='-')
    axes[0,0].plot(a,a**4*Asano_2013_sil.n_density(rho_gas*D_largeSil,a),
                 label='Asano et al. 2013',color='#658E9C',linestyle='-')
    axes[1,0].plot(a,a**4*Asano_2013_gra.n_density(rho_gas*D_largeC,a),
                    color='#658E9C',linestyle='-')
    axes[0,0].plot(a,a**4*Nozawa_2007_sil.n_density(rho_gas*D_largeSil,a),
                    label='Nozawa et al. 2007',color='#F5A65B',linestyle='-')
    axes[1,0].plot(a,a**4*Nozawa_2007_gra.n_density(rho_gas*D_largeC,a),
                    color='#F5A65B',linestyle='-')
    axes[0,0].plot(a,a**4*Marassi_2019_sil.n_density(rho_gas*D_largeSil,a),
                    label='Marassi et al. 2019',color='#F28C8C',linestyle='-')
    axes[1,0].plot(a,a**4*Marassi_2019_gra.n_density(rho_gas*D_largeC,a),
                    color='#F28C8C',linestyle='-')
    axes[0,0].plot(a,a**4*ramses_silLarge.n_density(0.986*rho_gas*D_largeSil,a)+a**4*ramses_silSmall.n_density(0.014*rho_gas*D_largeSil,a),
                    label='RAMSES',color='k',linestyle='-')
    axes[1,0].plot(a,a**4*ramses_graLarge.n_density(0.986*rho_gas*D_largeC,a)+a**4*ramses_graSmall.n_density(0.014*rho_gas*D_largeSil,a),
                    color='k',linestyle='-')

    axes[0,0].set_yscale('log')
    axes[0,0].set_xscale('log')
    axes[0,0].legend(loc='best',fontsize=10,frameon=False)
    axes[0,0].set_ylim([4e-30,3e-27])
    axes[0,0].tick_params(labelsize=14)
    axes[0,0].xaxis.set_ticks_position('both')
    axes[0,0].yaxis.set_ticks_position('both')
    axes[0,0].minorticks_on()
    axes[0,0].tick_params(which='both',axis="both",direction="in")

    axes[0,0].plot(a,5e-28*a**(0.5),':',color='gray',linewidth=2)
    axes[0,0].text(0.5, 0.52, r'MRN ($n\propto a^{-3.5}$)',
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes[0,0].transAxes,fontsize=14,rotation=17)

    axes[1,0].set_yscale('log')
    axes[1,0].set_xscale('log')
    axes[1,0].set_ylim([4e-30,3e-27])
    axes[1,0].tick_params(labelsize=14)
    axes[1,0].xaxis.set_ticks_position('both')
    axes[1,0].yaxis.set_ticks_position('both')
    axes[1,0].minorticks_on()
    axes[1,0].tick_params(which='both',axis="both",direction="in")

    axes[1,0].plot(a,5e-28*a**(0.5),':',color='gray',linewidth=2)
    axes[1,0].text(0.13, 0.37, r'MRN ($n\propto a^{-3.5}$)',
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes[1,0].transAxes,fontsize=14,rotation=17)


    # 4. Compute and plot the number-averaged cross-section
    nwav_Gra,data_Gra,columns_Gra,name_Gra = dust_efficiencies('draine_lee_1984/Gra_81')
    nwav_Sil,data_Sil,columns_Sil,name_Sil = dust_efficiencies('draine_lee_1984/suvSil_81')

    nrad = len(data_Sil.keys())
    C_sca_Asano_2013_sil = np.zeros(nwav_Sil)
    C_abs_Asano_2013_sil = np.zeros(nwav_Sil)
    g_Asano_2013_sil = np.zeros(nwav_Sil)
    C_sca_Asano_2013_gra = np.zeros(nwav_Gra)
    C_abs_Asano_2013_gra = np.zeros(nwav_Gra)
    g_Asano_2013_gra = np.zeros(nwav_Gra)

    C_sca_Gao_2020_sil = np.zeros(nwav_Sil)
    C_abs_Gao_2020_sil = np.zeros(nwav_Sil)
    g_Gao_2020_sil = np.zeros(nwav_Sil)
    C_sca_Gao_2020_gra = np.zeros(nwav_Gra)
    C_abs_Gao_2020_gra = np.zeros(nwav_Gra)
    g_Gao_2020_gra = np.zeros(nwav_Gra)

    C_sca_Nozawa_2007_sil = np.zeros(nwav_Sil)
    C_abs_Nozawa_2007_sil = np.zeros(nwav_Sil)
    g_Nozawa_2007_sil = np.zeros(nwav_Sil)
    C_sca_Nozawa_2007_gra = np.zeros(nwav_Gra)
    C_abs_Nozawa_2007_gra = np.zeros(nwav_Gra)
    g_Nozawa_2007_gra = np.zeros(nwav_Gra)

    C_sca_Marassi_2019_sil = np.zeros(nwav_Sil)
    C_abs_Marassi_2019_sil = np.zeros(nwav_Sil)
    g_Marassi_2019_sil = np.zeros(nwav_Sil)
    C_sca_Marassi_2019_gra = np.zeros(nwav_Gra)
    C_abs_Marassi_2019_gra = np.zeros(nwav_Gra)
    g_Marassi_2019_gra = np.zeros(nwav_Gra)

    C_sca_ramses_silLarge = np.zeros(nwav_Sil)
    C_abs_ramses_silLarge = np.zeros(nwav_Sil)
    g_ramses_silLarge = np.zeros(nwav_Sil)
    C_sca_ramses_graLarge = np.zeros(nwav_Gra)
    C_abs_ramses_graLarge = np.zeros(nwav_Gra)
    g_ramses_graLarge = np.zeros(nwav_Gra)


    nrad = len(data_Sil.keys())
    akeys= list(data_Sil.keys())
    for j in range(0,nwav_Sil):
        sizes_Sil = np.zeros(nrad)
        Q_sca_Sil = np.zeros(nrad)
        Q_abs_Sil = np.zeros(nrad)
        g_Sil = np.zeros(nrad)
        w_Sil = np.zeros(nrad)
        sizes_Gra = np.zeros(nrad)
        Q_sca_Gra = np.zeros(nrad)
        Q_abs_Gra = np.zeros(nrad)
        g_Gra = np.zeros(nrad)
        w_Gra = np.zeros(nrad)
        for k in range(0,nrad):
            tmpdt = data_Sil[akeys[k]]
            sizes_Sil[k] = float(akeys[k])
            Q_sca_Sil[k] = tmpdt[j,columns_Sil.index('Q_sca')]
            Q_abs_Sil[k] = tmpdt[j,columns_Sil.index('Q_abs')]
            g_Sil[k]     = tmpdt[j,columns_Sil.index('g=<cos>')]
            w_Sil        = tmpdt[:,columns_Sil.index('w(micron)')]
            tmpdt = data_Gra[akeys[k]]
            sizes_Gra[k] = float(akeys[k])
            Q_sca_Gra[k] = tmpdt[j,columns_Gra.index('Q_sca')]
            Q_abs_Gra[k] = tmpdt[j,columns_Gra.index('Q_abs')]
            g_Gra[k]     = tmpdt[j,columns_Gra.index('g=<cos>')]
            w_Gra        = tmpdt[:,columns_Gra.index('w(micron)')]
        C_sca_Asano_2013_sil[j] = Asano_2013_sil.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_Asano_2013_sil[j] = Asano_2013_sil.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_Asano_2013_sil[j]     = Asano_2013_sil.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_Asano_2013_gra[j] = Asano_2013_gra.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_Asano_2013_gra[j] = Asano_2013_gra.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_Asano_2013_gra[j]     = Asano_2013_gra.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

        C_sca_Gao_2020_sil[j] = Gao_2020_sil.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_Gao_2020_sil[j] = Gao_2020_sil.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_Gao_2020_sil[j]     = Gao_2020_sil.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_Gao_2020_gra[j] = Gao_2020_gra.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_Gao_2020_gra[j] = Gao_2020_gra.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_Gao_2020_gra[j]     = Gao_2020_gra.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

        C_sca_Nozawa_2007_sil[j] = Nozawa_2007_sil.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_Nozawa_2007_sil[j] = Nozawa_2007_sil.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_Nozawa_2007_sil[j]     = Nozawa_2007_sil.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_Nozawa_2007_gra[j] = Nozawa_2007_gra.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_Nozawa_2007_gra[j] = Nozawa_2007_gra.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_Nozawa_2007_gra[j]     = Nozawa_2007_gra.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

        C_sca_Marassi_2019_sil[j] = Marassi_2019_sil.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_Marassi_2019_sil[j] = Marassi_2019_sil.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_Marassi_2019_sil[j]     = Marassi_2019_sil.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_Marassi_2019_gra[j] = Marassi_2019_gra.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_Marassi_2019_gra[j] = Marassi_2019_gra.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_Marassi_2019_gra[j]     = Marassi_2019_gra.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

        C_sca_ramses_silLarge[j] = ramses_silLarge.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_ramses_silLarge[j] = ramses_silLarge.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_ramses_silLarge[j]     = ramses_silLarge.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_ramses_graLarge[j] = ramses_graLarge.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_ramses_graLarge[j] = ramses_graLarge.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_ramses_graLarge[j]     = ramses_graLarge.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

    axes[0,1].plot(w_Sil,C_abs_Asano_2013_sil * 1e-8,linewidth=2,color='#658E9C',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_Asano_2013_sil * 1e-8,linewidth=2,color='#658E9C',linestyle='--')
    axes[0,1].plot(w_Sil,g_Asano_2013_sil * 1e-8,linewidth=2,color='#658E9C',linestyle=':')
    axes[0,1].plot(w_Sil,C_abs_Gao_2020_sil * 1e-8,linewidth=2,color='#8CBA80',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_Gao_2020_sil * 1e-8,linewidth=2,color='#8CBA80',linestyle='--')
    axes[0,1].plot(w_Sil,g_Gao_2020_sil * 1e-8,linewidth=2,color='#8CBA80',linestyle=':')
    axes[0,1].plot(w_Sil,C_abs_Nozawa_2007_sil * 1e-8,linewidth=2,color='#F5A65B',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_Nozawa_2007_sil * 1e-8,linewidth=2,color='#F5A65B',linestyle='--')
    axes[0,1].plot(w_Sil,g_Nozawa_2007_sil * 1e-8,linewidth=2,color='#F5A65B',linestyle=':')
    axes[0,1].plot(w_Sil,C_abs_Marassi_2019_sil * 1e-8,linewidth=2,color='#F28C8C',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_Marassi_2019_sil * 1e-8,linewidth=2,color='#F28C8C',linestyle='--')
    axes[0,1].plot(w_Sil,g_Marassi_2019_sil * 1e-8,linewidth=2,color='#F28C8C',linestyle=':')
    axes[0,1].plot(w_Sil,C_abs_ramses_silLarge * 1e-8,linewidth=2,color='k',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_ramses_silLarge * 1e-8,linewidth=2,color='k',linestyle='--')
    axes[0,1].plot(w_Sil,g_ramses_silLarge * 1e-8,linewidth=2,color='k',linestyle=':')


    axes[1,1].plot(w_Gra,C_abs_Asano_2013_gra * 1e-8,linewidth=2,color='#658E9C',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_Asano_2013_gra * 1e-8,linewidth=2,color='#658E9C',linestyle='--')
    axes[1,1].plot(w_Gra,g_Asano_2013_gra * 1e-8,linewidth=2,color='#658E9C',linestyle=':')
    axes[1,1].plot(w_Gra,C_abs_Gao_2020_gra * 1e-8,linewidth=2,color='#8CBA80',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_Gao_2020_gra * 1e-8,linewidth=2,color='#8CBA80',linestyle='--')
    axes[1,1].plot(w_Gra,g_Gao_2020_gra * 1e-8,linewidth=2,color='#8CBA80',linestyle=':')
    axes[1,1].plot(w_Gra,C_abs_Nozawa_2007_gra * 1e-8,linewidth=2,color='#F5A65B',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_Nozawa_2007_gra * 1e-8,linewidth=2,color='#F5A65B',linestyle='--')
    axes[1,1].plot(w_Gra,g_Nozawa_2007_gra * 1e-8,linewidth=2,color='#F5A65B',linestyle=':')
    axes[1,1].plot(w_Gra,C_abs_Marassi_2019_gra * 1e-8,linewidth=2,color='#F28C8C',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_Marassi_2019_gra * 1e-8,linewidth=2,color='#F28C8C',linestyle='--')
    axes[1,1].plot(w_Gra,g_Marassi_2019_gra * 1e-8,linewidth=2,color='#F28C8C',linestyle=':')
    axes[1,1].plot(w_Gra,C_abs_ramses_graLarge * 1e-8,linewidth=2,color='k',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_ramses_graLarge * 1e-8,linewidth=2,color='k',linestyle='--')
    axes[1,1].plot(w_Gra,g_ramses_graLarge * 1e-8,linewidth=2,color='k',linestyle=':')

    # 5. If the export flag is True, we save these number-averaged cross-sections to individual files
    # indicating well the names as well as the properties of the underlying distribution assumed in the
    # header of the file
    if export:
        folder = './cross_section_sne/'
        # Convert wavelength from micron to angstrom 
        w_Sil = w_Sil[::-1] * 1e4
        w_Gra = w_Gra[::-1] * 1e4
        # Convert cross section from micron^2 to cm^2
        C_abs_Asano_2013_sil = C_abs_Asano_2013_sil[::-1] * 1e-8
        C_sca_Asano_2013_sil = C_sca_Asano_2013_sil[::-1] * 1e-8
        g_Asano_2013_sil = g_Asano_2013_sil[::-1]
        C_abs_Asano_2013_gra = C_abs_Asano_2013_gra[::-1] * 1e-8
        C_sca_Asano_2013_gra = C_sca_Asano_2013_gra[::-1] * 1e-8
        g_Asano_2013_gra = g_Asano_2013_gra[::-1]

        C_abs_Gao_2020_sil = C_abs_Gao_2020_sil[::-1] * 1e-8
        C_sca_Gao_2020_sil = C_sca_Gao_2020_sil[::-1] * 1e-8
        g_Gao_2020_sil = g_Gao_2020_sil[::-1]
        C_abs_Gao_2020_gra = C_abs_Gao_2020_gra[::-1] * 1e-8
        C_sca_Gao_2020_gra = C_sca_Gao_2020_gra[::-1] * 1e-8
        g_Gao_2020_gra = g_Gao_2020_gra[::-1]

        C_abs_Nozawa_2007_sil = C_abs_Nozawa_2007_sil[::-1] * 1e-8
        C_sca_Nozawa_2007_sil = C_sca_Nozawa_2007_sil[::-1] * 1e-8
        g_Nozawa_2007_sil = g_Nozawa_2007_sil[::-1]
        C_abs_Nozawa_2007_gra = C_abs_Nozawa_2007_gra[::-1] * 1e-8
        C_sca_Nozawa_2007_gra = C_sca_Nozawa_2007_gra[::-1] * 1e-8
        g_Nozawa_2007_gra = g_Nozawa_2007_gra[::-1]

        C_abs_Marassi_2019_sil = C_abs_Marassi_2019_sil[::-1] * 1e-8
        C_sca_Marassi_2019_sil = C_sca_Marassi_2019_sil[::-1] * 1e-8
        g_Marassi_2019_sil = g_Marassi_2019_sil[::-1]
        C_abs_Marassi_2019_gra = C_abs_Marassi_2019_gra[::-1] * 1e-8
        C_sca_Marassi_2019_gra = C_sca_Marassi_2019_gra[::-1] * 1e-8
        g_Marassi_2019_gra = g_Marassi_2019_gra[::-1]

        C_abs_ramses_silLarge = C_abs_ramses_silLarge[::-1] * 1e-8
        C_sca_ramses_silLarge = C_sca_ramses_silLarge[::-1] * 1e-8
        g_ramses_silLarge = g_ramses_silLarge[::-1]
        C_abs_ramses_graLarge = C_abs_ramses_graLarge[::-1] * 1e-8
        C_sca_ramses_graLarge = C_sca_ramses_graLarge[::-1] * 1e-8
        g_ramses_graLarge = g_ramses_graLarge[::-1]
        
        # Export the data to a file
        with open(folder+'/cross_section_sne_Asano_2013_sil.dat','w') as f:
            f.write('# Asano et al. 2013 silicates\n')
            f.write('# Modified Log-normal distribution (Hirashita 2015) for AGB production\n')
            f.write('# with a0=0.1 [micron], amin=5e-3 [micron], amax=5 [micron], alpha=0.47, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_Asano_2013_sil[j])+" "+
                        "{:14.6e}".format(C_sca_Asano_2013_sil[j])+" "+
                        "{:14.6e}".format(g_Asano_2013_sil[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Asano_2013_gra.dat','w') as f:
            f.write('# Asano et al. 2013 graphite\n')
            f.write('# Modified Log-normal distribution (Hirashita 2015) for AGB production\n')
            f.write('# with a0=0.1 [micron], amin=5e-3 [micron], amax=5 [micron], alpha=0.47, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_Asano_2013_gra[j])+" "+
                        "{:14.6e}".format(C_sca_Asano_2013_gra[j])+" "+
                        "{:14.6e}".format(g_Asano_2013_gra[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Gao_2020_sil.dat','w') as f:
            f.write('# Gao et al. 2020 silicates\n')
            f.write('# Power-law with exponential cutoff distribution\n')
            f.write('# with acut=0.04 [micron], amin=0.005 [micron], amax=5 [micron], alpha=0.5, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_Gao_2020_sil[j])+" "+
                        "{:14.6e}".format(C_sca_Gao_2020_sil[j])+" "+
                        "{:14.6e}".format(g_Gao_2020_sil[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Gao_2020_gra.dat','w') as f:
            f.write('# Gao et al. 2020 graphite\n')
            f.write('# Power-law with exponential cutoff distribution\n')
            f.write('# with acut=0.03 [micron], amin=0.005 [micron], amax=5 [micron], alpha=0.5, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_Gao_2020_gra[j])+" "+
                        "{:14.6e}".format(C_sca_Gao_2020_gra[j])+" "+
                        "{:14.6e}".format(g_Gao_2020_gra[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Nozawa_2007_sil.dat','w') as f:
            f.write('# Nozawa et al. 2007 silicates\n')
            f.write('# Power-law with exponential cutoff distribution\n')
            f.write('# with acut=0.0523 [micron], amin=0.0016 [micron], amax=1 [micron], alpha=1.25, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_Nozawa_2007_sil[j])+" "+
                        "{:14.6e}".format(C_sca_Nozawa_2007_sil[j])+" "+
                        "{:14.6e}".format(g_Nozawa_2007_sil[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Nozawa_2007_gra.dat','w') as f:
            f.write('# Nozawa et al. 2007 graphite\n')
            f.write('# Power-law with exponential cutoff distribution\n')
            f.write('# with acut=0.0214 [micron], amin=0.0016 [micron], amax=1 [micron], alpha=1.15, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_Nozawa_2007_gra[j])+" "+
                        "{:14.6e}".format(C_sca_Nozawa_2007_gra[j])+" "+
                        "{:14.6e}".format(g_Nozawa_2007_gra[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Marassi_2019_sil.dat','w') as f:
            f.write('# Marassi et al. 2019 silicates\n')
            f.write('# Log-normal distribution\n')
            f.write('# with a0=0.025 [micron], amin=0.001 [micron], amax=1 [micron], alpha=0.1, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_Marassi_2019_sil[j])+" "+
                        "{:14.6e}".format(C_sca_Marassi_2019_sil[j])+" "+
                        "{:14.6e}".format(g_Marassi_2019_sil[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Marassi_2019_gra.dat','w') as f:
            f.write('# Marassi et al. 2019 graphite\n')
            f.write('# Log-normal distribution\n')
            f.write('# with a0=0.075 [micron], amin=0.001 [micron], amax=1 [micron], alpha=0.1, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_Marassi_2019_gra[j])+" "+
                        "{:14.6e}".format(C_sca_Marassi_2019_gra[j])+" "+
                        "{:14.6e}".format(g_Marassi_2019_gra[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_ramses_silLarge.dat','w') as f:
            f.write('# RAMSES silicates\n')
            f.write('# Modified log-normal distribution (Hirashita 2015)\n')
            f.write('# with a0=0.1 [micron], amin=5e-3 [micron], amax=1 [micron], alpha=0.75, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_ramses_silLarge[j])+" "+
                        "{:14.6e}".format(C_sca_ramses_silLarge[j])+" "+
                        "{:14.6e}".format(g_ramses_silLarge[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_ramses_graLarge.dat','w') as f:
            f.write('# RAMSES graphite\n')
            f.write('# Modified log-normal distribution (Hirashita 2015)\n')
            f.write('# with a0=0.1 [micron], amin=5e-3 [micron], amax=1 [micron], alpha=0.75, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_ramses_graLarge[j])+" "+
                        "{:14.6e}".format(C_sca_ramses_graLarge[j])+" "+
                        "{:14.6e}".format(g_ramses_graLarge[j])+'\n')
            f.close()

    axes[0,1].set_yscale('log')
    axes[0,1].set_xscale('log')
    axes[0,1].tick_params(labelsize=14)
    axes[0,1].xaxis.set_ticks_position('both')
    axes[0,1].yaxis.set_ticks_position('both')
    axes[0,1].minorticks_on()
    axes[0,1].tick_params(which='both',axis="both",direction="in")
    axes[0,1].yaxis.set_label_position("right")
    axes[0,1].yaxis.tick_right()
    axes[0,1].set_ylim([4e-17,3e-9])

    axes[1,1].set_yscale('log')
    axes[1,1].set_xscale('log')
    axes[1,1].tick_params(labelsize=14)
    axes[1,1].xaxis.set_ticks_position('both')
    axes[1,1].yaxis.set_ticks_position('both')
    axes[1,1].minorticks_on()
    axes[1,1].tick_params(which='both',axis="both",direction="in")
    axes[1,1].yaxis.set_label_position("right")
    axes[1,1].yaxis.tick_right()
    axes[1,1].set_ylim([4e-17,3e-9])

    dummy_lines = [axes[1,1].plot([],[],color='k',linestyle='-',label=r'$C_{\rm abs}$')[0],
                   axes[1,1].plot([],[],color='k',linestyle='--',label=r'$C_{\rm sca}$')[0],
                   axes[1,1].plot([],[],color='k',linestyle=':',label=r'$g\times 10^{-13}$')[0]]
    first_legend = axes[1,1].legend(handles=dummy_lines, loc='lower left', frameon=False, fontsize=14)
    axes[1,1].add_artist(first_legend)


    # 5. Add text indicating that the top row is for silicates while the bottom row is for graphite
    axes[0,1].text(0.75, 0.91, 'Silicates', verticalalignment='top', horizontalalignment='left',
                   transform=axes[0,1].transAxes,fontsize=16,fontdict={'weight': 'bold'})
    axes[1,1].text(0.75, 0.91, 'Graphite', verticalalignment='top', horizontalalignment='left',
                   transform=axes[1,1].transAxes,fontsize=16,fontdict={'weight': 'bold'})

    # 6. Adjust figure and save
    fig.subplots_adjust(top=0.99,bottom=0.09,left=0.08,right=0.92,hspace=0,wspace=0)
    fig.savefig(folder+'/cross_section_sne.pdf', format='pdf', dpi=300)
