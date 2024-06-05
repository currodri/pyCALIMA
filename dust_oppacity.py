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
from dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,LogNormal_Distribution
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


    return data,columns,dust_type

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
                w = w[::-1]
                # Convert cross section from micron^2 to cm^2
                Q_abs_eff = Q_abs_eff[::-1] * 1e-8
                Q_sca_eff = Q_sca_eff[::-1] * 1e-8 
                Q_rp_eff = Q_rp_eff[::-1] * 1e-8
                f = open('averaged_cross_section_%.4f_micron_%s'%(dist[i].a0,filename.split('/')[-1]), 'w', encoding="utf-8")
                f.write("{:8d}".format(nwav)+'\n')
                for j in range(0,nwav):
                    f.write("{:14.6e}".format(w[j]/1e-4)+'\n')
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
    axes[0].set_ylabel(r'$C_{\rm abs}$', fontsize=16)
    axes[1].set_ylabel(r'$C_{\rm sca}$', fontsize=16)
    axes[2].set_ylabel(r'$C_{\rm rp}$', fontsize=16)
    axes[2].set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=16)
    axes[0].legend(loc='best',fontsize=14,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.06,left=0.13,right=0.99,hspace=0,wspace=0)
    fig.savefig('./cross_section_%s.pdf'%filename.split('/')[-1], format='pdf', dpi=300)

        