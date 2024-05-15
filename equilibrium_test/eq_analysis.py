"""
ANALYSIS OF EQUILIBRIUM TESTS FOR Dusty-PRISM

The scripts included in this Python file are used for the reading of the outputs for the equlibrium tests
of the Dusty-PRISM version of RAMSES-RTZ. This allows for a check of the evolution of ions',' metal densities','
molecules and dust at a fixed density and varying temperature.

By: F. Rodriguez Montero (currodri@gmail.com)
"""

# Import required libraries
import numpy as np
import yt
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import seaborn as sns
sns.set(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
})

yt.set_log_level("critical")
from unyt import mh,g

amu_to_g = 1.66054e-24    # atomic mass units in grams
mO_NIST_amu = 15.9994     # oxygen molecular weight [amu]
mN_NIST_amu = 14.0067   # nitrogen molecular weight [amu]
mC_NIST_amu = 12.0107     # carbon molecular weight [amu]
mMg_NIST_amu = 24.305   # magnesium molecular weight [amu]
mSi_NIST_amu = 28.0855    # silicon molecular weight [amu]
mS_NIST_amu = 32.065      # sulfur molecular weight [amu]
mFe_NIST_amu = 55.854        # iron molecular weight [amu]
mNe_NIST_amu = 20.1797       # neon molecular weight [amu]
mCa_NIST_amu = 40.078        # calcium molecular weight [amu]

mPAH = 4./3.*np.pi*2*(5e-8)**3. * g
mCSmall = 4./3.*np.pi*2.2*(5e-7)**3. * g
mCLarge = 4./3.*np.pi*2.2*(1e-6)**3. * g
mSilSmall = 4./3.*np.pi*3.3*(5e-7)**3. * g
mSilLarge = 4./3.*np.pi*3.3*(1e-6)**3. * g

# Define fields for yt
basic_hydro       = ['Density','x-velocity','y-velocity','z-velocity','radiation_pressure','Pressure']
metal_massfrac    = ['FeMassFrac','OMassFrac','NMassFrac','MgMassFrac','NeMassFrac',
                    'SiMassFrac','CaMassFrac','CMassFrac','SMassFrac']
co_massfrac       = ['COMassFrac']
metal_ion         = ['OI','OII','OIII','OIV','OV','OVI','OVII','OVIII',
                  'NI','NII','NIII','NIV','NV','NVI','NVII',
                  'CI','CII','CIII','CIV','CV','CVI',
                  'MgI','MgII','MgIII','MgIV','MgV','MgVI','MgVII','MgVIII','MgIX','MgX',
                  'SiI','SiII','SiIII','SiIV','SiV','SiVI','SiVII','SiVIII','SiIX','SiX','SiXI',
                  'SI','SII','SIII','SIV','SV','SVI','SVII','SVIII','SIX','SX','SXI',
                  'FeI','FeII','FeIII','FeIV','FeV','FeVI','FeVII','FeVIII','FeIX','FeX','FeXI',
                  'NeI','NeII','NeIII','NeIV','NeV','NeVI','NeVII','NeVIII','NeIX','NeX']
ions              = ['HI','HII','HeII','HeIII']
dust_densities    = ['pahs','CSmall','CLarge','SilSmall','SilLarge']
noadvect          = ['cooling_time','temperature','cooling_rate','heating_rate',
                  'cooling_primordial','cooling_fine_structure','cooling_CII',
                  'cooling_OI','cooling_CO','cooling_dust','cooling_dust_rec',
                  'heating_cr','heating_pe','heating_h2','heating_ct','dust_temperature']
def setup_yt(dust,pahs):
    @yt.derived_field(name='nH', sampling_type="cell", units='cm**-3',force_override=True)
    def _nH(field,data):
        n = data[('ramses','OMassFrac')] + data[('ramses','NMassFrac')] + \
            data[('ramses','CMassFrac')] + data[('ramses','MgMassFrac')] + \
            data[('ramses','SiMassFrac')]+ data[('ramses','SMassFrac')] + \
            data[('ramses','FeMassFrac')]+data[('ramses','NeMassFrac')] + \
            data[('ramses','COMassFrac')]
        try:
            n = n + data[('ramses','pahs')]+ data[('ramses','CSmall')] + \
                data[('ramses','CLarge')]  + data[('ramses','SilSmall')] + \
                data[('ramses','SilLarge')]
        except:
            pass
        n = (data[('gas','density')] * (1. - n))/mh
        return n

    @yt.derived_field(name='nHI', sampling_type="cell", units='cm**-3',force_override=True)
    def _nHI(field,data):
        return data[('gas','nH')] * data[('ramses','HI')]
    
    @yt.derived_field(name='nH2', sampling_type="cell", units='cm**-3',force_override=True)
    def _nH2(field,data):
        return data[('gas','nH')] * (1. - data[('ramses','HI')] - data[('ramses','HII')])/2.

    @yt.derived_field(name='nCO', sampling_type="cell", units='cm**-3',force_override=True)
    def _nCO(field,data):
        return (data[('gas','density')] * data[('ramses','COMassFrac')]) / (mC_NIST_amu*amu_to_g*g)

    @yt.derived_field(name='nCI', sampling_type="cell", units='cm**-3',force_override=True)
    def _nCI(field,data):
        return (data[('gas','density')] * data[('ramses','CMassFrac')] * data[('ramses','CI')]) / (mC_NIST_amu*amu_to_g*g)

    @yt.derived_field(name='nCII', sampling_type="cell", units='cm**-3',force_override=True)
    def _nCII(field,data):
        return (data[('gas','density')] * data[('ramses','CMassFrac')] * data[('ramses','CII')]) / (mC_NIST_amu*amu_to_g*g)
    if pahs:
        @yt.derived_field(name='nPAH', sampling_type="cell", units='cm**-3',force_override=True)
        def _nPAH(field,data):
            return (data[('gas','density')] * data[('ramses','pahs')] ) / (mPAH)
    if dust:

        @yt.derived_field(name='nCSmall', sampling_type="cell", units='cm**-3',force_override=True)
        def _nCSmall(field,data):
            return (data[('gas','density')] * data[('ramses','CSmall')] ) / (mCSmall)

        @yt.derived_field(name='nCLarge', sampling_type="cell", units='cm**-3',force_override=True)
        def _nCLarge(field,data):
            return (data[('gas','density')] * data[('ramses','CLarge')] ) / (mCLarge)

        @yt.derived_field(name='nSilSmall', sampling_type="cell", units='cm**-3',force_override=True)
        def _nSilSmall(field,data):
            return (data[('gas','density')] * data[('ramses','SilSmall')] ) / (mSilSmall)

        @yt.derived_field(name='nSilLarge', sampling_type="cell", units='cm**-3',force_override=True)
        def _nSilLarge(field,data):
            return (data[('gas','density')] * data[('ramses','SilLarge')] ) / (mSilLarge)


def plot_single_var(my_fields,varname='temperature'):
    """This function plots the evolution of a particular variable for the range of
    densities given.

    Args:
        my_fields (list): List of str containing the field names
        varname (str): Variable name to be plotted
    """
    
    # 1. Get the outputs in the directory
    outputs = sorted(list(filter(lambda file: file.startswith('output'), os.listdir())),key=lambda x: int(x.split('_')[-1]))
    output_dir = os.getcwd()
    
    # 2. Load snapshots
    sims = [yt.load(f'{output_dir}/output_{str(out.split("_")[-1])}',fields=my_fields) for out in outputs]
    
    # 3. Get raw data
    ntime = len(outputs)
    density = sims[-1].all_data()['ramses','Density'].to('g/cm**3')/mh
    ndensity = len(density)
    data = np.zeros((ndensity,ntime))
    times = np.zeros(ntime)
    for t in range(0,ntime):
        ds = sims[t]
        times[t] = ds.current_time.to('Myr')
        try:
            raw_ad = ds.all_data()[('gas',varname)]
        except:
            raw_ad = ds.all_data()[('ramses',varname)]
        for d in range(0,ndensity):
            data[d,t] = raw_ad[d]

    # 4. Sort in ascending density
    sort_dens = np.argsort(density)
    density = density[sort_dens]
    data = data[sort_dens,:]
    print(data,data.min(),data.max())

    # 5. Plot data
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(varname, fontsize=16)
    ax.set_xlabel(r'$t$ [Myr]',fontsize=16)
    ax.set_yscale('log')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_ylim([data.min(),data.max()])
    ax.set_xlim([times.min(),times.max()])

    segs = [np.column_stack([times,data[d,:]]) for d in range(ndensity)]
    line_segments = LineCollection(segs,array=density.to('cm**-3').d,
                                   norm=mpl.colors.LogNorm(vmin=density.min(),
                                                           vmax=density.max()),
                                   linestyle='solid',cmap='inferno')
    ax.add_collection(line_segments)
    axcb = fig.colorbar(line_segments)
    axcb.set_label(r'$n_H$ [cm$^{-3}$]',fontsize=16)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.95)
    fig.savefig('eq_evo_'+str(varname)+'.png', format='png', dpi=300)

def plot_n_eq_value(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final column density
    equilibrium values for the desired set of variables

    Args:
        my_fields (list): List of str containing the fields to load
    """
    from unyt import mh
    if dust:
        varnames = ['nH','nH2','nCO','nCI','nCII',
                    'nPAH','nCSmall','nCLarge','nSilSmall','nSilLarge']
    else:
        varnames = ['nH','nH2','nCO','nCI','nCII']

    # 1. Get the outputs in the directory
    cwd = os.getcwd()
    os.chdir(f'./{simname}')
    outputs = sorted(list(filter(lambda file: file.startswith('output'), os.listdir())),key=lambda x: int(x.split('_')[-1]))
    output_dir = os.getcwd()
    
    # 2. Load two last outputs
    sims = [yt.load(f'{output_dir}/output_{str(out.split("_")[-1])}',fields=my_fields) for out in outputs[-2:]]
    sims = [yt.load(f'{output_dir}/output_{str(outputs[0].split("_")[-1])}',fields=my_fields)] + sims
    
    # 3. Get raw data
    density = sims[-1].all_data()[('gas','density')].to('g/cm**3')/mh
    ndensity = len(density)
    nvars = len(varnames)
    data = np.zeros((nvars,ndensity,3))
    for t in range(0,3):
        ds = sims[t]
        for v in range(0, nvars):
            try:
                raw_ad = ds.all_data()[('gas',varnames[v])]
            except:
                raw_ad = ds.all_data()[('ramses',varnames[v])]
            for d in range(0,ndensity):
                data[v,d,t] = raw_ad[d].to('cm**-3')
                if t==2:
                    diff = abs(data[v,d,1] - data[v,d,2]) / data[v,d,2]
                    if diff >= conv_crit:
                        print(f'Variable {varnames[v]} has not converged yet (err={diff},nH={data[0,d,t]})!')
    
    # 4. Sort data
    sort_dens = np.argsort(density)
    data = data[:,sort_dens,:]
    # 4. Plot data
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$\log{n/n_{{\rm H}}}$', fontsize=16)
    ax.set_xlabel(r'$\log{n_{{\rm H}}/[{\rm cm}^{-3}]}$',fontsize=16)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    #ax.set_xlim([-4,4])
    ax.set_ylim([-12,-2])

    for v in range(1, nvars):
        p = ax.plot(np.log10(data[0,:,0]),np.log10(data[v,:,2]/data[0,:,0]),label=varnames[v])
        ax.plot(np.log10(data[0,:,0]),np.log10(data[v,:,0]/data[0,:,0]),linestyle='--',alpha=0.6,color=p[0].get_color())

    ax.legend(loc='best',fontsize=12,frameon=False,ncol=3)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.95)
    os.chdir(cwd)
    fig.savefig(f'final_eq_column_densities_{simname}.png', format='png', dpi=300)
    
def plot_n_init_value(my_fields,dust,simname):
    """This function plots the initial column density
    equilibrium values for the desired set of variables

    Args:
        my_fields (list): List of str containing the fields to load
    """
    from unyt import mh
    if dust:
        varnames = ['nH','nH2','nCO','nCI','nCII',
                    'nPAH','nCSmall','nCLarge','nSilSmall','nSilLarge']
    else:
        varnames = ['nH','nH2','nCO','nCI','nCII']

    # 1. Get the outputs in the directory
    cwd = os.getcwd()
    os.chdir(f'./{simname}')
    outputs = sorted(list(filter(lambda file: file.startswith('output'), os.listdir())),key=lambda x: int(x.split('_')[-1]))
    output_dir = os.getcwd()
    
    # 2. Load two last outputs
    sim = yt.load(f'{output_dir}/output_{str(outputs[0].split("_")[-1])}',fields=my_fields)
    
    # 3. Get raw data
    density = sim.all_data()[('gas','density')].to('g/cm**3')/mh
    ndensity = len(density)
    nvars = len(varnames)
    data = np.zeros((nvars,ndensity))
    ds = sim
    for v in range(0, nvars):
        try:
            raw_ad = ds.all_data()[('gas',varnames[v])]
        except:
            raw_ad = ds.all_data()[('ramses',varnames[v])]
        for d in range(0,ndensity):
            data[v,d] = raw_ad[d].to('cm**-3')    
    # 4. Sort data
    sort_dens = np.argsort(density)
    data = data[:,sort_dens]
    # 4. Plot data
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$\log{n/[{\rm cm}^{-3}]}$', fontsize=16)
    ax.set_xlabel(r'$\log{n_{{\rm H}}/[{\rm cm}^{-3}]}$',fontsize=16)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xlim([-4,4])
    ax.set_ylim([-10,5])

    for v in range(1, nvars):
        ax.plot(np.log10(data[0,:]),np.log10(data[v,:]),label=varnames[v])

    ax.legend(loc='best',fontsize=12,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.95)
    os.chdir(cwd)
    fig.savefig(f'initial_column_densities_{simname}.png', format='png', dpi=300)

def plot_T(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final temperature equilibrium values

    Args:
        my_fields (list): List of str containing the fields to load
    """
    from unyt import mh
    nolist = False
    if not isinstance(simname,list):
        simname = [simname]
        nolist = True

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$\log{T/[{\rm K}]}$', fontsize=16)
    ax.set_xlabel(r'$\log{n_{{\rm H}}/[{\rm cm}^{-3}]}$',fontsize=16)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xlim([-4,5.5])

    for sim in simname:
        # 1. Get the outputs in the directory
        cwd = os.getcwd()
        os.chdir(f'./{sim}')
        outputs = sorted(list(filter(lambda file: file.startswith('output'), os.listdir())),key=lambda x: int(x.split('_')[-1]))
        output_dir = os.getcwd()
        
        # 2. Load two last outputs
        sims = [yt.load(f'{output_dir}/output_{str(out.split("_")[-1])}',fields=my_fields) for out in outputs[-2:]]
        
        # 3. Get raw data
        density = sims[-1].all_data()[('gas','nH')].to('cm**-3')
        ndensity = len(density)
        if args.dust:
            data = np.zeros((ndensity,2,2))
        else:
            data = np.zeros((ndensity,2))
        for t in range(0,2):
            ds = sims[t]
            if args.dust:
                try:
                    raw_ad = ds.all_data()[('gas','temperature')]
                except:
                    raw_ad = ds.all_data()[('ramses','temperature')]
                for d in range(0,ndensity):
                    data[d,0,t] = raw_ad[d].to('K')
                    if t==1:
                        diff = abs(data[d,0,0] - data[d,0,1]) / data[d,0,1]
                        if diff >= conv_crit:
                            print(f'Temperature has not converged yet (err={diff},nH={data[d,0,t]}) for {sim}!')
                try:
                    raw_ad = ds.all_data()[('gas','dust_temperature')]
                except:
                    raw_ad = ds.all_data()[('ramses','dust_temperature')]
                for d in range(0,ndensity):
                    data[d,1,t] = raw_ad[d]
                    if t==1:
                        diff = abs(data[d,1,0] - data[d,1,1]) / data[d,1,1]
                        if diff >= conv_crit:
                            print(f'Dust temperature has not converged yet (err={diff},nH={data[d,1,t]}) for {sim}!')
            else:
                try:
                    raw_ad = ds.all_data()[('gas','temperature')]
                except:
                    raw_ad = ds.all_data()[('ramses','temperature')]
                for d in range(0,ndensity):
                    data[d,t] = raw_ad[d].to('K')
                    if t==1:
                        diff = abs(data[d,0] - data[d,1]) / data[d,1]
                        if diff >= conv_crit:
                            print(f'Temperature has not converged yet (err={diff},nH={data[d,t]}) for {sim}!')
        
        # 4. Sort data
        sort_dens = np.argsort(density)
        if args.dust:
            data = data[sort_dens,:,:]
            density = density[sort_dens]

            ax.plot(np.log10(density),np.log10(data[:,0,1]),marker='o',
                    markersize=2,markerfacecolor='None',linestyle='none',label=sim.split('/')[-1])
            ax.plot(np.log10(density),np.log10(data[:,1,1]),marker='x',
                    markersize=2,markerfacecolor='None',linestyle='none',label=sim.split('/')[-1])
        else:
            data = data[sort_dens,:]
            density = density[sort_dens]

            ax.plot(np.log10(density),np.log10(data[:,1]),marker='o',
                    markersize=2,markerfacecolor='None',linestyle='none',label=sim.split('/')[-1])
        os.chdir(cwd)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.95)
    ax.legend(loc='best',fontsize=12,frameon=False)
        
    if not nolist:
        print('Making comparison plot in final_temperature_comparison.png')
        fig.savefig('final_temperature_comparison.png', format='png', dpi=300)
    else:
        print(f'final_temperature_{simname[0].split("/")[-1]}.png')
        fig.savefig(f'final_temperature_{simname[0].split("/")[-1]}.png', format='png', dpi=300)

def plot_lambda_tot_value(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final heating and cooling rate
    contributions by different processes to the global heating and
    cooling

    Args:
        my_fields (list): List of str containing the fields to load
    """
    from unyt import mh
    varnames_cooling = ['cooling_rate','heating_rate']

    # 1. Get the outputs in the directory

    os.chdir(f'./{simname}')
    outputs = sorted(list(filter(lambda file: file.startswith('output'), os.listdir())),key=lambda x: int(x.split('_')[-1]))
    output_dir = os.getcwd()
    
    # 2. Load two last outputs
    sims = [yt.load(f'{output_dir}/output_{str(out.split("_")[-1])}',fields=my_fields) for out in outputs[-2:]]
    
    # 3. Get raw data
    density = sims[-1].all_data()[('gas','nH')].to('cm**-3')
    ndensity = len(density)
    nvars = len(varnames_cooling)
    data_cooling = np.zeros((nvars,ndensity,2))
    for t in range(0,2):
        ds = sims[t]
        for v in range(0, nvars):
            try:
                raw_ad = ds.all_data()[('gas',varnames_cooling[v])]
            except:
                raw_ad = ds.all_data()[('ramses',varnames_cooling[v])]
            for d in range(0,ndensity):
                data_cooling[v,d,t] = raw_ad[d]
                if t==1:
                    diff = abs(data_cooling[v,d,0] - data_cooling[v,d,1]) / data_cooling[v,d,1]
                    if diff >= conv_crit:
                        print(f'Variable {varnames_cooling[v]} has not converged yet (err={diff},nH={density[d]})!')
    
    # 4. Sort data
    sort_dens = np.argsort(density)
    data_cooling = data_cooling[:,sort_dens,:]
    density = density[sort_dens]

    # 5. Plot data
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,6), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$\log{n\Lambda_{\rm tot} \quad {\rm or }\quad \Gamma}$ [erg/s]', fontsize=16)
    ax.set_xlabel(r'$\log{n_{{\rm H}}/[{\rm cm}^{-3}]}$',fontsize=16)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xlim([-4,4])
    # ax.set_ylim([-3,0])
    
    nvars = len(varnames_cooling)
    for v in range(0, nvars):
        if varnames_cooling[v].split('_')[0] == 'cooling':
            y = data_cooling[v,:,1]
        else:
            y = data_cooling[v,:,1]
        print(y)
        ax.plot(np.log10(density),np.log10(y),label=varnames_cooling[v])

    ax.legend(loc='best',fontsize=12,frameon=False)

    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.95,hspace=0)
    os.chdir('../')
    fig.savefig(f'final_tot_cooling_{simname}.png', format='png', dpi=300)

def plot_lambda_eq_value(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final heating and cooling rate
    contributions by different processes to the global heating and
    cooling

    Args:
        my_fields (list): List of str containing the fields to load
    """
    from unyt import mh
    varnames_cooling = ['cooling_rate',
                        'cooling_primordial','cooling_fine_structure','cooling_CII',
                        'cooling_OI','cooling_CO','cooling_dust','cooling_dust_rec']
    varnames_heating = ['heating_rate','heating_cr','heating_pe','heating_h2','heating_ct']

    # 1. Get the outputs in the directory
    cwd = os.getcwd()
    os.chdir(f'./{simname}')
    outputs = sorted(list(filter(lambda file: file.startswith('output'), os.listdir())),key=lambda x: int(x.split('_')[-1]))
    output_dir = os.getcwd()
    
    # 2. Load two last outputs
    sims = [yt.load(f'{output_dir}/output_{str(out.split("_")[-1])}',fields=my_fields) for out in outputs[-2:]]
    
    # 3. Get raw data
    density = sims[-1].all_data()[('gas','nH')].to('cm**-3')
    ndensity = len(density)
    nvars = len(varnames_cooling)
    data_cooling = np.zeros((nvars,ndensity,2))
    for t in range(0,2):
        ds = sims[t]
        for v in range(0, nvars):
            try:
                raw_ad = ds.all_data()[('gas',varnames_cooling[v])]
            except:
                raw_ad = ds.all_data()[('ramses',varnames_cooling[v])]
            for d in range(0,ndensity):
                data_cooling[v,d,t] = raw_ad[d]
                if t==1:
                    diff = abs(data_cooling[v,d,0] - data_cooling[v,d,1]) / data_cooling[v,d,1]
                    if diff >= conv_crit:
                        print(f'Variable {varnames_cooling[v]} has not converged yet (err={diff},nH={density[d]})!')
    nvars = len(varnames_heating)
    data_heating = np.zeros((nvars,ndensity,2))
    for t in range(0,2):
        ds = sims[t]
        for v in range(0, nvars):
            try:
                raw_ad = ds.all_data()[('gas',varnames_heating[v])]
            except:
                raw_ad = ds.all_data()[('ramses',varnames_heating[v])]
            for d in range(0,ndensity):
                data_heating[v,d,t] = raw_ad[d]
                if t==1:
                    diff = abs(data_heating[v,d,0] - data_heating[v,d,1]) / data_heating[v,d,1]
                    if diff >= conv_crit:
                        print(f'Variable {varnames_heating[v]} has not converged yet (err={diff},nH={density[d]})!')
    
    # 4. Sort data
    sort_dens = np.argsort(density)
    data_cooling = data_cooling[:,sort_dens,:]
    data_heating = data_heating[:,sort_dens,:]
    density = density[sort_dens]

    # 5. Plot data
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6,9), dpi=300, facecolor='w', edgecolor='k')

    ax[0].set_ylabel(r'$\log{\Lambda/\Lambda_{\rm tot}}$', fontsize=16)
    ax[0].tick_params(labelsize=12)
    ax[0].xaxis.set_ticks_position('both')
    ax[0].yaxis.set_ticks_position('both')
    ax[0].minorticks_on()
    ax[0].tick_params(which='both',axis="both",direction="in")
    ax[0].set_xlim([-4,4])
    ax[0].set_ylim([-3,0])

    nvars = len(varnames_cooling)
    for v in range(1, nvars):
       ax[0].plot(np.log10(density),np.log10(data_cooling[v,:,1]/data_cooling[0,:,1]),label=varnames_cooling[v])

    ax[0].legend(loc='best',fontsize=12,frameon=False)

    ax[1].set_ylabel(r'$\log{\Gamma/\Gamma_{\rm tot}}$', fontsize=16)
    ax[1].set_xlabel(r'$\log{n_{{\rm H}}/[{\rm cm}^{-3}]}$',fontsize=16)
    ax[1].tick_params(labelsize=12)
    ax[1].xaxis.set_ticks_position('both')
    ax[1].yaxis.set_ticks_position('both')
    ax[1].minorticks_on()
    ax[1].tick_params(which='both',axis="both",direction="in")
    ax[1].set_xlim([-4,4])
    ax[1].set_ylim([-3,0])

    nvars = len(varnames_heating)
    for v in range(1, nvars):
       ax[1].plot(np.log10(density),np.log10(data_heating[v,:,1]/data_heating[0,:,1]),label=varnames_heating[v])

    ax[1].legend(loc='best',fontsize=12,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.95,hspace=0)
    os.chdir(cwd)
    print(f'final_eq_cooling_{simname}.png')
    fig.savefig(f'final_eq_cooling_{simname}.png', format='png', dpi=300)
    
def plot_T_for_proposal(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final temperature equilibrium values

    Args:
        my_fields (list): List of str containing the fields to load
    """
    from unyt import mh
    nolist = False
    if not isinstance(simname,list):
        simname = [simname]
        nolist = True

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(3,4), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$\log{(T/{\rm K})}$', fontsize=16)
    ax.set_xlabel(r'$\log{(n_{{\rm H}}/{\rm cm}^{-3})}$',fontsize=16)
    ax.tick_params(labelsize=16)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xlim([-2,4])
    ax.set_ylim([1.7,4.3])

    for sim in simname:
        # 1. Get the outputs in the directory
        cwd = os.getcwd()
        os.chdir(f'./{sim}')
        outputs = sorted(list(filter(lambda file: file.startswith('output'), os.listdir())),key=lambda x: int(x.split('_')[-1]))
        output_dir = os.getcwd()
        
        # 2. Load two last outputs
        sims = [yt.load(f'{output_dir}/output_{str(out.split("_")[-1])}',fields=my_fields) for out in outputs[-2:]]
        
        # 3. Get raw data
        density = sims[-1].all_data()[('gas','nH')].to('cm**-3')
        ndensity = len(density)
        if args.dust:
            data = np.zeros((ndensity,2,2))
        else:
            data = np.zeros((ndensity,2))
        for t in range(0,2):
            ds = sims[t]
            if args.dust:
                try:
                    raw_ad = ds.all_data()[('gas','temperature')]
                except:
                    raw_ad = ds.all_data()[('ramses','temperature')]
                for d in range(0,ndensity):
                    data[d,0,t] = raw_ad[d].to('K')
                    if t==1:
                        diff = abs(data[d,0,0] - data[d,0,1]) / data[d,0,1]
                        if diff >= conv_crit:
                            print(f'Temperature has not converged yet (err={diff},nH={data[d,0,t]})!')
                try:
                    raw_ad = ds.all_data()[('gas','dust_temperature')]
                except:
                    raw_ad = ds.all_data()[('ramses','dust_temperature')]
                for d in range(0,ndensity):
                    data[d,1,t] = raw_ad[d]
                    if t==1:
                        diff = abs(data[d,1,0] - data[d,1,1]) / data[d,1,1]
                        if diff >= conv_crit:
                            print(f'Dust temperature has not converged yet (err={diff},nH={data[d,1,t]})!')
            else:
                try:
                    raw_ad = ds.all_data()[('gas','temperature')]
                except:
                    raw_ad = ds.all_data()[('ramses','temperature')]
                for d in range(0,ndensity):
                    data[d,t] = raw_ad[d].to('K')
                    if t==1:
                        diff = abs(data[d,0] - data[d,1]) / data[d,1]
                        if diff >= conv_crit:
                            print(f'Temperature has not converged yet (err={diff},nH={data[d,t]})!')
        
        # 4. Sort data
        sort_dens = np.argsort(density)
        if args.dust:
            data = data[sort_dens,:,:]
            density = density[sort_dens]

            ax.plot(np.log10(density),np.log10(data[:,0,1]),linestyle='-',linewidth=3,label=sim.split('/')[-1])
            ax.plot(np.log10(density),np.log10(data[:,1,1]),linestyle='-',linewidth=3,label=sim.split('/')[-1])
        else:
            data = data[sort_dens,:]
            density = density[sort_dens]

            ax.plot(np.log10(density),np.log10(data[:,1]),marker='o',
                    markersize=2,markerfacecolor='None',linestyle='none',label=sim.split('/')[-1])
        os.chdir(cwd)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.22,right=0.98)
        
    if not nolist:
        print('Making comparison plot in final_temperature_comparison.eps')
        fig.savefig('final_temperature_comparison.eps', format='eps', dpi=300)
    else:
        print(f'final_temperature_{simname[0].split("/")[-1]}.eps')
        fig.savefig(f'final_temperature_{simname[0].split("/")[-1]}.eps', format='eps', dpi=300)
            
if __name__ == '__main__':

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description='Plotting evolution of equilibrium tests of Dusty-PRISM')
    parser.add_argument('type', type=str, help='Type of plot to do')
    parser.add_argument('--varname', type=str, nargs='+', default='temperature', help='Variable name to plot.')
    parser.add_argument('--dust',action='store_true',help='Use if simulation includes dust.')
    parser.add_argument('--pahs',action='store_true',help='Use if simulation includes PAHs.')
    parser.add_argument('--simname', type=str, default='dust',nargs='+', help='Simulation name.')
    args = parser.parse_args()
    
    if len(args.simname) == 1:
        args.simname = args.simname[0]
    if args.dust and args.pahs:
        fields = basic_hydro + metal_massfrac + dust_densities + co_massfrac + metal_ion + ions + noadvect
        setup_yt(True,True)
    elif args.dust:
        fields = basic_hydro + metal_massfrac + dust_densities[1:] + co_massfrac + metal_ion + ions + noadvect
        setup_yt(True,False)
    else:
        fields = basic_hydro + metal_massfrac + co_massfrac + metal_ion + ions + ['unknown1','unknown2','unknown3','unknown4','unknown5'] + noadvect
        setup_yt(False,False)
    
    if args.type == 'single':
        plot_T(fields,args.dust,args.simname)
        plot_single_var(fields,args.varname[0])
    elif args.type == 'column_density_1':
        plot_T(fields,args.dust,args.simname)
        if args.dust:
            plot_n_eq_value(fields,True,args.simname)
        else:
            plot_n_eq_value(fields,False,args.simname)
    elif args.type == 'cooling':
        plot_T(fields,args.dust,args.simname)
        plot_lambda_eq_value(fields,args.dust,args.simname)
    elif args.type == 'tot_cooling':
        plot_T(fields,args.dust,args.simname)
        plot_lambda_tot_value(fields,args.dust,args.simname)
    elif args.type == 'temperature':
        plot_T(fields,args.dust,args.simname)
    elif args.type == 'temperature_proposal':
        plot_T_for_proposal(fields,args.dust,args.simname)
    elif args.type == 'initial_density':
        plot_n_init_value(fields,args.dust,args.simname)