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

yt.set_log_level("critical")
from unyt import mh,g
from pycalima.plotting_style import use_calima_style

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

mPAHSmall = 4./3.*np.pi*2*(5e-8)**3. * g
mPAHLarge = 4./3.*np.pi*2*(1e-7)**3. * g
mCSmall = 4./3.*np.pi*2.2*(1e-6)**3. * g
mCLarge = 4./3.*np.pi*2.2*(1e-5)**3. * g
mSilSmall = 4./3.*np.pi*3.3*(5e-7)**3. * g
mSilLarge = 4./3.*np.pi*3.3*(1e-5)**3. * g

clean_name = {'no_dust':'No Dusty-PRISM',
                  'acc_chaabouni':r'$\alpha(T)$ by Chaabouni et al. (2012)',
                  'acc_LDW85':r'$\alpha(T)$ by Leitch-Devlin and D. A. Williams (1985)',
                  'acc_nhmax1d6':r'Accretion cut-off at $n_{\rm H}=10^6$ cm$^{-3}$',
                  'acc_cou':r'+ Coulomb enhancement',
                  'acc_coa':r'+ Coagulation',
                  'acc_coa_sha':r'+ Shattering',
                  'acss_cou':r'+ Sputtering + Coulomb',
                  'acss_turb_HA19': r'$\Delta V$ by Hirashita \& Aoyama (2019)',
                  'acss_turb_OC07': r'$\Delta V$ by our model',
                  'acss_turb_poppe': r'Coagulation by Poppe et al (1997)',
                  'acsst_ratd_lowtens': r'+ RATD',
                  'acsstr_h2': r'H$_2$ formation on grains',
                  'acsstrh_col': r'Updated collisional cooling',
                  'acsstrh_c': r'+ PAH Coalescence',
                  'acsstrh_c_diss':r'+ PAH photo-dissociation',
                  'acsstrh_cd_evap':r'+ PAH cluster evaporation',
                  'acsstrh_cde_free':r'+ PAH freezing',
                  'acsstrh_cdef_spu':r'+ PAH sputtering',
                  'acsstrh_cdefs_peh':r'+ PAH PEH',
                  'acsstrh_cdefsp_h2':r'+ PAH H$_2$ formation',
                  'pah_h2':r'+ PAH H$_2$ formation',
                  'pah_peh':r'PAH photoelectric heating'}

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
dust_densities    = ['PAHSmall','PAHLarge','CSmall','CLarge','SilSmall','SilLarge']
noadvect          = ['cooling_time','temperature','cooling_rate','heating_rate',
                  'cooling_primordial','cooling_fine_structure','cooling_CII',
                  'cooling_OI','cooling_CO','cooling_dust','cooling_dust_rec',
                  'heating_cr','heating_pe','heating_h2','heating_ct','dust_temperature',
                  'fionPAHSmall','fionPAHLarge']
def setup_yt(dust,pahs):
    @yt.derived_field(name='nH', sampling_type="cell", units='cm**-3',force_override=True)
    def _nH(field,data):
        n = data[('ramses','OMassFrac')] + data[('ramses','NMassFrac')] + \
            data[('ramses','CMassFrac')] + data[('ramses','MgMassFrac')] + \
            data[('ramses','SiMassFrac')]+ data[('ramses','SMassFrac')] + \
            data[('ramses','FeMassFrac')]+data[('ramses','NeMassFrac')] + \
            data[('ramses','COMassFrac')]
        try:
            n = n + data[('ramses','PAHSmall')] +data[('ramses','PAHLarge')]+ \
                data[('ramses','CSmall')] + \
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
        @yt.derived_field(name='nPAHSmall', sampling_type="cell", units='cm**-3',force_override=True)
        def _nPAHSmall(field,data):
            return (data[('gas','density')] * data[('ramses','PAHSmall')] ) / (mPAHSmall)
        @yt.derived_field(name='nPAHLarge', sampling_type="cell", units='cm**-3',force_override=True)
        def _nPAHLarge(field,data):
            return (data[('gas','density')] * data[('ramses','PAHLarge')] ) / (mPAHLarge)
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
    use_calima_style()
    
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
    fig.savefig('./plots/eq_evo_'+str(varname)+'.png', format='png', dpi=300)

def plot_n_eq_value(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final column density
    equilibrium values for the desired set of variables

    Args:
        my_fields (list): List of str containing the fields to load
    """
    use_calima_style()
    from unyt import mh
    if dust:
        varnames = ['nH','nH2','nCO','nCI','nCII',
                    'nPAHSmall','nPAHLarge','nCSmall','nCLarge','nSilSmall','nSilLarge']
        # varnames = ['nH','nSilLarge']
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
                    # if diff >= conv_crit:
                    #     print(f'Variable {varnames[v]} has not converged yet (err={diff},nH={data[0,d,t]})!')
    
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
    ax.set_ylim([-14,-2])

    for v in range(1, nvars):
        p = ax.plot(np.log10(data[0,:,0]),np.log10(data[v,:,2]/data[0,:,0]),label=varnames[v])
        ax.plot(np.log10(data[0,:,0]),np.log10(data[v,:,0]/data[0,:,0]),linestyle='--',alpha=0.6,color=p[0].get_color())
    ax.legend(loc='best',fontsize=12,frameon=False,ncol=3)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.95)
    os.chdir(cwd)
    fig.savefig(f'./plots/final_eq_column_densities_{simname}.png', format='png', dpi=300)
    
def plot_n_init_value(my_fields,dust,simname):
    """This function plots the initial column density
    equilibrium values for the desired set of variables

    Args:
        my_fields (list): List of str containing the fields to load
    """
    use_calima_style()
    from unyt import mh
    if dust:
        varnames = ['nH','nH2','nCO','nCI','nCII',
                    'nPAHSmall','nPAHLarge','nCSmall','nCLarge','nSilSmall','nSilLarge']
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
    fig.savefig(f'./plots/initial_column_densities_{simname}.png', format='png', dpi=300)

def plot_T(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final temperature equilibrium values

    Args:
        my_fields (list): List of str containing the fields to load
    """
    use_calima_style()
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
        fig.savefig('./plots/final_temperature_comparison.png', format='png', dpi=300)
    else:
        print(f'final_temperature_{simname[0].split("/")[-1]}.png')
        fig.savefig(f'./plots/final_temperature_{simname[0].split("/")[-1]}.png', format='png', dpi=300)

def plot_lambda_tot_value(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final heating and cooling rate
    contributions by different processes to the global heating and
    cooling

    Args:
        my_fields (list): List of str containing the fields to load
    """
    use_calima_style()
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
    fig.savefig(f'./plots/final_tot_cooling_{simname}.png', format='png', dpi=300)

def plot_lambda_eq_value(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final heating and cooling rate
    contributions by different processes to the global heating and
    cooling

    Args:
        my_fields (list): List of str containing the fields to load
    """
    use_calima_style()
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
    print(f'./plots/final_eq_cooling_{simname}.png')
    fig.savefig(f'./plots/final_eq_cooling_{simname}.png', format='png', dpi=300)
    
def plot_T_for_proposal(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final temperature equilibrium values

    Args:
        my_fields (list): List of str containing the fields to load
    """
    use_calima_style()
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
        fig.savefig('./plots/final_temperature_comparison.eps', format='eps', dpi=300)
    else:
        print(f'final_temperature_{simname[0].split("/")[-1]}.eps')
        fig.savefig(f'./plots/final_temperature_{simname[0].split("/")[-1]}.eps', format='eps', dpi=300)
        
def plot_for_thesis(my_fields,dust,simname,conv_crit=0.1):
    """This function plots the final temperature equilibrium values

    Args:
        my_fields (list): List of str containing the fields to load
    """
    use_calima_style()
    import matplotlib.gridspec as gridspec
    from unyt import mh
    nolist = False
    if not isinstance(simname,list):
        simname = [simname]
        nolist = True
        
    if dust:
        varnames = ['temperature','nH','nH2','nCO',
                    'nPAHSmall','nPAHLarge','nCSmall','nCLarge','nSilSmall','nSilLarge']
        # varnames = ['nH','nSilLarge']
    else:
        varnames = ['temperature','nH','nH2','nCO']
        
    Gerin15 = np.array([[55.783283190707735, 152.45686750100288],
                        [70.45508197126216, 130.45465399466],
                        [66.96780041562548, 99.84324605738185],
                        [83.19050251471889, 89.99285939717006],
                        [36.70519167987747, 104.35977492938696],
                        [34.32521669231977, 95.45700488715778],
                        [44.09385859819482, 89.9621503325627],
                        [66.94134089630272, 84.79121490024798],
                        [54.79117801380037, 91.31921236022974]
                        ])
        
    # Create a figure
    fig = plt.figure(figsize=(10, 6))

    # Define a GridSpec with 3 rows and 2 columns
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1, 1])

    # Large subplot on the left (spanning all rows in the first column)
    ax1 = fig.add_subplot(gs[:, 0])  # Span all rows in the first column

    # Smaller subplots on the right
    ax2 = fig.add_subplot(gs[0, 1])  # First row, second column
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim([1e-2,1e4])
    ax2.set_ylim([2e1,1e12])
    ax2.set_xticklabels([])
    ax3 = fig.add_subplot(gs[1, 1])  # Second row, second column
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_xlim([1e-2,1e4])
    ax3.set_xticklabels([])
    ax3.set_ylim([4e-3,3e1])
    ax4 = fig.add_subplot(gs[2, 1])  # Third row, second column
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.set_xlim([1e-2,1e4])
    ax4.set_xticklabels([])
    ax4.set_ylim([1e-1,1e1])
    ax5 = fig.add_subplot(gs[3, 1])  # Fourth row, second column
    ax5.set_yscale('log')
    ax5.set_xscale('log')
    ax5.set_xlim([1e-2,1e4])
    ax5.set_ylim([7e-2,3e0])
    ax5.set_xticks([1e-2,1e0,1e2,1e4],labels=['',r'$10^0$',r'$10^2$',r'$10^4$'])
    ax5.set_xlabel(r'$n_{{\rm H}} [{\rm cm}^{-3}]$',fontsize=16)

    ax1.set_ylabel(r'$ T [{\rm K}]$', fontsize=16)
    ax1.set_xlabel(r'$n_{{\rm H}} [{\rm cm}^{-3}]$',fontsize=16)
    ax1.tick_params(labelsize=16)
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.minorticks_on()
    ax1.tick_params(which='both',axis="both",direction="in")
    ax1.set_xlim([1e-2,1e4])
    ax1.set_ylim([40,1e4])
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax1.scatter(Gerin15[:,0],Gerin15[:,1],marker='o',s=20,color='r')
    
    # axes[1].set_yscale('log')
    # axes[1].set_xscale('log')
    # axes[1].set_ylabel(r'$\rho_X/\rho_X^0$', fontsize=16)
    # axes[1].set_xlabel(r'$n_{{\rm H}} [{\rm cm}^{-3}]$',fontsize=16)
    # axes[1].tick_params(labelsize=16)
    # axes[1].xaxis.set_ticks_position('both')
    # axes[1].yaxis.set_ticks_position('both')
    # axes[1].minorticks_on()
    # axes[1].tick_params(which='both',axis="both",direction="in")
    # axes[1].set_xlim([1e-2,1e4])
    
    # Choose a colormap
    colormap = plt.cm.tab20c  # You can use any colormap available in matplotlib

    # Generate a range of colors from the colormap
    colors = [colormap(i / len(simname)) for i in range(len(simname))]
    
    # inset Axes....
    x1, x2, y1, y2 = 1.1e2, 2e3, 70, 140  # subregion of the original image
    axins = ax1.inset_axes(
        [0.09, 0.05, 0.35, 0.4],
        xlim=(x1, x2), ylim=(y1, y2))#, xticklabels=[], yticklabels=[])
    axins.spines['top'].set_linewidth(1)     # Top axis
    axins.spines['bottom'].set_linewidth(1)  # Bottom axis
    axins.spines['left'].set_linewidth(1)    # Left axis
    axins.spines['right'].set_linewidth(1)   # Right axis
    # axins.set_yscale('log')
    # axins.set_xscale('log')
    axins.grid(True, which='both', linestyle='--', linewidth=0.5)
    axins.tick_params(labelsize=12)
    axins.xaxis.set_ticks_position('both')
    axins.yaxis.set_ticks_position('both')
    axins.minorticks_on()
    axins.tick_params(which='both',axis="both",direction="in")
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])

    # inset Axes....
    x1, x2, y1, y2 = 1.5e0, 3e1, 120, 500  # subregion of the original image
    axins2 = ax1.inset_axes(
        [0.5, 0.5, 0.47, 0.47],
        xlim=(x1, x2), ylim=(y1, y2))#, xticklabels=[], yticklabels=[])
    axins2.spines['top'].set_linewidth(1)     # Top axis
    axins2.spines['bottom'].set_linewidth(1)  # Bottom axis
    axins2.spines['left'].set_linewidth(1)    # Left axis
    axins2.spines['right'].set_linewidth(1)   # Right axis
    # axins.set_yscale('log')
    # axins.set_xscale('log')
    axins2.grid(True, which='both', linestyle='--', linewidth=0.5)
    axins2.tick_params(labelsize=12)
    axins2.xaxis.set_ticks_position('both')
    axins2.yaxis.set_ticks_position('both')
    axins2.minorticks_on()
    axins2.tick_params(which='both',axis="both",direction="in")
    axins2.set_yscale('log')

    for s,sim in enumerate(simname):
        # 1. Get the outputs in the directory
        cwd = os.getcwd()
        os.chdir(f'./{sim}')
        outputs = sorted(list(filter(lambda file: file.startswith('output'), os.listdir())),key=lambda x: int(x.split('_')[-1]))
        output_dir = os.getcwd()
        
        # 2. Load two last outputs
        print(f'Loading {sim}...')
        if 'no_dust' in sim and dust:
            tmp_varnames = ['temperature','nH','nH2','nCO']
            tmp_fields = basic_hydro + metal_massfrac + co_massfrac + metal_ion + ions + dust_densities + noadvect
        else:
            tmp_varnames = varnames
            tmp_fields = my_fields
        sims = [yt.load(f'{output_dir}/output_{str(out.split("_")[-1])}',fields=tmp_fields) for out in outputs[-2:]]
        sims = [yt.load(f'{output_dir}/output_{str(outputs[0].split("_")[-1])}',fields=tmp_fields)] + sims
        
        # 3. Get raw data
        density = sims[-1].all_data()[('gas','nH')].to('cm**-3')
        ndensity = len(density)
        nvars = len(tmp_varnames)
        data = np.zeros((ndensity,nvars,3))
        for t in range(0,3):
            ds = sims[t]
            for v in range(0, nvars):
                try:
                    raw_ad = ds.all_data()[('gas',tmp_varnames[v])]
                except:
                    raw_ad = ds.all_data()[('ramses',tmp_varnames[v])]
                for d in range(0,ndensity):
                    if tmp_varnames[v] == 'temperature':
                        data[d,v,t] = raw_ad[d].to('K')
                        if t==2 and v==0:
                            diff = abs(data[d,v,1] - data[d,v,2]) / data[d,v,2]
                            if diff >= conv_crit:
                                print(f'Temperature has not converged yet (err={diff},nH={data[d,0,t]})!')
                    else:
                        data[d,v,t] = raw_ad[d]
        
        # 4. Sort data
        sort_dens = np.argsort(density)
        if dust and 'no_dust' not in sim:
            data = data[sort_dens,:,:]
            density = density[sort_dens]

            ax1.plot(density,data[:,0,2],linestyle='-',linewidth=3,label=clean_name[sim.split('/')[-1]],color=colors[s])
            
            axins.plot(density,data[:,0,2], linestyle='-',linewidth=3,color=colors[s])
            axins2.plot(density,data[:,0,2], linestyle='-',linewidth=3,color=colors[s])

            ax2.plot(density,data[:,2,2]/data[:,2,0],linestyle='-',linewidth=2,color=colors[s])
            ax2.plot(density,data[:,3,2]/data[:,3,0],linestyle=':',linewidth=2,color=colors[s])
            
            ax3.plot(density,data[:,4,2]/data[:,4,0],linestyle='-',linewidth=2,color=colors[s])
            ax3.plot(density,data[:,5,2]/data[:,5,0],linestyle=':',linewidth=2,color=colors[s])
            
            ax4.plot(density,data[:,6,2]/data[:,6,0],linestyle='-',linewidth=2,color=colors[s])
            ax4.plot(density,data[:,7,2]/data[:,7,0],linestyle=':',linewidth=2,color=colors[s])
            
            ax5.plot(density,data[:,8,2]/data[:,8,0],linestyle='-',linewidth=2,color=colors[s])
            ax5.plot(density,data[:,9,2]/data[:,9,0],linestyle=':',linewidth=2,color=colors[s])
        
        elif 'no_dust' in sim:
            data = data[sort_dens,:,:]
            density = density[sort_dens]
            ax1.plot(density,data[:,0,2],linestyle='-',linewidth=3,label=clean_name[sim.split('/')[-1]],color=colors[s])
            
            axins.plot(density,data[:,0,2], linestyle='-',linewidth=3,color=colors[s])
            axins2.plot(density,data[:,0,2], linestyle='-',linewidth=3,color=colors[s])
            
            ax2.plot(density,data[:,2,2]/data[:,2,0],linestyle='-',linewidth=2,color=colors[s])
            ax2.plot(density,data[:,3,2]/data[:,3,0],linestyle=':',linewidth=2,color=colors[s])
            
            
        else:
            data = data[sort_dens,:]
            density = density[sort_dens]

            ax1.plot(np.log10(density),np.log10(data[:,1]),marker='o',
                    markersize=2,markerfacecolor='None',linestyle='none',label=sim.split('/')[-1])
        os.chdir(cwd)
        
    # Add a common y-axis label on the right side of the smaller subplots
    fig.text(0.97, 0.5, r'$\rho_X/\rho_{X,0}$', va='center', rotation=270, fontsize=16)
    for ax in [ax2, ax3, ax4, ax5]:
        ax.tick_params(which='both',axis="both",direction="in")
        ax.yaxis.tick_right()  # Move y-ticks to the right
        ax.yaxis.set_label_position('right')  # Move y-axis label to the right
        ax.tick_params(labelsize=16)
        ax.minorticks_on()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax1.legend(loc='best',fontsize=10,frameon=False,ncol=1)
    dummy_lines = [ax2.plot([],[],color='k',linestyle='-',label=r'H$_2$')[0],
                   ax2.plot([],[],color='k',linestyle=':',label=r'CO')[0]]
    new_leg = ax2.legend(handles=dummy_lines, loc='best', frameon=False, fontsize=12,ncol=2)
    ax2.add_artist(new_leg)
    dummy_lines = [ax3.plot([],[],color='k',linestyle='-',label=r'smallPAHs')[0],
                   ax3.plot([],[],color='k',linestyle=':',label=r'largePAHs')[0]]
    new_leg = ax3.legend(handles=dummy_lines, loc='best', frameon=False, fontsize=12,ncol=2)
    ax3.add_artist(new_leg)
    dummy_lines = [ax4.plot([],[],color='k',linestyle='-',label=r'smallC')[0],
                   ax4.plot([],[],color='k',linestyle=':',label=r'largeC')[0]]
    new_leg = ax4.legend(handles=dummy_lines, loc='best', frameon=False, fontsize=12,ncol=2)
    ax4.add_artist(new_leg)
    dummy_lines = [ax5.plot([],[],color='k',linestyle='-',label=r'smallSil')[0],
                   ax5.plot([],[],color='k',linestyle=':',label=r'largeSil')[0]]
    new_leg = ax5.legend(handles=dummy_lines, loc='best', frameon=False, fontsize=12,ncol=2)
    ax5.add_artist(new_leg)

    ax1.indicate_inset_zoom(axins, edgecolor="black")
    ax1.indicate_inset_zoom(axins2, edgecolor="black")
    
    # Create a shared legend for all subplots
    handles, labels = [], []
    for handle, label in zip(*ax1.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

    # Add a legend at the top of the figure, extending from side to side
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.01),
               bbox_transform=fig.transFigure, frameon=False, fontsize=9)
    
    fig.subplots_adjust(top=0.80,bottom=0.1,left=0.07,right=0.93,hspace=0,wspace=0)
        
    if not nolist:
        print('Making comparison plot in equil_comparison_thesis.pdf')
        fig.savefig('./plots/equil_comparison_thesis.pdf', format='pdf', dpi=300)
    else:
        print(f'equil_thesis_{simname[0].split("/")[-1]}.png')
        fig.savefig(f'./plots/equil_thesis_{simname[0].split("/")[-1]}.png', format='png', dpi=300)
            
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
        fields = basic_hydro + metal_massfrac + co_massfrac + dust_densities + metal_ion + ions + noadvect
        setup_yt(True,True)
    elif args.dust:
        fields = basic_hydro + metal_massfrac + co_massfrac + dust_densities[1:] + metal_ion + ions + noadvect
        setup_yt(True,False)
    else:
        fields = basic_hydro + metal_massfrac + co_massfrac + metal_ion + ions + ['unknown1','unknown2','unknown3','unknown4','unknown5','unknown6'] + noadvect
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
    elif args.type == 'thesis_plot':
        plot_for_thesis(fields,args.dust,args.simname)