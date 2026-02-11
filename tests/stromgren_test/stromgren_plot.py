"""
ANALYSIS OF STROMGREN SPHERE TESTS FOR Dusty-PRISM

The scripts included in this Python file are used for the reading of outputs for the
Strömgren sphere tests with the Dusty-PRISM version of RAMSES-RTZ. This allows for a study
of the influence of dust in an ionised gas environment.

By: F. Rodriguez Montero (currodri@gmail.com)

"""

# Import required libraries
import itertools
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
from yt import YTArray, YTQuantity
yt.set_log_level("critical")
yt.enable_parallelism()
from unyt import mh,g

# Metals and dust parameters
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
                  'MgI','MgII','MgIII','MgIV','MgV','MgVI',
                  'SiI','SiII','SiIII','SiIV','SiV','SiVI',
                  'SI','SII','SIII','SIV','SV','SVI',
                  'FeI','FeII','FeIII','FeIV','FeV','FeVI',
                  'NeI','NeII','NeIII','NeIV','NeV','NeVI']
ions              = ['HI','HII','HeII','HeIII']
dust_densities    = ['pahs','CSmall','CLarge','SilSmall','SilLarge']
noadvect          = ['cooling_time','temperature','cooling_rate','heating_rate',
                  'cooling_primordial','cooling_fine_structure','cooling_CII',
                  'cooling_OI','cooling_CO','cooling_dust','cooling_dust_rec',
                  'heating_cr','heating_pe','heating_h2','heating_ct']
noadvect_dust     = ['dust_temperature']

def flatten_list(nested_list):
    return list(itertools.chain(*nested_list))

def setup_yt(dust):
    
    # Define my own spherical coordinates
    @yt.derived_field(name='radius', sampling_type="cell", units='cm',force_override=True)
    def _radius(field,data):
        length = float(data.ds.domain_width.in_units('cm')[0]/2)
        x = data['x'].in_units('cm') - YTArray([length],'cm')
        y = data['y'].in_units('cm') - YTArray([length],'cm')
        z = data['z'].in_units('cm') - YTArray([length],'cm')
        return np.sqrt(x**2 + y**2 + z**2)
    
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
    
    @yt.derived_field(name='nHII', sampling_type="cell", units='cm**-3',force_override=True)
    def _nHII(field,data):
        return data[('gas','nH')] * data[('ramses','HII')]
    
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
    
    @yt.derived_field(name='nOI', sampling_type="cell", units='cm**-3',force_override=True)
    def _nOI(field,data):
        return (data[('gas','density')] * data[('ramses','OMassFrac')] * data[('ramses','OI')]) / (mO_NIST_amu*amu_to_g*g)
    
    @yt.derived_field(name='nOII', sampling_type="cell", units='cm**-3',force_override=True)
    def _nOII(field,data):
        return (data[('gas','density')] * data[('ramses','OMassFrac')] * data[('ramses','OII')]) / (mO_NIST_amu*amu_to_g*g)
    
    @yt.derived_field(name='nOIII', sampling_type="cell", units='cm**-3',force_override=True)
    def _nOIII(field,data):
        return (data[('gas','density')] * data[('ramses','OMassFrac')] * data[('ramses','OIII')]) / (mO_NIST_amu*amu_to_g*g)
    
    @yt.derived_field(name='nNI', sampling_type="cell", units='cm**-3',force_override=True)
    def _nNI(field,data):
        return (data[('gas','density')] * data[('ramses','NMassFrac')] * data[('ramses','NI')]) / (mN_NIST_amu*amu_to_g*g)
    
    @yt.derived_field(name='nNII', sampling_type="cell", units='cm**-3',force_override=True)
    def _nNII(field,data):
        return (data[('gas','density')] * data[('ramses','NMassFrac')] * data[('ramses','NII')]) / (mN_NIST_amu*amu_to_g*g)
    
    if dust:
        @yt.derived_field(name='nPAH', sampling_type="cell", units='cm**-3',force_override=True)
        def _nPAH(field,data):
            return (data[('gas','density')] * data[('ramses','pahs')] ) / (mPAH)

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
        
def plot_slices_profiles(my_fields,simname,dust):
    
    
    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    
    # 1. Get the outputs in the directory
    cwd = os.getcwd()
    os.chdir(f'./{simname}')
    outputs = sorted(list(filter(lambda file: file.startswith('output'), os.listdir())),key=lambda x: int(x.split('_')[-1]))
    output_dir = os.getcwd()
    
    # 2. Load selected 4 outputs
    myouts = ['output_00002','output_00003','output_00005','output_00007']
    sims = [yt.load(f'{output_dir}/output_{str(out.split("_")[-1])}',fields=my_fields) for out in myouts]
    
    # 3. Get slices and radial profiles for each output
    if dust:
        varnames = [['nHI','nHII'],['temperature','dust_temperature'],
                    ['nCI','nCII','nCO'],
                    ['nPAH','nCSmall','nCLarge','nSilSmall','nSilLarge']]
        varnames = [[('gas','nHI'),('gas','nHII')],[('gas','temperature'),('ramses','dust_temperature')],
                    [('gas','nOI'),('gas','nOII'),('gas','nOIII')],
                    [('gas','nPAH'),('gas','nCSmall'),('gas','nCLarge'),('gas','nSilSmall'),('gas','nSilLarge')]]
        labels  = [r'$n_{\rm Hx}$ [cm$^{-3}$]',
                   r'$T_x$ [K]',r'$n_{\rm Cx}$ [cm$^{-3}$]',
                   r'$n_{\rm Dust}$ [cm$^{-3}$']
        colors = [['b','r'],['k','m'],['green','orange','r'],['m','y','b','r','orange']]
    else:
        varnames = [[('gas','nHI'),('gas','nHII')],[('gas','temperature')],
                    [('gas','nOI'),('gas','nOII'),('gas','nOIII')],
                    [('gas','nNI'),('gas','nNII')]]
        labels  = [r'$n_{\rm Hx}$ [cm$^{-3}$]',
                   r'$T$ [K]',r'$n_{\rm Ox}$ [cm$^{-3}$]',
                   r'$n_{\rm Nx}$ [cm$^{-3}$]']
        colors = [['b','r'],['k'],['green','orange','r'],['m','y']]
    slices = []
    profiles = []
    
    for s in range(0, len(sims)):
        sim = sims[s]
        proj = sim.proj(('gas','temperature'),'z',weight_field=("gas", "volume"))
        width = (10, "pc")  # we want a 10 pc view
        res = [1024, 1024]  # create an image with 1000x1000 pixels
        frb = proj.to_frb(width, res, center=[0.5,0.5,0.5])
        # Slices
        slices.append(np.array(frb[('gas','temperature')].to('K')))
        
        # Loop over profile values
        varlist = flatten_list(varnames)
        sp = sim.sphere(sim.domain_center, (5, "pc"))
        profile = yt.create_profile(
                                    sp,
                                    ('gas','radius'),
                                    varlist,
                                    n_bins=256,
                                    weight_field=("gas", "volume"),
                                    logs={('gas','radius'): False}
                                )
        profiles.append(profile)
    
    # 4. Prepare figure with all it's details
    ncol = 4
    nrow = 2
    figsize = plt.figaspect(float((5.0 * ncol) / (20.0 * nrow)))
    fig = plt.figure(figsize=2*figsize, facecolor='w', edgecolor='k')
    plot_grid = fig.add_gridspec(nrow, ncol, wspace=0, hspace=0)
    axes = []
    for i in range(0,nrow):
        axes.append([])
        for j in range(0,ncol):
            axes[i].append(fig.add_subplot(plot_grid[i,j]))
            # Setup axis
            ax = axes[i][-1]
            if j <= 1:
                if i == 0:
                    ax.xaxis.set_ticks_position('top')
                else:
                    ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                ax.minorticks_on()
                ax.tick_params(labelsize=14)
                ax.set_xlim([-5,5])
                ax.set_ylim([-5,5])
                if j == 0 and i == 1:
                    ax.set_ylabel(r'$y$ [pc]', fontsize=16)
                    ax.set_xlabel(r'$x$ [pc]', fontsize=16)
                    ax.xaxis.set_ticks_position('both')
                elif i == 0 and j == 0:
                    ax.set_ylabel(r'$y$ [pc]', fontsize=16)
                elif j == 0:
                    ax.set_xlabel(r'$x$ [pc]', fontsize=16)
                elif j == 1 and i == 1:
                    ax.set_xlabel(r'$x$ [pc]', fontsize=16)
                if j == 1 and i ==0:
                    ax.get_yaxis().set_ticks([])
                    ax.get_xaxis().set_ticks([])
                if j == 1 and i ==1:
                    ax.get_yaxis().set_ticks([])
                    ax.get_xaxis().set_ticks([])
                if j == 0 and i == 0:
                    ax.get_xaxis().set_ticks([])
            elif j > 1:
                ax.set_xlim([0,5])
                ax.set_yscale('log')
                if j == 2:
                    ax.tick_params(which="both",axis="y",direction="in",pad=-35)
                    
                    ax.yaxis.set_ticks_position('left')
                    ax.minorticks_on()
                    ax.tick_params(labelsize=14)
                    if i == 1:
                        ax.xaxis.set_ticks_position('bottom')
                        ax.set_xlabel(r'$r$ [pc]', fontsize=16)
                    else:
                        ax.xaxis.set_ticks_position('top')
                elif j == 3:
                    ax.tick_params(which="both",axis="y",direction="out")
                    ax.xaxis.set_ticks_position('top')
                    ax.yaxis.set_ticks_position('right')
                    ax.minorticks_on()
                    ax.tick_params(labelsize=14)
                    if i == 1:
                        ax.xaxis.set_ticks_position('bottom')
                        ax.set_xlabel(r'$r$ [pc]', fontsize=16)
                    else:
                        ax.xaxis.set_ticks_position('top')
                    
    axes = np.asarray(axes)                

    # 5. Add slices of temperature to the left half of axes
    for s in range(0,len(slices)):
        if s==0:
            ax = axes[0,0]
        elif s==1:
            ax = axes[0,1]
        elif s==2:
            ax = axes[1,0]
        elif s ==3:
            ax = axes[1,1]
        plot = ax.imshow(np.log10(slices[s]), cmap='inferno',
                                origin='lower',vmin=2,vmax=5,
                                extent=[-5,5,-5,5],
                                interpolation='none')
        ax.text(0.05, 0.90, r'$t = {z:.2f}$ kyr'.format(z=sims[s].current_time.to('kyr').d), # Time
                            verticalalignment='bottom', horizontalalignment='left',
                            transform=ax.transAxes,
                            color='white', fontsize=16,fontweight='bold')

        fontprops = fm.FontProperties(size=16,weight='bold')
        slb = AnchoredSizeBar(ax.transData,
                                1, '1 pc', 
                                'lower right', 
                                pad=0.1,
                                color='white',
                                frameon=False,
                                size_vertical=0.1,
                                fontproperties=fontprops)
        ax.add_artist(slb)
        if s == 0:
            cbaxes = cbaxes = inset_axes(
                                        ax,
                                        width="100%",  # width: 5% of parent_bbox width
                                        height="100%",  # height: 50%
                                        loc="upper left",
                                        bbox_to_anchor=(0., 1., 2, 0.05),
                                        bbox_transform=ax.transAxes,
                                        borderpad=0,
                                    )
            cbar = fig.colorbar(plot, cax=cbaxes, orientation='horizontal',location='top')
            cbar.set_label(r'$T$ [K]',color='k',fontsize=16)
            cbar.ax.tick_params(length=0,width=0,labelsize=14)
    
    
        
    # 6. Add profiles to the right half of boxes
    for s in range(0,len(slices)):
        for p in range(0,len(varnames)):
            if p == 0:
                ax = axes[0,2]
                ax.set_ylabel(labels[p],fontsize=16,labelpad=-55)
            elif p == 1:
                ax = axes[0,3]
                ax.set_ylabel(labels[p],fontsize=16)
                ax.yaxis.set_label_position("right")
            elif p == 2:
                ax = axes[1,2]
                ax.set_ylabel(labels[p],fontsize=16,labelpad=-55)
            elif p == 3:
                ax = axes[1,3]
                ax.set_ylabel(labels[p],fontsize=16)
                ax.yaxis.set_label_position("right")
            varlist = varnames[p]
            
            for v in range(0, len(varlist)):
                if s == len(slices)-1:
                    plot = ax.plot(profiles[s].x.to('pc'),profiles[s][varlist[v]],label=varlist[v][1],color=colors[p][v],linewidth=3)
                else:
                    plot = ax.plot(profiles[s].x.to('pc'),profiles[s][varlist[v]],alpha=min(s/(len(slices)-1),0.3),color=colors[p][v],linewidth=3)
            if s == len(slices)-1:
                ax.legend(loc='best',fontsize=12,frameon=False)
    
    # 7. Save figure
    fig.subplots_adjust(top=0.91,bottom=0.07,left=0.04,right=0.95)
    os.chdir(cwd)
    fig.savefig(f'slice_profiles_{simname}.png', format='png', dpi=300)
    
                
    
if __name__ == '__main__':

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description='Plotting evolution of Stromgren sphere tests of Dusty-PRISM')
    parser.add_argument('type', type=str, help='Type of plot to do')
    parser.add_argument('--dust',action='store_true',help='Use if simulation includes dust.')
    parser.add_argument('--simname', type=str, default='dust', help='Simulation name.')
    args = parser.parse_args()

    if args.dust:
        fields = basic_hydro + metal_massfrac + dust_densities + co_massfrac + metal_ion + ions + noadvect + noadvect_dust
        setup_yt(True)
    else:
        fields = basic_hydro + metal_massfrac + co_massfrac + metal_ion + ions + noadvect
        setup_yt(False)
    
    if args.type == 'slice_profiles':
        plot_slices_profiles(fields,args.simname,args.dust)
    