"""
IBANEZ-MEJIAS+2019 CHARGING MODEL

"""

# LIBRARIES
import numpy as np


# FUNCTIONS
def grain_charge_dist(Gtot,T,ne,grain_type,grain_radius,gamma=None):
    from scipy.stats import norm
    # This uses the fitting function from Ibanez-Mejias et al. (2019)
    # (https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.1220I/abstract)
    # which are detailed in the Eq. 17-19.
    # We assume a discrete Gaussian distribution correcting for
    # integer dust charges
    
    # Fitting parameters form their Table 1
    fit_params = {'silicate':{'3.5A':{'alpha':0.3263,'k':0.0149,'b':-0.1212,'hz':57,
                                        'c+':0.4123,'eta+':0.2513,'d':0.1891,'c-':0.4845,
                                        'eta-':0.3532},
                                '5A':{'alpha':0.3141,'k':0.0372,'b':-0.3043,'hz':86,
                                        'c+':0.2734,'eta+':0.2925,'d':0.3233,'c-':0.3615,
                                        'eta-':0.6532},
                                '10A':{'alpha':0.3535,'k':0.0494,'b':-0.4865,'hz':73,
                                        'c+':0.4353,'eta+':0.7459,'d':0.4451,'c-':0.1053,
                                        'eta-':0.5803},
                                '50A':{'alpha':0.5115,'k':0.0717,'b':-0.4106,'hz':107,
                                        'c+':1.0758,'eta+':1.7832,'d':0.5860,'c-':-1.0379e3,
                                        'eta-':7.7069e3},
                                '100A':{'alpha':0.3525,'k':0.6591,'b':-0.1649,'hz':384,
                                        'c+':1.6245,'eta+':2.8390,'d':0.6346,'c-':-4.2075e2,
                                        'eta-':1.9840e3},
                                '500A':{'alpha':0.3643,'k':2.6283,'b':0.5217,'hz':345,
                                        'c+':4.0732,'eta+':11.0200,'d':0.6797,'c-':-0.2418,
                                        'eta-':0.5910},
                                '1000A':{'alpha':0.3927,'k':3.6493,'b':0.8389,'hz':372,
                                        'c+':5.9813,'eta+':20.6410,'d':0.6961,'c-':-0.1885,
                                        'eta-':0.4237}},

                'graphite':{'3.5A':{'alpha':0.4699,'k':0.0085,'b':-0.1162,'hz':48,
                                        'c+':0.3103,'eta+':0.2744,'d':0.2551,'c-':0.3766,
                                        'eta-':0.5241},
                                '5A':{'alpha':0.4386,'k':0.0195,'b':-0.3084,'hz':95,
                                        'c+':0.3699,'eta+':0.5654,'d':0.4158,'c-':0.2890,
                                        'eta-':1.6241},
                                '10A':{'alpha':0.4994,'k':0.0199,'b':-0.4959,'hz':78,
                                        'c+':0.6511,'eta+':0.9839,'d':0.5275,'c-':0.0213,
                                        'eta-':0.0977},
                                '50A':{'alpha':0.6009,'k':0.0523,'b':-0.4092,'hz':218,
                                        'c+':1.6536,'eta+':2.6688,'d':0.6671,'c-':-9.5138,
                                        'eta-':35.3519},
                                '100A':{'alpha':0.2900,'k':2.2310,'b':-0.2061,'hz':1063,
                                        'c+':2.5445,'eta+':4.3352,'d':0.7010,'c-':-2.5341e3,
                                        'eta-':8.1962e3},
                                '500A':{'alpha':0.3400,'k':5.8944,'b':0.1727,'hz':1034,
                                        'c+':5.9455,'eta+':18.3186,'d':0.8377,'c-':-2.4189e3,
                                        'eta-':4.9424e3},
                                '1000A':{'alpha':0.3500,'k':9.6536,'b':0.4183,'hz':1273,
                                        'c+':8.7003,'eta+':36.1014,'d':0.9094,'c-':-2.6009e3,
                                        'eta-':4.7029e3}}}

    sfit = fit_params[grain_type][grain_radius]
    if gamma != None:
        charPar = gamma
    else:
        charPar = Gtot *np.sqrt(T) / ne
    Z = sfit['k'] * (1.0 - np.exp(-charPar/sfit['hz'])) * (charPar**sfit['alpha']) + sfit['b']
    if Z>0:
        sigma = sfit['c+'] * (1.0 - np.exp(-Z/sfit['eta+'])) + sfit['d']
    else:
        sigma = sfit['c-'] * (1.0 - np.exp(-abs(Z)/sfit['eta-'])) + sfit['d']
        
    Zmin = round(Z - 3.*sigma)
    Zmax = round(Z + 3.*sigma)
    x = np.arange(Zmin,Zmax+1)
    dist = np.zeros(len(x))
    for i in range(0,len(x)):
        dist[i] = (1. / (sigma * np.sqrt(2.*np.pi))) * np.exp(-0.5*((float(x[i]) - Z) / sigma)**2.)
    dist = dist / np.sum(dist)
    return dist,x

def grain_mean_charge(Gtot,T,ne,grain_type,grain_radius,gamma=None):
    from scipy.stats import norm
    # This uses the fitting function from Ibanez-Mejias et al. (2019)
    # (https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.1220I/abstract)
    # which are detailed in the Eq. 17-19.
    
    # Fitting parameters form their Table 1
    fit_params = {'silicate':{'3.5A':{'alpha':0.3263,'k':0.0149,'b':-0.1212,'hz':57,
                                        'c+':0.4123,'eta+':0.2513,'d':0.1891,'c-':0.4845,
                                        'eta-':0.3532},
                                '5A':{'alpha':0.3141,'k':0.0372,'b':-0.3043,'hz':86,
                                        'c+':0.2734,'eta+':0.2925,'d':0.3233,'c-':0.3615,
                                        'eta-':0.6532},
                                '10A':{'alpha':0.3535,'k':0.0494,'b':-0.4865,'hz':73,
                                        'c+':0.4353,'eta+':0.7459,'d':0.4451,'c-':0.1053,
                                        'eta-':0.5803},
                                '50A':{'alpha':0.5115,'k':0.0717,'b':-0.4106,'hz':107,
                                        'c+':1.0758,'eta+':1.7832,'d':0.5860,'c-':-1.0379e3,
                                        'eta-':7.7069e3},
                                '100A':{'alpha':0.3525,'k':0.6591,'b':-0.1649,'hz':384,
                                        'c+':1.6245,'eta+':2.8390,'d':0.6346,'c-':-4.2075e2,
                                        'eta-':1.9840e3},
                                '500A':{'alpha':0.3643,'k':2.6283,'b':0.5217,'hz':345,
                                        'c+':4.0732,'eta+':11.0200,'d':0.6797,'c-':-0.2418,
                                        'eta-':0.5910},
                                '1000A':{'alpha':0.3927,'k':3.6493,'b':0.8389,'hz':372,
                                        'c+':5.9813,'eta+':20.6410,'d':0.6961,'c-':-0.1885,
                                        'eta-':0.4237}},

                'graphite':{'3.5A':{'alpha':0.4699,'k':0.0085,'b':-0.1162,'hz':48,
                                        'c+':0.3103,'eta+':0.2744,'d':0.2551,'c-':0.3766,
                                        'eta-':0.5241},
                                '5A':{'alpha':0.4386,'k':0.0195,'b':-0.3084,'hz':95,
                                        'c+':0.3699,'eta+':0.5654,'d':0.4158,'c-':0.2890,
                                        'eta-':1.6241},
                                '10A':{'alpha':0.4994,'k':0.0199,'b':-0.4959,'hz':78,
                                        'c+':0.6511,'eta+':0.9839,'d':0.5275,'c-':0.0213,
                                        'eta-':0.0977},
                                '50A':{'alpha':0.6009,'k':0.0523,'b':-0.4092,'hz':218,
                                        'c+':1.6536,'eta+':2.6688,'d':0.6671,'c-':-9.5138,
                                        'eta-':35.3519},
                                '100A':{'alpha':0.2900,'k':2.2310,'b':-0.2061,'hz':1063,
                                        'c+':2.5445,'eta+':4.3352,'d':0.7010,'c-':-2.5341e3,
                                        'eta-':8.1962e3},
                                '500A':{'alpha':0.3400,'k':5.8944,'b':0.1727,'hz':1034,
                                        'c+':5.9455,'eta+':18.3186,'d':0.8377,'c-':-2.4189e3,
                                        'eta-':4.9424e3},
                                '1000A':{'alpha':0.3500,'k':9.6536,'b':0.4183,'hz':1273,
                                        'c+':8.7003,'eta+':36.1014,'d':0.9094,'c-':-2.6009e3,
                                        'eta-':4.7029e3}}}

    sfit = fit_params[grain_type][grain_radius]
    if gamma != None:
        charPar = gamma
    else:
        charPar = Gtot *np.sqrt(T) / ne
    Z = sfit['k'] * (1.0 - np.exp(-charPar/sfit['hz'])) * (charPar**sfit['alpha']) + sfit['b']
    return Z

def grain_charge_sigma(Gtot,T,ne,grain_type,grain_radius,gamma=None):
    from scipy.stats import norm
    # This uses the fitting function from Ibanez-Mejias et al. (2019)
    # (https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.1220I/abstract)
    # which are detailed in the Eq. 17-19.
    
    # Fitting parameters form their Table 1
    fit_params = {'silicate':{'3.5A':{'alpha':0.3263,'k':0.0149,'b':-0.1212,'hz':57,
                                        'c+':0.4123,'eta+':0.2513,'d':0.1891,'c-':0.4845,
                                        'eta-':0.3532},
                                '5A':{'alpha':0.3141,'k':0.0372,'b':-0.3043,'hz':86,
                                        'c+':0.2734,'eta+':0.2925,'d':0.3233,'c-':0.3615,
                                        'eta-':0.6532},
                                '10A':{'alpha':0.3535,'k':0.0494,'b':-0.4865,'hz':73,
                                        'c+':0.4353,'eta+':0.7459,'d':0.4451,'c-':0.1053,
                                        'eta-':0.5803},
                                '50A':{'alpha':0.5115,'k':0.0717,'b':-0.4106,'hz':107,
                                        'c+':1.0758,'eta+':1.7832,'d':0.5860,'c-':-1.0379e3,
                                        'eta-':7.7069e3},
                                '100A':{'alpha':0.3525,'k':0.6591,'b':-0.1649,'hz':384,
                                        'c+':1.6245,'eta+':2.8390,'d':0.6346,'c-':-4.2075e2,
                                        'eta-':1.9840e3},
                                '500A':{'alpha':0.3643,'k':2.6283,'b':0.5217,'hz':345,
                                        'c+':4.0732,'eta+':11.0200,'d':0.6797,'c-':-0.2418,
                                        'eta-':0.5910},
                                '1000A':{'alpha':0.3927,'k':3.6493,'b':0.8389,'hz':372,
                                        'c+':5.9813,'eta+':20.6410,'d':0.6961,'c-':-0.1885,
                                        'eta-':0.4237}},

                'graphite':{'3.5A':{'alpha':0.4699,'k':0.0085,'b':-0.1162,'hz':48,
                                        'c+':0.3103,'eta+':0.2744,'d':0.2551,'c-':0.3766,
                                        'eta-':0.5241},
                                '5A':{'alpha':0.4386,'k':0.0195,'b':-0.3084,'hz':95,
                                        'c+':0.3699,'eta+':0.5654,'d':0.4158,'c-':0.2890,
                                        'eta-':1.6241},
                                '10A':{'alpha':0.4994,'k':0.0199,'b':-0.4959,'hz':78,
                                        'c+':0.6511,'eta+':0.9839,'d':0.5275,'c-':0.0213,
                                        'eta-':0.0977},
                                '50A':{'alpha':0.6009,'k':0.0523,'b':-0.4092,'hz':218,
                                        'c+':1.6536,'eta+':2.6688,'d':0.6671,'c-':-9.5138,
                                        'eta-':35.3519},
                                '100A':{'alpha':0.2900,'k':2.2310,'b':-0.2061,'hz':1063,
                                        'c+':2.5445,'eta+':4.3352,'d':0.7010,'c-':-2.5341e3,
                                        'eta-':8.1962e3},
                                '500A':{'alpha':0.3400,'k':5.8944,'b':0.1727,'hz':1034,
                                        'c+':5.9455,'eta+':18.3186,'d':0.8377,'c-':-2.4189e3,
                                        'eta-':4.9424e3},
                                '1000A':{'alpha':0.3500,'k':9.6536,'b':0.4183,'hz':1273,
                                        'c+':8.7003,'eta+':36.1014,'d':0.9094,'c-':-2.6009e3,
                                        'eta-':4.7029e3}}}

    sfit = fit_params[grain_type][grain_radius]
    if gamma != None:
        charPar = gamma
    else:
        charPar = Gtot *np.sqrt(T) / ne
    Z = sfit['k'] * (1.0 - np.exp(-charPar/sfit['hz'])) * (charPar**sfit['alpha']) + sfit['b']
    if Z>0:
        sigma = sfit['c+'] * (1.0 - np.exp(-Z/sfit['eta+'])) + sfit['d']
    else:
        sigma = sfit['c-'] * (1.0 - np.exp(-abs(Z)/sfit['eta-'])) + sfit['d']
    return sigma

def plot_grain_charge_dist(Gtot,T,ne,grain_type,grain_radius,gamma=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,4), dpi=300, facecolor='w', edgecolor='k')
    
    dist,Z = grain_charge_dist(Gtot,T,ne,grain_type,grain_radius,gamma)
    meanZ = grain_mean_charge(Gtot,T,ne,grain_type,grain_radius,gamma)
    print('Mean charge: '+str(meanZ))
    
    ax.step(Z,dist,color='b',alpha=0.5,label='Charge distribution',where='mid')
    ax.axvline(meanZ,color='r',linestyle='--',label='Mean charge: '+str(round(meanZ,2)))

    draine_Z = np.array([0,1,2,3])
    draine_P = np.array([0.,0.2,0.75,0.08])
    ax.step(draine_Z,draine_P,alpha=0.5,color='g',linestyle=':',label='Weingartner & Draine (2001)',where='mid')
    
    ax.set_ylabel(r'Probability', fontsize=13)
    ax.set_xlabel(r'Grain charge $Z$',fontsize=16)
    ax.set_ylim([0,1])
    ax.set_xlim([min(Z)-1,max(Z)+1])
    ax.tick_params(labelsize=12)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.legend(loc='best',fontsize=10,frameon=False)
    
    fig.subplots_adjust(top=0.97,bottom=0.13,left=0.15,right=0.99)
    fig.savefig('dust_grain_charge_dist_'+grain_type+'_'+grain_radius+'.png',format='png',dpi=300)
    plt.close(fig)

def grain_charge_probability(Gtot,T,ne,grain_type,grain_radius,Zi,gamma=None):
    """
    Compute the probability of a grain having a specific charge Zi.
    
    This function uses the same fitting function from Ibanez-Mejias et al. (2019)
    as grain_charge_dist, but returns only the probability for a specific charge.
    
    Parameters
    ----------
    Gtot : float
        Total radiation field intensity.
    T : float
        Temperature in K.
    ne : float
        Electron density in cm^-3.
    grain_type : str
        Type of grain ('silicate' or 'graphite').
    grain_radius : str
        Grain radius designation (e.g., '50A', '100A', etc.).
    Zi : int
        Specific charge for which to compute the probability.
    gamma : float, optional
        Charging parameter. If None, computed as Gtot * sqrt(T) / ne.
        
    Returns
    -------
    float
        Probability of the grain having charge Zi.
    """
    # Fitting parameters from Ibanez-Mejias et al. (2019) Table 1
    fit_params = {'silicate':{'3.5A':{'alpha':0.3263,'k':0.0149,'b':-0.1212,'hz':57,
                                        'c+':0.4123,'eta+':0.2513,'d':0.1891,'c-':0.4845,
                                        'eta-':0.3532},
                                '5A':{'alpha':0.3141,'k':0.0372,'b':-0.3043,'hz':86,
                                        'c+':0.2734,'eta+':0.2925,'d':0.3233,'c-':0.3615,
                                        'eta-':0.6532},
                                '10A':{'alpha':0.3535,'k':0.0494,'b':-0.4865,'hz':73,
                                        'c+':0.4353,'eta+':0.7459,'d':0.4451,'c-':0.1053,
                                        'eta-':0.5803},
                                '50A':{'alpha':0.5115,'k':0.0717,'b':-0.4106,'hz':107,
                                        'c+':1.0758,'eta+':1.7832,'d':0.5860,'c-':-1.0379e3,
                                        'eta-':7.7069e3},
                                '100A':{'alpha':0.3525,'k':0.6591,'b':-0.1649,'hz':384,
                                        'c+':1.6245,'eta+':2.8390,'d':0.6346,'c-':-4.2075e2,
                                        'eta-':1.9840e3},
                                '500A':{'alpha':0.3643,'k':2.6283,'b':0.5217,'hz':345,
                                        'c+':4.0732,'eta+':11.0200,'d':0.6797,'c-':-0.2418,
                                        'eta-':0.5910},
                                '1000A':{'alpha':0.3927,'k':3.6493,'b':0.8389,'hz':372,
                                        'c+':5.9813,'eta+':20.6410,'d':0.6961,'c-':-0.1885,
                                        'eta-':0.4237}},

                'graphite':{'3.5A':{'alpha':0.4699,'k':0.0085,'b':-0.1162,'hz':48,
                                        'c+':0.3103,'eta+':0.2744,'d':0.2551,'c-':0.3766,
                                        'eta-':0.5241},
                                '5A':{'alpha':0.4386,'k':0.0195,'b':-0.3084,'hz':95,
                                        'c+':0.3699,'eta+':0.5654,'d':0.4158,'c-':0.2890,
                                        'eta-':1.6241},
                                '10A':{'alpha':0.4994,'k':0.0199,'b':-0.4959,'hz':78,
                                        'c+':0.6511,'eta+':0.9839,'d':0.5275,'c-':0.0213,
                                        'eta-':0.0977},
                                '50A':{'alpha':0.6009,'k':0.0523,'b':-0.4092,'hz':218,
                                        'c+':1.6536,'eta+':2.6688,'d':0.6671,'c-':-9.5138,
                                        'eta-':35.3519},
                                '100A':{'alpha':0.2900,'k':2.2310,'b':-0.2061,'hz':1063,
                                        'c+':2.5445,'eta+':4.3352,'d':0.7010,'c-':-2.5341e3,
                                        'eta-':8.1962e3},
                                '500A':{'alpha':0.3400,'k':5.8944,'b':0.1727,'hz':1034,
                                        'c+':5.9455,'eta+':18.3186,'d':0.8377,'c-':-2.4189e3,
                                        'eta-':4.9424e3},
                                '1000A':{'alpha':0.3500,'k':9.6536,'b':0.4183,'hz':1273,
                                        'c+':8.7003,'eta+':36.1014,'d':0.9094,'c-':-2.6009e3,
                                        'eta-':4.7029e3}}}

    sfit = fit_params[grain_type][grain_radius]
    if gamma != None:
        charPar = gamma
    else:
        charPar = Gtot *np.sqrt(T) / ne
    Z = sfit['k'] * (1.0 - np.exp(-charPar/sfit['hz'])) * (charPar**sfit['alpha']) + sfit['b']
    if Z>0:
        sigma = sfit['c+'] * (1.0 - np.exp(-Z/sfit['eta+'])) + sfit['d']
    else:
        sigma = sfit['c-'] * (1.0 - np.exp(-abs(Z)/sfit['eta-'])) + sfit['d']
    
    # Compute the probability for the specific charge Zi
    prob_Zi = (1. / (sigma * np.sqrt(2.*np.pi))) * np.exp(-0.5*((float(Zi) - Z) / sigma)**2.)
    
    # To get the normalized probability, we need to compute the normalization factor
    # by integrating over the range where the distribution is significant
    Zmin = round(Z - 3.*sigma)
    Zmax = round(Z + 3.*sigma)
    x = np.arange(Zmin, Zmax+1)
    norm_factor = 0.0
    for i in range(0, len(x)):
        norm_factor += (1. / (sigma * np.sqrt(2.*np.pi))) * np.exp(-0.5*((float(x[i]) - Z) / sigma)**2.)
    
    # Return normalized probability
    return prob_Zi / norm_factor

