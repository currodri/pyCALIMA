"""
SETUP STROMGREN SPHERE TESTS WITH THE DUST

These scripts give guidelines and abundances of metals/dust
for the given metallicity, such that the initial depletion
is considered.

By: F. Rodriguez Montero (currodri@gmail.com)
"""

# Import libraries
import numpy as np
import argparse


# CLOUDY solar abundances
table_Solar_Fe = 0.0012917570
table_Solar_O  = 0.0057326442
table_Solar_N  = 0.00069290468
table_Solar_Si = 0.00066494756
table_Solar_Ca = 6.4145074e-05
table_Solar_Al = 5.5625916e-05
table_Solar_Mg = 0.00070797884
table_Solar_Eu = 3.6810875e-10
table_Solar_C  = 0.0023647147
table_Solar_Ne = 0.0012564824
table_Solar_S  = 0.00030926749
table_Solar_H  = 0.73738783
table_Solar_He = 0.24924204

# Dust depletion values at Zsun
# These values are used for starting isolated sims and tests
# following the fractional contributions of the BARE-GR-S model
# from Zubko et al. (2004) - see Table 6
# (https://ui.adsabs.harvard.edu/abs/2004ApJS..152..211Z/abstract)
GD_solar = 162.0
fCDust_inPAH=1.342e-1
fMagnesium_indust=8.37e-1
fOxygen_indust=2.72e-1
# and Dopita et al. (2000) - see Table 1
# (https://ui.adsabs.harvard.edu/abs/2000ApJ...539..742D/abstract)
fCarbon_indust=4.9881e-1
fIron_indust=9.9e-1
fSilicon_indust=9.0e-1

mH_amu = 1.007825       # Hydrogen molecular weight [amu]
mO_amu = 15.9994        # Oxygen molecular weight [amu]
mC_amu = 12.0107        # Carbon molecular weight [amu]
mMg_amu = 24.305        # Magnesium molecular weight [amu]
mSi_amu = 28.0855       # Silicon molecular weight [amu]
mFe_amu = 55.854        # Iron molecular weight [amu]

nsilMg = 1.
nsilFe = 1.
nsilSi = 1.
nsilO = 4.

numtot=mMg_amu*nsilMg+mFe_amu*nsilFe+mSi_amu*nsilSi+mO_amu*nsilO
MgoverSil=mMg_amu*nsilMg/numtot
FeoverSil=mFe_amu*nsilFe/numtot
SioverSil=mSi_amu*nsilSi/numtot
OoverSil =mO_amu *nsilO /numtot


def GD_RR14(OHratio):
    # This function returns an estimate of the gas to dust ration
    # (G/D) based on a broken power-law fit by Remy-Ruyer et al. (2014)
    # (see their parameters in Table 1:
    # https://ui.adsabs.harvard.edu/abs/2014A%26A...563A..31R/abstract)
    # This assumes that the solar abundance is (O/H)sun = 4.9e-4
    
    # This uses the XCO,Z case (right column of Table 1)
    a = 2.21; alpha_H = 1.0; b = 0.96
    alpha_L = 3.10; x_t = 8.10; xsun = 8.69
    
    x = 12.0 + np.log10(OHratio)
    
    if x > x_t:
        y = a + alpha_H * (xsun - x)
    else:
        y = a + alpha_L * (xsun - x)
        
    return 10**y

def init_dust_depletion(z_ave,GDinit,fpah_ini,fsmall_ini):
    # This routine is used for initialising isolated galaxy simulations
    # following the fractional contributions of the BARE-GR-S model
    # from Zubko et al. (2004) - see Table 6
    # (https://ui.adsabs.harvard.edu/abs/2004ApJS..152..211Z/abstract)
    # and Dopita et al. (2000) - see Table 1
    # (https://ui.adsabs.harvard.edu/abs/2000ApJ...539..742D/abstract)
    
    
    fIron = z_ave * table_Solar_Fe
    fOxygen = z_ave * table_Solar_O
    fNitrogen = z_ave * table_Solar_N
    fMagnesium = z_ave * table_Solar_Mg
    fNeon = z_ave * table_Solar_Ne
    fSilicon = z_ave * table_Solar_Si
    fCalcium = z_ave * table_Solar_Ca
    fCarbon = z_ave * table_Solar_C
    fSulfur = z_ave * table_Solar_S
    
    ndchemtype = 2
    dndsize = 1
    
    if GDinit==-1.:
        GD = GD_RR14(z_ave*8.69)
    else:
        GD = GDinit
        
    if GD != 0.0:
        GDfactor = GD_solar / GD # Scaled to solar G/D=162 (Zubko et al. 2004)
        for jj in range(1,ndchemtype+1):
            if jj==1:
                dustC   = fCarbon * fCarbon_indust * GDfactor
                if (fpah_ini==-1.0):
                    dustPAH = fCDust_inPAH * dustC
                else:
                    dustPAH = min(max(fpah_ini,0.0),1.) * dustC
                dustC   = (1. - fCDust_inPAH) * dustC
                dustCsmall = min(max(fsmall_ini,0.0),1.0) * dustC
                dustClarge = (1. - min(max(fsmall_ini,0.),1.)) * dustC
                # Deplete carbon
                fCarbon = max(fCarbon - (dustC + dustPAH),0.)
            elif (jj==ndchemtype):
                MdustSil = np.array([fMagnesium * fMagnesium_indust / MgoverSil \
                                    ,fIron * fIron_indust / FeoverSil \
                                    ,fSilicon * fSilicon_indust / SioverSil \
                                    ,fOxygen * fOxygen_indust / OoverSil])
                Mfrac = np.zeros(4)
                Mfrac[0] = fMagnesium * fMagnesium_indust / (mMg_amu*nsilMg)
                Mfrac[1] = fIron * fIron_indust / (mFe_amu*nsilFe)
                Mfrac[2] = fSilicon * fSilicon_indust / (mSi_amu*nsilSi)
                Mfrac[3] = fOxygen * fOxygen_indust / (mO_amu*nsilO)
                ikey = np.argmin(Mfrac)
                dustSil = GDfactor * MdustSil[ikey]
                dustSilsmall = min(max(fsmall_ini,0),1) * dustSil
                dustSillarge = (1 - min(max(fsmall_ini,0),1)) * dustSil
                # Deplete silicon
                fSilicon    = max(fSilicon - dustSil*SioverSil, 0)
                # Deplete iron
                fIron    = max(fIron - dustSil*FeoverSil, 0)
                # Deplete magnesium
                fMagnesium    = max(fMagnesium - dustSil*MgoverSil, 0)
                # Deplete oxygen
                fOxygen    = max(fOxygen - dustSil*OoverSil, 0)
    else:
        dustPAH = 0.
        dustCsmall = 0.
        dustClarge = 0.
        dustSilsmall = 0.
        dustSillarge = 0.
    
    realGD = 1./(dustPAH+dustCsmall+dustClarge+dustSilsmall+dustSillarge)
    print('====== DUST AND METAL FRACTIONS ======')
    print(f'This has been computed for {z_ave} Zsun')
    print('- METALS')
    print(f'Iron mass fraction:      {fIron}')
    print(f'Oxygen mass fraction:    {fOxygen}')
    print(f'Nitrogen mass fraction:  {fNitrogen}')
    print(f'Magnesium mass fraction: {fMagnesium}')
    print(f'Neon mass fraction:      {fNeon}')
    print(f'Silicon mass fraction:   {fSilicon}')
    print(f'Calcium mass fraction:   {fCalcium}')
    print(f'Carbon mass fraction:    {fCarbon}')
    print(f'Sulfur mass fraction:    {fSulfur}')
    print('- DUST')
    print(f'GD real:                 {realGD}')
    print(f'PAHs mass fraction:      {dustPAH}')
    print(f'Csmall mass fraction:    {dustCsmall}')
    print(f'Clarge mass fraction:    {dustClarge}')
    print(f'Silsmall mass fraction:  {dustSilsmall}')
    print(f'Sillarge mass fraction:  {dustSillarge}')
    
        
        
            
if __name__ == '__main__':

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description='Setting up Stromgren sphere for Dusty-PRISM')
    parser.add_argument('--z_ave', type=float, default=1., help='Metallicity in solar units.')
    parser.add_argument('--GDinit', type=float, default=162., help='Gas-to-dust ratio.')
    parser.add_argument('--fpah_ini', type=float, default=fCDust_inPAH, help='Initial fraction of C in PAHs.')
    parser.add_argument('--fsmall_ini', type=float, default=0.5, help='Fraction of dust mass in small grains')
    args = parser.parse_args()
    
    
    init_dust_depletion(args.z_ave,args.GDinit,args.fpah_ini,args.fsmall_ini)
    