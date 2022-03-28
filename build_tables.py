"""
STELLAR YIELDS FOR RAMSES

This simple script helps on the construction of stellar yields
for the RAMSES RTZ + Dust version.

These are the numbers calculated from YD Stellar Yields for the SNII+pre-SNII phase
Charbier IMF [0.01,100]Msun, failed SNII>30 Msun, low-high mass@8Msun
Limongi&Chieffi18 yields with Prantzos IDROV + Karakas10
Mg SNII yields are artificially doubled to match with observations (Mg~Si)
Dust is made of C and olivine MgFeSiO4 with Dwek98 efficiencies
dubois@iap.fr

By: Curro Rodriguez (currodri@gmail.com)
"""

# Importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
Zsun_old = 0.02
Zsun_asplund = 0.01345
yield_dir = '/home/dubois/StellarYields'
yield_files = os.listdir(yield_dir)
ELEMENTS = np.array(['p','He','C','N','O','F','Ne','Mg','Si','S','Fe'])

def get_agbdata():

    # AGB WIND YIELDS

    agbdata = {}

    # These files have a header with 14 lines that looks like this

    #---Details of Columns:
    #    M0 (solMass)          (F5.2)  [1/6.5] Initial mass [ucd=phys.mass]
    #    Z0                    (D10.3)  Initial metallicity [ucd=phys.abund.Z]
    #    M1 (solMass)          (F6.3)  Final mass (1) [ucd=phys.mass]
    #    El                    (a4)    Species i (2) [ucd=phys.atmol.element]
    #    Yield (solMass)       (D9.2)  Net yield [ucd=phys.composition.yield]
    #    M(i)lost (solMass)    (D9.2)  Mass of species i lost in the wind [ucd=phys.mass]
    #    M(i)0 (solMass)       (D9.2)  Mass of species i initially present in the wind [ucd=phys.mass]
    #    M(i)lostall (solMass) (D9.2)  Total mass lost in the wind [ucd=phys.mass]
    #----- --------- ----- ---- --------- --------- --------- -----------
    #M0
    #(sol            M1 (so     Yield     M(i)lost  M(i)0     M(i)lostall
    #Mass) Z0        lMass) El  (solMass) (solMass) (solMass) (solMass)
    #----- --------- ----- ---- --------- --------- --------- -----------

    column_names = ['M0','Z0','M1','El','Yield','M(i)lost','M(i)0','M(i)lostall']
    header_length = 14

    Zkarakas = np.array([0.0001,0.004,0.008,0.02])
    
    # Just load one file to check how many masses do we have for the grid
    karakas_filename = 'karakas_z%s_simplified.txt'%(str(Zkarakas[0]))
    filepath = os.path.join(yield_dir,karakas_filename)
    ka1 = pd.read_csv(filepath, header=13, delim_whitespace=True,names=column_names)
    M0 = np.unique(ka1["M0"].to_numpy())

    # We can now build the data dictionary and fill it up!

    nmet = len(Zkarakas)
    nmassAGB = len(M0)

    for i in range(0, len(ELEMENTS)):
        agbdata[ELEMENTS[i]] = np.zeros((nmet,nmassAGB))

    for i in range(0, len(Zkarakas)):
        karakas_filename = 'karakas_z%s_simplified.txt'%(str(Zkarakas[i]))
        filepath = os.path.join(yield_dir,karakas_filename)
        ka = pd.read_csv(filepath, header=header_length-1, delim_whitespace=True,names=column_names)
        zka = Zkarakas[i]
        for j in range(0, len(ELEMENTS)):
            myield = ka['M(i)lost'][ka['El'] == ELEMENTS[j]]
            agbdata[ELEMENTS[j]][i,:] = myield[0:nmassAGB]

    return agbdata, Zkarakas,M0

def get_snIIdata():
    # SNII YIELDS + PRE-SNII WIND YIELDS

    # As for detailed in the Dust in RAMSES paper, we follow the case in which 
    # stellar yields from Limongi&Chieffi18 are fixed for a particular
    # ZAS metallicity and ZAS rotational velocity of the star

    VZAS = np.array([150,100,50,50,50,50,50]) # in km/s
    ZZAS = np.array([1e-3,1e-2,1e-1,10**(-0.6),10**(-0.3),1,10**(0.3)]) # in Zsun (Asplund units)
    ZZASlabel = np.array(['-3','-2','-1','-0.6','-0.3','0','0.3'])

    snIIdata = {}

    # These files have a header with 13 lines that looks like this

    #---Details of Columns:
    #    M0 (solMass)          (F6.2)  [1/6.5] Initial mass [ucd=phys.mass]
    #    [Fe/H]                (F7.2)  Initial metallicity [ucd=phys.abund.Z]
    #    vel (km/s)            (F7.2)  Initial rotation [ucd=phys.mass]
    #    M1 (solMass)          (F7.2)  Final mass (1) [ucd=phys.mass]
    #    El                    (a4)    Species i (2) [ucd=phys.atmol.element]
    #    M(i)lost (solMass)    (D9.2)  Mass of species i lost in the wind [ucd=phys.mass]
    #    M(i)lostall (solMass) (D9.2)  Total mass lost in the wind [ucd=phys.mass]
    #----- ------ ------ ------ ---- --------- -----------
    #M0
    #(sol         vel    M1 (so      M(i)lost  M(i)lostall
    #Mass) Z0     (km/s) lMass) El   (solMass) (solMass)
    #----- ------ ------ ------ ---- --------- -----------

    column_names = ['M0','Z0','vel','M1','El','M(i)lost','M(i)lostall']
    header_length = 13

    # Just load one file to check how many masses do we have for the grid
    filename = "limongichieffi_z%s_vel%i_simplified.txt"%(ZZASlabel[0],VZAS[0])
    filepath = os.path.join(yield_dir,filename)
    lc18 = pd.read_csv(filepath, header=header_length-1, delim_whitespace=True,names=column_names)
    M0 = np.unique(lc18["M0"].to_numpy())
    # We can now build the data dictionary and fill it up!

    nmet = len(ZZAS)
    nmassSNII = len(M0)

    for i in range(0, len(ELEMENTS)):
        snIIdata[ELEMENTS[i]] = np.zeros((nmet,nmassSNII))

    for i in range(0, len(ZZAS)):
        filename = "limongichieffi_z%s_vel%i_simplified.txt"%(ZZASlabel[i],VZAS[i])
        print(filename)
        filepath = os.path.join(yield_dir,filename)
        ka = pd.read_csv(filepath, header=header_length-1, delim_whitespace=True,names=column_names)
        zka = ZZAS[i]
        for j in range(0, len(ELEMENTS)):
            myield = ka['M(i)lost'][ka['El'] == ELEMENTS[j]]
            snIIdata[ELEMENTS[j]][i,:] = myield[0:nmassSNII]
    
    return snIIdata,ZZAS,M0


# See equations 22-24 in Dwek (1998)
cond_eff_SNII = {'Mg':0.8,'Si':0.8,
                'O':0.8,'Fe':0.8,
                 'C':0.5}
cond_eff_SNIa = {'Mg':0.8,'Si':0.8,
                 'O':1.0,'Fe':0.8,
                 'C':0.5}
cond_eff_AGB = {'C/O>1':1.0,'C/O<1':0.8}
Sil_comp = ['Mg','O','Si','Fe']
mu = {'O':15.9994,'C':12.0107,'Mg':24.305,
      'Si':28.0855,'Fe':55.854}

def dwek_SNII(Mej):
    Mdust = {}
    Mdust['O'] = 0.0
    for metal in cond_eff_SNII:
        if metal != 'C' and metal != 'O':
            Mdust[metal] = cond_eff_SNII[metal]*Mej[metal]
            print(metal)
            Mdust['O'] = Mdust['O'] + Mdust[metal]/mu[metal]
        elif metal == 'C':
            Mdust[metal] = cond_eff_SNII[metal]*Mej[metal]
    Mdust['O'] = mu['O'] * Mdust['O']
    
    Mdust['Sil'] = 0.0
    for metal in Sil_comp:
        Mdust['Sil'] = Mdust['Sil'] + Mdust[metal]
    return Mdust

# Olivine composition MgFeSiO4
nsil = {'Mg':1,'Fe':1,'Si':1,'O':4}

def dwek_SNII_yohan(Mej):
    Mdust = {}
    for metal in cond_eff_SNII:
        if metal != 'C':
            Mdust[metal] = Mej[metal]/(mu[metal]*nsil[metal])
        elif metal == 'C':
            Mdust[metal] = cond_eff_SNII[metal]*Mej[metal]
    Mcarbon = Mdust['C']
    del Mdust['C']
    keyElement = min(Mdust,key=Mdust.get)
    MSil = cond_eff_SNII[keyElement]*Mdust[keyElement]*(mu[keyElement]*nsil[keyElement])
    return Mcarbon,MSil

def dwek_AGB_yohan(Mej):
    if Mej['C']>=Mej['O']:
        Msil = 0.0
        Mcarbon = cond_eff_AGB['C/O>1']*(Mej['C'] - (mu['C']/mu['O'])*Mej['O'])
    elif Mej['C']<Mej['O']:
        Mcarbon = 0.0
        Mdust = {}
        for metal in cond_eff_SNII:
            if metal != 'C':
                Mdust[metal] = Mej[metal]/(mu[metal]*nsil[metal])
        keyElement = min(Mdust,key=Mdust.get)
        Msil = cond_eff_AGB['C/O<1']*Mdust[keyElement]*(mu[keyElement]*nsil[keyElement])
    else:
        Mcarbon,Msil = 0.0,0.0
    return Mcarbon,Msil

def add_dust_AGB(agbdata):
    agbdata['CDust'] = np.zeros(agbdata['p'].shape)
    agbdata['SilDust'] = np.zeros(agbdata['p'].shape)

    nx,ny = agbdata['p'].shape
    for i in range(0, nx):
        for j in range(0, ny):
            Mej = {'Mg':agbdata['Mg'][i][j],'Fe':agbdata['Fe'][i][j],
                    'O':agbdata['O'][i][j],'Si':agbdata['Si'][i][j],
                    'C':agbdata['C'][i][j]}
            Mcarbon,Msil = dwek_AGB_yohan(Mej)
            agbdata['CDust'][i,j] = Mcarbon
            agbdata['SilDust'][i,j] = Msil
    return agbdata

def add_dust_SNII(sniidata):
    sniidata['CDust'] = np.zeros(sniidata['p'].shape)
    sniidata['SilDust'] = np.zeros(sniidata['p'].shape)

    nx,ny = sniidata['p'].shape
    for i in range(0, nx):
        for j in range(0, ny):
            Mej = {'Mg':sniidata['Mg'][i][j],'Fe':sniidata['Fe'][i][j],
                    'O':sniidata['O'][i][j],'Si':sniidata['Si'][i][j],
                    'C':sniidata['C'][i][j]}
            Mcarbon,Msil = dwek_SNII_yohan(Mej)
            sniidata['CDust'][i,j] = Mcarbon
            sniidata['SilDust'][i,j] = Msil
    return sniidata