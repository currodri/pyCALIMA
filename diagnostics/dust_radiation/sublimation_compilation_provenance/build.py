import json, math
from build_base import *
OUT='/private/tmp/claude-502/-Users-currodri-Library-CloudStorage-OneDrive-TheUniversityofChicago-HotDust-Quasar/9b0de9ae-12ba-4775-919a-eb50ad854d56/scratchpad/sublimation_data/sublimation_parameters.csv'
FE={(r['formula'],r['src']):r for r in json.load(open('freeevap_results.json'))}
NZ={r['solid']:r for r in json.load(open('nozawa_conv.json'))}
W18RHO={'C':2.27,'SiC':3.21,'TiC':4.93,'Fe':7.87,'Cr':7.19,'FeS':4.83,'MnS':4.08,'Na2S':1.86,'NaCl':2.17,'KCl':1.99,'Al2O3':3.97,'MgAl2O4':3.58,'CaTiO3':3.98,'TiO2':4.23,'Mg2SiO4':3.21,'MgSiO3':3.19,'Fe2SiO4':4.39,'SiO':2.18,'SiO2':2.65,'MgO':3.58,'FeO':5.99}
W18FIT={'C':'fit 4: ln p[dyn]=c0+c1/(T+c2) of C(g) atom','SiC':'fit 2 dGf[J/mol] from atoms','TiC':'fit 2','Fe':'fit 3: ln pvap[dyn] poly','Cr':'fit 3','FeS':'fit 3 (pvap of FeS(g) molecule)','MnS':'fit 1 (Sharp & Huebner 1990; 1 atm)','Na2S':'fit 2','NaCl':'fit 3 (NaCl(g) monomer)','KCl':'fit 3 (KCl(g) monomer)','Al2O3':'fit 5: -dGf/RT from atoms','MgAl2O4':'fit 5','CaTiO3':'fit 1 (Sharp & Huebner 1990; 1 atm)','TiO2':'fit 3 (pvap of TiO2(g) molecule)','Mg2SiO4':'fit 2','MgSiO3':'fit 2','Fe2SiO4':'fit 5','SiO':'fit 10 (Gail et al. 2013)','SiO2':'fit 3 (pvap of SiO2(g) molecule)','MgO':'fit 3 (pvap of MgO(g) molecule)','FeO':'fit 3 (pvap of FeO(g) molecule)'}
def w18(mat,formula,phase,src='D2',cong='',alpha='',acond='',sigma='',ssrc='',Tc='',Tcc='',notes='',rho=None):
    r=FE[(formula,src)]
    vap='; '.join(f'{k}:{v}' for k,v in r['vap'] if v>=0.005)
    tab='Table D.2 '+W18FIT.get(formula,'') if src=='D2' else 'Table D.4 (SUPCRTBL) -dGf/RT=b0/T+b1 lnT+b2+b3T+b4T^2 from atoms, 1 bar'
    add(material=mat,formula=formula,phase=phase,source_id=f'W18-{src}+FE',source_kind='derived (this work) from thermochemical compilation (NIST-JANAF / SUPCRTBL via Woitke et al. 2018)',
        original_form=tab+'; gas kp from Table D.1 (fit 4, neutral molecules)',original_coefficients='see Woitke et al. 2018 tables; reproduced by scripts in NOTES',
        A10=r['A'],B10_K=r['B'],T_range_K=f"{r['T1']}-{r['T2']} (fit; max dev {r['maxdev_dex']:.3f} dex)",
        vapour_species=f"equilibrium free-evaporation vapour mole fractions at {r['Tmid']:.0f} K: "+vap,
        congruency=cong or 'congruent ASSUMED (vacuum, equal alpha for all species)',alpha=alpha,alpha_conditions=acond,
        mu_unit_amu=f"{r['mfu']:.3f}",rho_g_cm3=(rho if rho is not None else W18RHO.get(formula,'')),sigma_erg_cm2=sigma,sigma_source=ssrc,Tcond_K=Tc,Tcond_condition=Tcc,
        verification='derived; inputs [V-txt] from arXiv:1712.01010 tables',notes='P_eff = J_fu*sqrt(2 pi m_fu k T)/alpha, J_fu = formula units cm^-2 s^-1 for congruent Langmuir evaporation into vacuum. '+notes)
def noz(mat,formula,phase,alpha='',acond='',cong='',Tc='',Tcc='',notes='',rho=''):
    r=NZ[formula]; prod=' + '.join(f"{n}{s}" if n>1 else s for s,n in r['products'].items())
    add(material=mat,formula=formula,phase=phase,source_id='N03+conv',source_kind='thermochemical fit (JANAF 1985) converted (this work)',
        original_form=f"dG0/kT = -A/T + B per key species for condensation {prod.replace(' + ',' + ')} -> {formula}(s); p in bar [unit inferred: JANAF 1985 std state 1 bar]",
        original_coefficients=f"A={r['A']:.6g} K, B={r['B']}; key species per formula unit = {r['nu']}",
        A10=r['A10'],B10_K=r['B10'],T_range_K='not stated (fit over "temperatures of interest")',vapour_species=prod+' (only these products allowed)',
        congruency=cong or 'congruent ASSUMED in conversion',alpha=alpha,alpha_conditions=acond,mu_unit_amu=f"{r['mfu']:.3f}",rho_g_cm3=rho,
        sigma_erg_cm2=r['sigma'],sigma_source='Nozawa et al. 2003 Table 2 (orig. refs therein)',Tcond_K=Tc,Tcond_condition=Tcc,
        verification='[V-txt] arXiv astro-ph/0307108 Table 2',notes='Conversion: P_eff from prod p_i^n_i = exp(nu(B-A/T)) with stoichiometric fluxes; a0='+f"{r['a0']} A. "+notes)
L03='Lodders 2003 Table 8, 1e-4 bar, solar gas [HTML-x, single extraction]'
MO12='Morley et al. 2012 (1 bar, solar), approx.'
# ---------------- GRAPHITE ----------------
G='graphite'
add(material=G,formula='C',phase='graphite',source_id='GD89',source_kind='thermochemical fit (Chase et al. 1985)',original_form='p_sat(C1) = 6e11 T^0.5 exp(-81200/T) dyn cm^-2 (eq. 3.12); R = 4.6e29 (alpha_N/0.1) exp[-(81200-20000 N^-1/3)/T] cm^-2 s^-1 (eq. 3.13)',
    original_coefficients='6e11, 0.5, 81200 K; theta_inf = 2e4 K; alpha_N = 0.1 adopted (arbitrary)',A10='',B10_K='',T_range_K='3000-6000 (fit to within 14%)',vapour_species='C1 only',congruency='congruent',
    alpha='0.1 (adopted, arbitrary)',alpha_conditions='GD89: sticking coefficients unknown',mu_unit_amu='12.011',rho_g_cm3='',Bk_K='81200',sigma_erg_cm2='~2000 (derived here from theta_inf=2e4 K, rho=2.24)',sigma_source='derived from GD89 theta_inf definition',
    verification='[V-img] ADS scan p.236',notes='P0 column = 6e11/sqrt(2 pi m k) = 4.56e30 cm^-2 s^-1 (T-independent because p_sat ~ T^0.5). Footnote 2 better fit: p_sat(C1) ~ 2.2e16 T^-0.575 exp(-87300/T) dyn cm^-2 (600-6000 K, 25%).',P0x=4.56e30)
add(material=G,formula='C',phase='graphite (pyrolytic)',source_id='Z70-C1',source_kind='laboratory (Knudsen effusion + TOF mass spec, vacuum)',original_form='log10 P_C1[atm] = A - M*1e4/T',original_coefficients='A=8.072, M=3.7180',A10=14.078,B10_K=37180,T_range_K='2423-3003',
    vapour_species='C1 partial pressure',congruency='congruent',alpha='alpha1=0.37 (TW57) / 0.24 (Z66)',alpha_conditions='vacuum, Langmuir vs Knudsen; species-specific',mu_unit_amu='12.011',rho_g_cm3='2.27',verification='[P-img] OSTI 4717386 Table III',notes='Zavitsanos 1970 GE report (basis of Zavitsanos & Carlson 1973 JCP 59, 2966 which was not opened).')
add(material=G,formula='C2',phase='graphite (pyrolytic)',source_id='Z70-C2',source_kind='laboratory',original_form='log10 P_C2[atm] = A - M*1e4/T',original_coefficients='A=10.106, M=4.4830',A10=16.112,B10_K=44830,T_range_K='2423-3003',vapour_species='C2 partial pressure',congruency='congruent',
    alpha='alpha2=0.34 (TW57) / 0.50 (Z66)',alpha_conditions='vacuum',mu_unit_amu='24.022',rho_g_cm3='2.27',verification='[P-img] OSTI 4717386 Table III',notes='mu = C2 molecule mass; multiply molecule flux by 2 for C-atom flux.')
add(material=G,formula='C3',phase='graphite (pyrolytic)',source_id='Z70-C3',source_kind='laboratory',original_form='log10 P_C3[atm] = A - M*1e4/T',original_coefficients='A=10.168, M=4.1343',A10=16.174,B10_K=41343,T_range_K='2423-3003',vapour_species='C3 partial pressure (dominant equilibrium species, ~5x C1 at 3000 K)',congruency='congruent',
    alpha='alpha3=0.08 (TW57) / 0.023 (Z66)',alpha_conditions='vacuum; small alpha3 makes C1/C2 dominate the free-evaporation flux (Clarke & Fox 1969)',mu_unit_amu='36.033',rho_g_cm3='2.27',verification='[P-img] OSTI 4717386 Table III',notes='mu = C3 molecule mass; multiply by 3 for atom flux.')
add(material=G,formula='C3',phase='graphite',source_id='TW57',source_kind='laboratory (Langmuir vs Knudsen, vacuum)',original_form='log10 p3[atm] = -40296.0/T + 9.811',original_coefficients='9.811, 40296',A10=15.817,B10_K=40296,T_range_K='~2400',vapour_species='C3',congruency='congruent',
    alpha='overall ~0.15; alpha(C1,C2,C3)=0.37,0.34,0.08',alpha_conditions='vacuum',mu_unit_amu='36.033',rho_g_cm3='2.27',verification='[ABS] Thorn & Winslow 1957 JCP 26, 186',notes='')
add(material=G,formula='C',phase='graphite',source_id='Z66',source_kind='laboratory (Langmuir 2250-2760 K)',original_form='alpha ratios only',original_coefficients='alpha2/alpha1=2.1, alpha1/alpha3=10.6; overall 0.095',T_range_K='2250-2760',vapour_species='C1,C2,C3',
    alpha='overall 0.095; alpha1=0.24, alpha2=0.50, alpha3=0.023',alpha_conditions='vacuum',mu_unit_amu='',verification='[ABS] DTIC AD0633611',notes='Hoch & Yamawaki 1972 warn pyrolytic graphite vaporises predominantly from cracks. Pound 1972 Table 1 lists alpha=1+-0.1 (Marshall & Norton 1950, total mass loss, 2357-2870 K) [P-img].')
add(material=G,formula='C3',phase='graphite',source_id='vL14a',source_kind='compilation (quoting Zavitsanos & Carlson 1973)',original_form='p_v = exp(-A/T + B) dyn cm^-2',original_coefficients='A=93646+-503 K, B=36.7+-1.8',A10=36.7/LN10,B10_K=93646/LN10,T_range_K='2400-3000',
    vapour_species='C3 component',congruency='congruent',alpha='0.1 (adopted, arbitrary)',alpha_conditions='',mu_unit_amu='12.011',rho_g_cm3='2.16',verification='[V-txt] arXiv:1410.3494 Table 3',
    notes='vL use mu of the dust compound (12.011) in J, not C3 (their footnote 2). P0/nu0 columns use 12.011 as in source.')
B2=36.03*mu*2.15e11/kB
add(material=G,formula='C3',phase='graphite (used for carbonaceous dust)',source_id='vL14b',source_kind='compilation (quoting Zavitsanos & Carlson 1973)',original_form='P_v = P0 exp(-mu m_u H/kT)',original_coefficients='P0=2.95e14 dyn cm^-2, H=2.15e11 erg/g, mu=36.03',A10=math.log10(2.95e14),B10_K=B2/LN10,T_range_K='',
    vapour_species='C3',alpha='1 (sticking efficiency assumed unity)',mu_unit_amu='36.03',rho_g_cm3='1.8',verification='[V-txt] arXiv:1404.3271 Sect. 2.2.1',notes='[inconsistent] P is ~25x below vL14a/Z70 C3 at 2500 K although both cite Z73; prefer Z70/vL14a.')
add(material=G,formula='C',phase='graphite (dominant species)',source_id='Ph89',source_kind='thermochemical fit (Kelley 1973) + vacuum sublimation',original_form='p_s = 6e15 exp(-92200/T) dyn cm^-2 (eq. 9)',original_coefficients='6e15, 92200 K',A10=15.778,B10_K=40042,T_range_K='not stated',vapour_species='dominant gas-phase constituent',
    alpha='S_i~1',mu_unit_amu='',verification='[P-img] NED Level5 HTML; book pagination [unverified]',notes='Phinney 1989 (AGN dust). Kelley 1973 h0/k: 8.58e4 (C), 9.82e4 (C2), 9.45e4 K (C3); silicate ~6.6e4 K.')
add(material=G,formula='C',phase='"carbon"',source_id='K11',source_kind='compilation (source of values not stated)',original_form='P_v = P0 exp(-mu m_u H/kT)',original_coefficients='P0=4.31e16 dyn cm^-2, H=7.27e11 erg/g, mu=12.0, rho=1.95',A10=math.log10(4.31e16),B10_K=12.0*mu*7.27e11/kB/LN10,
    vapour_species='',mu_unit_amu='12.0',rho_g_cm3='1.95',verification='[V-txt] arXiv:1104.5627 Table 1',notes='Kobayashi et al. 2011 EPS 63, 1067. B/k=104900 K is much larger than graphite data; use with caution.')
for n,(A,B) in {'C1':(14.111,37236),'C2':(15.663,42857),'C3':(15.789,40236)}.items():
    add(material=G,formula=n,phase='graphite',source_id='JANAF-'+n,source_kind='thermochemical (NIST-JANAF 4th ed.) fitted by fork B',original_form='log10 P[bar] = log10 Kf; linear fit on JANAF grid',original_coefficients=f'A[bar]={A-6:.3f}, B={B}',A10=A,B10_K=B,T_range_K='2000-4000',vapour_species=n,
        mu_unit_amu=f'{12.011*int(n[1]):.3f}',rho_g_cm3='2.27',verification='derived from janaf.nist.gov tables (max resid <=0.055 dex)',notes='C1+C2+C3 ~2x below CRC (Leider et al. 1973) at 2839 K.')
w18(G,'C','graphite',notes='Total C-atom flux incl. C1..C5 (C3 and C dominate); equal-alpha assumption is poor for graphite (alpha3<<alpha1).')
noz(G,'C','graphite',notes='C1 atom only (underestimates total vapour by ~10x at 2500 K).',rho='')
# ---------------- AMORPHOUS CARBON ----------------
add(material='amorphous carbon',formula='C',phase='amorphous',source_id='none',source_kind='no laboratory data found',original_form='',T_range_K='',vapour_species='',alpha='',
    notes='No P_sat or alpha distinct from graphite found. Models adopt graphite (e.g. vL14b with rho=1.8; Kobayashi et al. 2011 "carbon" rho=1.95). Amorphous C is metastable w.r.t. graphite, so its P_sat should be >= graphite (thermodynamic argument; magnitude unknown).',verification='literature search negative')
# ---------------- SILICATES ----------------
add(material='astronomical silicate',formula='Mg2SiO4 (GD89 proxy)',phase='silicate',source_id='GD89',source_kind='thermochemical (Chase et al. 1985), vapour of cosmic Mg:Si:O',original_form='R_Si = 7e30 <alpha_Si> exp(-68100/T) cm^-2 s^-1 (eq. 3.17); R_N+1 = 7e29 (<alpha>/0.1) exp[-(68100-20000 N^-1/3)/T] (3.18); total atoms R_N = 7 R_Si (3.19)',
    original_coefficients='7e30, 68100 K, theta_inf=2e4 K, rho=3.5, N=1.22(a/A)^3',Bk_K='68100',T_range_K='not stated',vapour_species='Mg, MgO, Si, SiO, SiO2, O, O2 considered',congruency='congruent (implicit)',alpha='0.1 (adopted)',
    mu_unit_amu='140.69',rho_g_cm3='3.5',sigma_erg_cm2='~1900 (derived here from theta_inf, per-atom volume)',sigma_source='derived from GD89',verification='[V-img] ADS scan pp.236-237',
    notes='P0 column = R_Si/<alpha> per formula unit (=per Si). Per atom multiply by 7. WD00 quote nu0=2e15 s^-1 (Mg2SiO4) and 2e14 s^-1 (graphite), both consistent with alpha=0.1.',P0x=7e30)
add(material='astronomical silicate',formula='olivine',phase='"olivine" (interplanetary)',source_id='K11-ol',source_kind='compilation (source not stated; numerically identical B/k to vL14a forsterite/Nagahara 1994)',original_form='P_v = P0 exp(-mu m_u H/kT)',original_coefficients='P0=6.72e14 dyn cm^-2, H=3.21e10 erg/g, mu=169.1, rho=3.3',
    A10=math.log10(6.72e14),B10_K=169.1*mu*3.21e10/kB/LN10,mu_unit_amu='169.1',rho_g_cm3='3.3',verification='[V-txt] arXiv:1104.5627 Table 1',notes='mu=169.1 suggests MgFeSiO4 composition.')
add(material='astronomical silicate',formula='pyroxene',phase='"pyroxene"',source_id='K11-px',source_kind='compilation (source not stated)',original_form='P_v = P0 exp(-mu m_u H/kT)',original_coefficients='P0=3.12e11, H=9.60e10 erg/g, mu=60.1, rho=3.3',
    A10=math.log10(3.12e11),B10_K=60.1*mu*9.60e10/kB/LN10,mu_unit_amu='60.1',rho_g_cm3='3.3',verification='[V-txt] arXiv:1104.5627 Table 1',notes='mu=60.1 = SiO2 mass (evaporating unit taken as SiO2?).')
add(material='astronomical silicate',formula='obsidian',phase='obsidian glass',source_id='K11-ob',source_kind='compilation',original_form='P_v = P0 exp(-mu m_u H/kT)',original_coefficients='P0=1.07e14, H=7.12e10, mu=67.0, rho=2.37',
    A10=math.log10(1.07e14),B10_K=67.0*mu*7.12e10/kB/LN10,mu_unit_amu='67.0',rho_g_cm3='2.37',verification='[V-txt] arXiv:1104.5627 Table 1',notes='Comparison material only.')
FO='forsterite'
add(material=FO,formula='Mg2SiO4',phase='crystalline',source_id='vL14a',source_kind='laboratory (Nagahara et al. 1994, Knudsen) via compilation',original_form='p_v = exp(-A/T + B) dyn cm^-2',original_coefficients='A=65308+-3969 K, B=34.1+-2.5',A10=34.1/LN10,B10_K=65308/LN10,T_range_K='1673-2133',
    vapour_species='Mg, SiO, O, O2 (total)',congruency='congruent (vacuum, Hashimoto 1989; H2, Tsuchiyama et al. 1998)',alpha='0.1 (Gail 2010 via vL); measured 0.09-0.14',alpha_conditions='vacuum 1873-2073 K (Hashimoto 1989); compilation 0.03-0.2, ~0.1 in vacuum and H2 (Tsuchiyama et al. 1998 Table 2)',
    mu_unit_amu='140.694',rho_g_cm3='3.27',sigma_erg_cm2='436 (Nozawa) / 4410+-210 calorimetric (Chen & Navrotsky 2010)',sigma_source='N03; Householder et al. 2025 poster',Tcond_K='1354 [unverified]',Tcond_condition=L03,verification='[V-txt] arXiv:1410.3494 Table 3',
    notes='Hashimoto 1989 alpha(T) two-point (fork): log10 alpha = 1.740 - 5250/T (0.04 at 1673 K, 0.12 at 1973 K) [derived from rounded values]. Face dependence: Vc>Va>=Vb (Vc/Vb~4.7 vacuum, ~3 in H2; Yokoyama et al. 2007).')
w18(FO,'Mg2SiO4','crystalline',cong='congruent (lab-confirmed)',alpha='~0.1 (0.03-0.2)',acond='vacuum & H2 (Tsuchiyama et al. 1998 compilation)',sigma='436',ssrc='N03',Tc='1354 [unverified]',Tcc=L03)
w18(FO,'Mg2SiO4','crystalline',src='D4',cong='congruent (lab-confirmed)',alpha='~0.1',acond='as above',rho='3.21')
noz(FO,'Mg2SiO4','crystalline',alpha='~0.1',rho='3.21')
add(material=FO,formula='Mg2SiO4',phase='crystalline',source_id='Hash89',source_kind='laboratory (vacuum Langmuir)',original_form='alpha_v = J_L/J_e (all vapour species in J_e)',original_coefficients='alpha=0.09-0.14 (1600-1800 C); alpha_v\'=0.4-0.6 if only Mg, SiO2, O counted; dH*=136.6+-1.9 kcal/mol',
    T_range_K='1873-2073',vapour_species='postulated kinetic path Mg2SiO4 -> 2Mg + SiO2(g) + 2O (or O2)',congruency='congruent',alpha='0.09-0.14 (increasing with T)',alpha_conditions='vacuum',verification='[V-img] LPSC XX 385 (1989)',notes='Precursor of Hashimoto 1990 Nature 347, 53 (not opened).')
add(material=FO,formula='Mg2SiO4',phase='crystalline {010}',source_id='Ts98',source_kind='laboratory (H2 gas, 1.4e-5 bar)',original_form='j_Fo = 2480 exp(-372 kJ/mol / RT) mol cm^-2 s^-1',original_coefficients='2480 mol cm^-2 s^-1; 372 kJ/mol',T_range_K='1473-1723',
    vapour_species='Mg, SiO, H2O',congruency='congruent',alpha='0.04-0.12 (this study); ~0.1 overall',alpha_conditions='p(H2)=1.4e-5 bar',verification='[V-img] Mineral. J. 20, 113 (J-STAGE)',notes='H2 accelerates evaporation (atomic H controls: Kuroda & Hashimoto 2002 log10 J[g cm^-2 s^-1] = -32.6e3/(2.303RT) + log10 P_H[atm] + 4.00). Not vacuum P_sat.')
EN='enstatite'
add(material=EN,formula='MgSiO3',phase='crystalline',source_id='vL14a',source_kind='laboratory (Mysen & Kushiro 1988; Kushiro & Mysen 1991) via compilation',original_form='p_v = exp(-A/T + B) dyn cm^-2',original_coefficients='A=68908+-8773 K, B=38.1+-5.0',A10=38.1/LN10,B10_K=68908/LN10,T_range_K='1573-1923',
    vapour_species='',congruency='INCONGRUENT: SiO2 component lost preferentially, forsterite residue layer (Tachibana et al. 2002)',alpha='0.1 (adopted, arbitrary)',alpha_conditions='no measured alpha found',mu_unit_amu='100.389',rho_g_cm3='3.20',
    sigma_erg_cm2='400 (Nozawa) / 4790+-450 calorimetric',sigma_source='N03; Householder et al. 2025 poster',verification='[V-txt] arXiv:1410.3494 Table 3',notes='Equilibrium gas Mg/Si~0.053 over En+Fo (Tachibana & Tsuchiyama 1999). Steady Fo layer <=0.1 um under nebular conditions.')
w18(EN,'MgSiO3','crystalline',cong='INCONGRUENT in reality (Fo residue); congruent calc is an upper-limit-type approximation',sigma='400',ssrc='N03')
w18(EN,'MgSiO3','crystalline',src='D4',cong='INCONGRUENT in reality',rho='3.19')
noz(EN,'MgSiO3','crystalline',cong='congruent assumed (actually incongruent)',rho='3.19')
add(material=EN,formula='MgSiO3',phase='crystalline',source_id='Tach02',source_kind='laboratory (vacuum & H2)',original_form='diffusion-controlled rate constant',original_coefficients='Ea=457+-58 kJ/mol; k_En(1400 C, vacuum)=6e-13 cm^2 s^-1 (Tachibana & Tsuchiyama 1999)',T_range_K='1573-1773',
    vapour_species='SiO-rich (rate species SiO in their model)',congruency='INCONGRUENT -> forsterite layer',alpha='not found',verification='[abs] GCA 66, 713; [V-txt] LPSC 1532 (1999)',notes='')
FA='fayalite'
add(material=FA,formula='Fe2SiO4',phase='crystalline',source_id='vL14a',source_kind='laboratory (Nagahara et al. 1994) via compilation',original_form='p_v = exp(-A/T + B) dyn cm^-2',original_coefficients='A=60377+-1082 K, B=37.7+-0.7',A10=37.7/LN10,B10_K=60377/LN10,T_range_K='1373-1433',
    vapour_species='Fe, SiO, O2 (expected)',congruency='see Fe-bearing olivine notes',alpha='0.1 (adopted; Gail & Sedlmayr 1999)',mu_unit_amu='203.774',rho_g_cm3='4.39',sigma_erg_cm2='2470+-200 calorimetric (Lilova et al. 2018)',sigma_source='Householder et al. 2025 poster',
    verification='[V-txt] arXiv:1410.3494 Table 3',notes='~1.6 dex above the JANAF free-evaporation estimate at 1400 K [discrepancy unresolved].')
w18(FA,'Fe2SiO4','crystalline',cong='congruent assumed',alpha='0.1 (adopted)',acond='G&S99')
add(material='ferrosilite',formula='FeSiO3',phase='crystalline',source_id='W18-D4+FE',source_kind='see next row',notes='No laboratory data found; only thermochemistry (next row).',verification='')
ROWS.pop()
w18('ferrosilite','FeSiO3','crystalline (SUPCRTBL)',src='D4',notes='No laboratory data found. FeSiO3 is metastable at low P (would decompose to fayalite + silica); treat as rough.',rho='')
add(material='Mg-Fe olivine (Fo95Fa5)',formula='(Mg0.95Fe0.05)2SiO4',phase='crystalline',source_id='Costa17',source_kind='laboratory (Knudsen effusion mass spectrometry)',original_form='log10 P_i[bar] = a - b/T (Fig. 6)',
    original_coefficients='Fe: 8.3473-26591; SiO: 6.7264-24122; Mg: 11.3808-33941; O: 5.4275-22093; O2: 3.4588-17866',A10=8.3473+6,B10_K=26591,T_range_K='~1750-1960 (solid)',
    vapour_species='Fe > SiO > Mg > O2 > O (A10/B10 columns = Fe partial pressure)',congruency='INCONGRUENT: (Fe/Mg)gas > (Fe/Mg)olivine; Fo95 -> Fo96 during runs',alpha='',mu_unit_amu='55.845',
    verification='[V-img] NSF-PAR manuscript of Icarus 289, 42 (axis label 10^4/T but coefficients consistent with 1/T)',notes='Hvap(Fo95Fa5)=585+-4 kJ/mol. Key evidence for Fe release. Other species: SiO 12.726-24122/T, Mg 17.381-33941/T [dyn].')
for f in ['MgFeSiO4','Mg0.8Fe0.2SiO3','Mg0.5Fe0.5SiO3','Mg0.4Fe0.6SiO3','Mg0.8Fe1.2SiO4']:
    add(material='intermediate Mg-Fe silicate',formula=f,phase='crystalline/amorphous',source_id='none',source_kind='no direct data',
        notes='No measured P_sat/alpha. Lab: Fe-bearing olivine evaporates faster than forsterite (~5x at 1673 K, San Carlos Fo~90; Nagahara & Ozawa 1998) and loses Fe preferentially (diffusion-limited); Costa et al. 2017 KEMS on Fo95. Options: interpolate between end members with activities, or use Fe-loss + forsterite residue model (see NOTES).',verification='literature search negative')
add(material='amorphous silicate',formula='Mg/Fe silicate glass',phase='amorphous',source_id='none',source_kind='no data',notes='No laboratory evaporation kinetics for amorphous Mg-Fe silicates found. Calorimetric surface enthalpies of amorphous: forsterite ~1657 (extrap.), enstatite 1800+-100, fayalite ~930 (extrap.) erg cm^-2 (Householder poster).',verification='search negative')
# ---------------- SILICA / SiO ----------------
S2='silica'
add(material=S2,formula='SiO2',phase='high cristobalite',source_id='vL14a',source_kind='laboratory (Hashimoto 1990) via compilation',original_form='p_v = exp(-A/T + B) dyn cm^-2 (SiO2(s)->SiO2(g))',original_coefficients='A=69444+-3447 K, B=33.1+-1.8',A10=33.1/LN10,B10_K=69444/LN10,T_range_K='1833-1958',
    vapour_species='SiO2(g) channel (equilibrium vapour is mainly SiO + O2)',congruency='congruent',alpha='1.0 relative to SiO2(g) (0.7-0.9 measured)',alpha_conditions='vacuum',mu_unit_amu='60.084',rho_g_cm3='2.60',
    sigma_erg_cm2='605 (Nozawa) / 1000+-70 cryst., 260 amorph. (calorimetric)',sigma_source='N03; Householder poster (Parks 1990; Chiang 1996)',verification='[V-txt] arXiv:1410.3494 Table 3 note f',notes='Use THIS row with alpha~1, OR the equilibrium rows below with alpha~0.01; both describe the same measured flux.')
add(material=S2,formula='SiO2',phase='high cristobalite',source_id='Hash89',source_kind='laboratory (vacuum Langmuir)',original_form='alpha_v=J_L/J_e',original_coefficients='alpha=0.011-0.015 (all species); alpha\'=0.7-0.9 (SiO2(g) only); dH*=138.0+-13.7 kcal/mol',T_range_K='1833-1958',
    vapour_species='equilibrium: SiO >> O2 > O; kinetic: SiO2(g)',congruency='congruent',alpha='0.011-0.015 (vs total equilibrium vapour)',alpha_conditions='vacuum; G&S99 adopt 0.01',verification='[V-img] LPSC XX 385',notes='')
w18(S2,'SiO2','quartz',alpha='0.011-0.015 (Hashimoto 1989)',acond='vacuum',sigma='605',ssrc='N03')
w18(S2,'SiO2','quartz (SUPCRTBL)',src='D4',alpha='0.011-0.015')
noz(S2,'SiO2','quartz',alpha='0.011-0.015',rho='2.65')
add(material=S2,formula='SiO2',phase='quartz',source_id='Lamy74/Schick60',source_kind='laboratory (quoted by Lamy 1974 from Schick 1960)',original_form='log10 p_SiO[Torr] = 11.445 - 26335/T (T<2000 K); 10.876 - 25265.2/T (T>2000 K)',original_coefficients='as given',A10=14.570,B10_K=26335,T_range_K='<2000 (second branch >2000)',
    vapour_species='SiO (+1/2 O2)',alpha='0.7 (Lamy adopted, Delsemme 1965)',mu_unit_amu='44.085',rho_g_cm3='2.65',verification='[P-img] ADS 1974A&A....35..197L eq.2',notes='p_SiO only; Schick-type data considered too high for solid SiO by Gail et al. 2013.')
add(material=S2,formula='SiO2',phase='pure SiO2',source_id='Costa17-SiO2',source_kind='laboratory (KEMS)',original_form='log10 Kp[SiO2(s)=SiO+1/2O2] = -38642/T + 11.87',original_coefficients='38642+-320, 11.87+-0.17',T_range_K='~1830-1980',vapour_species='SiO >> O2 > O',verification='[V-img] Fig. 4C',notes='Kp presumably bar^1.5. Not converted.')
SO='silicon monoxide'
add(material=SO,formula='SiO',phase='solid (amorphous)',source_id='G13',source_kind='laboratory (Knudsen cell + QCM)',original_form='P = exp(-T_v/T + S_v) dyn cm^-2 (eq.4); alpha = alpha0 exp(-T0/T) (eq.3)',original_coefficients='T_v=49520+-1400 K, S_v=32.52+-0.97; alpha0=0.521+-0.156, T0=3686+-428 K',
    A10=32.52/LN10,B10_K=49520/LN10,T_range_K='1275-1525',vapour_species='SiO',congruency='congruent',alpha='0.521 exp(-3686/T) (~0.04 at 1400 K)',alpha_conditions='vacuum',mu_unit_amu='44.085',rho_g_cm3='2.13',
    verification='[V-txt] arXiv:1305.2879',notes='Gail et al. 2013 A&A 555, A119. Same as vL14a & Woitke fit 10.')
add(material=SO,formula='SiO',phase='solid',source_id='FN08',source_kind='laboratory (quoted by Gail et al. 2013 from Ferguson & Nuth 2008)',original_form='P = exp(-T_v/T + S_v) dyn cm^-2',original_coefficients='T_v=40850+-1270, S_v=32.93+-0.90',A10=32.93/LN10,B10_K=40850/LN10,vapour_species='SiO',mu_unit_amu='44.085',rho_g_cm3='2.13',verification='[2nd]',notes='')
add(material=SO,formula='SiO',phase='solid',source_id='Schick60',source_kind='laboratory (old; quoted by Gail et al. 2013; = G&S99 eq.14 from Nuth & Donn 1982)',original_form='p = exp(26.66 - 25670/T) dyn cm^-2',original_coefficients='26.66, 25670',A10=26.66/LN10,B10_K=25670/LN10,vapour_species='SiO',mu_unit_amu='44.085',rho_g_cm3='2.13',verification='[2nd]',notes='Gail+2013: overestimates p_vap; do not use.')
# ---------------- Fe ----------------
F='iron'
add(material=F,formula='Fe',phase='metal',source_id='Tach11',source_kind='laboratory (vacuum)',original_form='ln alpha_e = 1.44(+-0.22) - 2595(+-353)/T (eq.3); with ambient vapour alpha = exp(1.44-2595/T) - 0.54 p/p_eq (eq.4)',original_coefficients='1.44, 2595 K, 0.54',T_range_K='1347-1718',
    vapour_species='Fe',congruency='congruent',alpha='0.6 (1350 K) - 0.9 (1720 K); ~1 near m.p.',alpha_conditions='vacuum; condensation alpha_c~1 at S>10, heterogeneous nucleation on Al2O3',verification='[V-txt] ApJ 736, 16 full text',notes='Pound 1972: 1+-0.2 (1540-1740 K), 0.9+-0.1 (1358-1520 K).')
add(material=F,formula='Fe',phase='metal',source_id='JANAF-Fe',source_kind='thermochemical (NIST-JANAF) fitted by fork B',original_form='log10 P[bar]=7.124-20924/T',original_coefficients='',A10=13.124,B10_K=20924,T_range_K='1000-1800',vapour_species='Fe',congruency='congruent',alpha='exp(1.44-2595/T)',alpha_conditions='Tachibana et al. 2011',
    mu_unit_amu='55.845',rho_g_cm3='7.87',sigma_erg_cm2='1800',sigma_source='N03 (Elliott et al. 1963)',Tcond_K='1357 (appearance), 1334 (50%)',Tcond_condition=L03,verification='derived; checks CRC 1 Pa at 1728 K',notes='RECOMMENDED for Fe.')
w18(F,'Fe','metal (alpha-delta)',alpha='exp(1.44-2595/T)',acond='Tach11',sigma='1800',ssrc='N03')
noz(F,'Fe','metal',rho='7.87')
add(material=F,formula='Fe',phase='metal',source_id='V10',source_kind='thermochemical',original_form='log10 p_Fe[bar] = 7.23 - 20995/T (eq.6)',original_coefficients='7.23, 20995',A10=13.23,B10_K=20995,vapour_species='Fe',mu_unit_amu='55.845',rho_g_cm3='7.87',verification='[V-txt] arXiv:1001.3639',notes='Visscher et al. 2010 ApJ 716, 1060.')
add(material=F,formula='Fe',phase='metal',source_id='vL14a',source_kind='laboratory (Ferguson et al. 2004) via compilation',original_form='p_v = exp(-A/T + B) dyn cm^-2',original_coefficients='A=48354+-1151, B=29.2+-0.7',A10=29.2/LN10,B10_K=48354/LN10,T_range_K='1573-1973',vapour_species='Fe',alpha='1.0',mu_unit_amu='55.845',rho_g_cm3='7.87',
    verification='[V-txt] arXiv:1410.3494; Ferguson abstract only',notes='~0.5 dex below JANAF; Ferguson et al. state their data are ~2x lower than previous.')
add(material=F,formula='Fe',phase='metal',source_id='Lamy74',source_kind='compilation (Smithells 1967)',original_form='log10 p[Torr] = -21080/T + 16.89 - 2.14 log10 T',original_coefficients='21080, 16.89, 2.14',T_range_K='900-1812',vapour_species='Fe',mu_unit_amu='55.845',rho_g_cm3='7.87',verification='[P-img] Lamy 1974',
    notes='Not of A-B/T form; = 20.015 - 21080/T - 2.14 log10 T [dyn]. Kobayashi et al. 2011 Table 1 footnote mis-transcribes this formula; their Fe H=2.97e10 erg/g, P0=5e4 dyn cm^-2 are suspect - do not use.')
# ---------------- Si, Cr ----------------
add(material='silicon',formula='Si',phase='solid',source_id='JANAF-Si',source_kind='thermochemical (NIST-JANAF) fitted by fork B',original_form='log10 P[bar]=7.577-23319/T',A10=13.577,B10_K=23319,T_range_K='1000-1600',vapour_species='Si',congruency='congruent',alpha='not found',mu_unit_amu='28.086',
    sigma_erg_cm2='800',sigma_source='N03 (Elliott & Gleiser 1960)',verification='derived',notes='Si(s) also oxidises readily; no alpha found.')
noz('silicon','Si','solid')
add(material='chromium',formula='Cr',phase='metal',source_id='JANAF-Cr',source_kind='thermochemical (NIST-JANAF) fitted by fork B',original_form='log10 P[bar]=7.156-20149/T',A10=13.156,B10_K=20149,T_range_K='1000-2100',vapour_species='Cr',congruency='congruent',alpha='0.9+-0.1',alpha_conditions='vacuum 1318-1563 K (McCabe et al. 1956 via Pound 1972)',
    mu_unit_amu='51.996',rho_g_cm3='7.19',sigma_erg_cm2='1880 (Nozawa) / 3330 (Lee et al. 2018)',sigma_source='N03; Lee+2018 Table A.2 (Tran+2016)',Tcond_K='~1520 (1 bar); 1296 (50%, into Fe alloy, 1e-4 bar)',Tcond_condition=MO12+'; '+L03,verification='derived; CRC 1 Pa at 1656 K')
add(material='chromium',formula='Cr',phase='metal',source_id='MO12',source_kind='thermochemical',original_form="log p'_Cr[bar] = 7.490 - 20592/T (eq.5)",A10=13.490,B10_K=20592,vapour_species='Cr',mu_unit_amu='51.996',rho_g_cm3='7.19',verification='[V-txt] arXiv:1206.4313',notes='')
w18('chromium','Cr','metal',alpha='0.9',acond='Pound 1972')
noz('chromium','Cr','metal',rho='7.19')
# ---------------- FeS ----------------
T='troilite'
add(material=T,formula='FeS',phase='troilite',source_id='JANAF-S2buf',source_kind='thermochemical (NIST-JANAF) fitted by fork B',original_form='FeS(s) = Fe(s) + 1/2 S2(g); log10 P_S2[bar] = 6.106 - 16289/T',A10=12.106,B10_K=16289,T_range_K='700-1400',
    vapour_species='S2 (at Fe/FeS buffer)',congruency='INCONGRUENT: S lost, porous Fe-metal residue (Tsuchiyama et al. 1997; Tachibana & Tsuchiyama 1998)',alpha='alpha_S2 = 0.23; alpha_H2S = 9.5e-4; q = 0.609',alpha_conditions='1073-1243 K, p(H2)=1e-6-1 atm (Tachibana & Tsuchiyama 1998, quoted by Tachibana et al. 1999)',
    mu_unit_amu='64.13',sigma_erg_cm2='380',sigma_source='N03 (Kozasa & Hasegawa 1988)',Tcond_K='704 (appearance), 664 (50% S)',Tcond_condition=L03,verification='derived (JANAF); alpha [V-img] Antarct. Met. Res. 12, 213',
    notes='RECOMMENDED: S flux J_S = 2 alpha_S2 q P_S2/sqrt(2 pi m_S2 k T); residual Fe then evaporates with Fe parameters. mu=S2 mass.')
add(material=T,formula='FeS',phase='pyrrhotite Fe0.886S',source_id='Ts97',source_kind='laboratory (1 atm H2-CO2, and vacuum)',original_form='k_FeS = (1.61+-0.42)e-3 exp(-115+-2 kJ/mol / RT) m/s',original_coefficients='1.61e-3 m/s, 115 kJ/mol',T_range_K='773-1573',
    vapour_species='H2S (in H2); Fe + 1/2 S2 (vacuum)',congruency='INCONGRUENT (spongy Fe residue); apparently congruent in vacuum only if alpha_FeS << 1',alpha='1.3e-4 (900 C) - 9.4e-6 (500 C)',alpha_conditions='p(H2)~0.63 atm',verification='[V-txt] Geochem. J. 31, 289',notes='Not a vacuum P_sat.')
w18(T,'FeS','troilite',cong='NOT congruent: congruent Fe pressure exceeds P_sat(Fe metal) -> Fe residue; this row is a formal upper bound',sigma='380',ssrc='N03')
noz(T,'FeS','troilite',cong='congruent assumed (actually incongruent)',rho='4.83')
# ---------------- sulfides, salts ----------------
w18('alabandite','MnS','MnS',notes='Woitke fit 1 from Sharp & Huebner 1990.',sigma='2326 (est.)',ssrc='quoted by Householder poster (orig. unspecified)',Tc='~1340 (1 bar)',Tcc=MO12)
add(material='alabandite',formula='MnS',phase='MnS',source_id='MO12',source_kind='thermochemical (Visscher et al. 2006), H2S-bearing solar gas',original_form="log p'_Mn[bar] = 11.532 - 23810/T - [Fe/H]",verification='[V-txt] arXiv:1206.4313',notes='Not a vacuum P_sat (H2S of solar gas built in). No alpha found.')
add(material='sphalerite',formula='ZnS',phase='ZnS',source_id='MO12',source_kind='thermochemical (Visscher et al. 2006), H2S-bearing solar gas',original_form="log p'_Zn[bar] = 12.812 - 15873/T - [Fe/H]",vapour_species='Zn (+H2S in gas)',sigma_erg_cm2='1760 calorimetric (Subramani et al. 2022) / 860 (model)',sigma_source='Householder poster',
    Tcond_K='~800 (1 bar)',Tcond_condition=MO12,verification='[V-txt] arXiv:1206.4313',notes='No vacuum P_sat (JANAF and Woitke lack ZnS; Zn not in GGchem). No alpha.')
w18('sodium sulfide','Na2S','Na2S',cong='congruent assumed (no condensed residue expected; Na metal far more volatile)',sigma='1033 (est.)',ssrc='quoted by Householder poster',Tc='~1000 (1 bar)',Tcc=MO12)
add(material='sodium sulfide',formula='Na2S',phase='Na2S',source_id='JANAF-Na2S',source_kind='thermochemical (NIST-JANAF) congruent calc by fork B',original_form='log10 K[bar^2.5] = 15.258 - 32758/T (Na2S -> 2Na + 1/2 S2)',A10=12.179,B10_K=13103,T_range_K='600-1400',vapour_species='A10/B10 = P(Na); P(S2)=11.800-13103/T',congruency='congruent',mu_unit_amu='22.990',verification='derived',notes='Agrees with W18 row.')
add(material='sodium sulfide',formula='Na2S',phase='Na2S',source_id='MO12',source_kind='thermochemical, H2S-bearing solar gas',original_form="log p'_Na[bar] = 8.550 - 13889/T - 0.50[Fe/H]",verification='[V-txt] arXiv:1206.4313',notes='Not vacuum.')
w18('halite','NaCl','NaCl',alpha='0.4+-0.05',acond='853 K, (100) single crystal, 1e-5 torr (Jaeckel & Peperle 1961 via Pound 1972)',sigma='113.3',ssrc='Lee et al. 2018 Table A.2 (Janz 2013)',notes='Includes ~15-20% dimer.')
add(material='halite',formula='NaCl',phase='NaCl',source_id='JANAF-NaCl',source_kind='thermochemical (NIST-JANAF) fitted by fork B',original_form='log10 P[bar] = 7.378 - 11570/T (monomer); dimer Na2Cl2 8.117 - 12702/T',A10=13.378,B10_K=11570,T_range_K='600-1000',vapour_species='NaCl monomer (+~20% Na2Cl2 at 800 K)',
    congruency='congruent',alpha='0.4',mu_unit_amu='58.443',rho_g_cm3='2.17',verification='derived',notes='Lee+2018 alt: log10 P[bar] = 5.07184 - 8388.497/(T-82.638) (Stull 1947).')
w18('sylvite','KCl','KCl',alpha='0.63-0.72',acond='672-788 K, single crystal, 1e-7 torr (Bradley & Volans 1953 via Pound 1972)',sigma='100.3',ssrc='Lee et al. 2018 Table A.2',Tc='~800 (1 bar)',Tcc=MO12)
add(material='sylvite',formula='KCl',phase='KCl',source_id='JANAF-KCl',source_kind='thermochemical (NIST-JANAF) fitted by fork B',original_form='log10 P[bar] = 7.297 - 11157/T (monomer); K2Cl2 8.424 - 12659/T',A10=13.297,B10_K=11157,T_range_K='600-1000',vapour_species='KCl (+~18% dimer)',congruency='congruent',alpha='0.63-0.72',mu_unit_amu='74.551',rho_g_cm3='1.99',verification='derived; matches Morley eq.18 (7.611-11382/T bar) within 0.03 dex',notes='')
# ---------------- oxides ----------------
C='corundum'
add(material=C,formula='Al2O3',phase='alpha-Al2O3 single crystal',source_id='Tak15',source_kind='laboratory (vacuum)',original_form='alpha_e per crystal axis (Table 1)',original_coefficients='c/a/m: 1785C 0.055/0.081/0.181; 1675C 0.042/0.059/0.135; 1600C 0.027/0.050/0.093; Ea 730/687/706 kJ/mol',T_range_K='1873-2058',
    vapour_species='Al, O dominant (+AlO, Al2O, AlO2, O2)',congruency='congruent',alpha='0.02-0.2 (alpha_m >> alpha_a > alpha_c), increasing with T',alpha_conditions='vacuum; condensation coeff. at 1848 K, S~4: 0.04-0.06 (c), 0.06-0.08 (a), 0.1-0.2 (m)',
    verification='[ABS]+[HTML-x] ApJS 218, 2',notes='Takigawa et al. 2015 (the brief\'s "2009 ApJ 707 L97" is forsterite).')
w18(C,'Al2O3','corundum',alpha='0.02-0.2',acond='Tak15',sigma='690 (Nozawa) / 2040 calorimetric (McHale et al. 1997)',ssrc='N03; Householder poster',Tc='1677 (appearance), 1653 (50% Al)',Tcc=L03,notes='RECOMMENDED for Al2O3.')
w18(C,'Al2O3','corundum (SUPCRTBL)',src='D4',rho='3.97')
noz(C,'Al2O3','corundum',rho='3.97')
add(material=C,formula='Al2O3',phase='corundum',source_id='vL14a',source_kind='compilation (Lihrmann 2008)',original_form='p_v = exp(-A/T + B) dyn cm^-2',original_coefficients='A=77365+-3868, B=39.3+-2.0',A10=39.3/LN10,B10_K=77365/LN10,T_range_K='not given',alpha='0.1 (Schaefer & Fegley 2004)',mu_unit_amu='101.961',rho_g_cm3='4.00',
    verification='[V-txt] arXiv:1410.3494',notes='[doubtful] ~2.4 dex above JANAF-based rows at 1800 K; Lihrmann 2008 concerns the Al-C-O system - possibly not vacuum.')
w18('spinel','MgAl2O4','spinel',sigma='',Tc='1397 (Mg appearance as spinel)',Tcc=L03,notes='No lab alpha found.')
w18('spinel','MgAl2O4','spinel (SUPCRTBL)',src='D4',rho='3.58')
w18('perovskite','CaTiO3','perovskite',Tc='1593 (appearance), 1582 (50% Ti)',Tcc=L03,notes='Woitke fit 1 (Sharp & Huebner 1990). No lab alpha.')
w18('rutile','TiO2','rutile',sigma='480.6 (Lee+2015) / 618 (Jeong+2000) / 2060+-100 calorimetric',ssrc='Lee et al. 2015 A&A 575, A11; Householder poster',notes='Vapour dominated by TiO2(g) molecule.')
w18('rutile','TiO2','rutile (SUPCRTBL)',src='D4',rho='')
add(material='rutile',formula='TiO2',phase='rutile',source_id='Lee18',source_kind='thermochemical (Woitke & Helling 2004)',original_form='ln P[dyn] = 35.8027 - 74734.7/T',A10=35.8027/LN10,B10_K=74734.7/LN10,vapour_species='TiO2(g)',mu_unit_amu='79.866',rho_g_cm3='4.23',sigma_erg_cm2='480.6; sigma(T)=535.124-0.04396T (500-2000 K)',sigma_source='Lee et al. 2015',verification='[P-txt] arXiv:1801.08482 Table A.1',notes='a0=1.956 A.')
w18('periclase','MgO','periclase',alpha='0.12-0.22',acond='vacuum 1873-2173 K (Hashimoto 1989); G&S99 adopt 0.2',sigma='1100',ssrc='N03',notes='Decomposes MgO -> Mg + O/O2 with activation barrier (Hashimoto 1989).')
w18('periclase','MgO','periclase (SUPCRTBL)',src='D4',rho='3.58')
noz('periclase','MgO','periclase',alpha='0.12-0.22',rho='3.58')
w18('wustite','FeO','FeO',sigma='580',ssrc='N03',notes='Real FeO is non-stoichiometric (Fe1-xO), metastable < 843 K. No lab alpha.')
w18('wustite','FeO','FeO (SUPCRTBL)',src='D4',rho='')
noz('wustite','FeO','FeO',rho='5.99')
w18('magnetite','Fe3O4','magnetite (SUPCRTBL)',src='D4',rho='',cong='doubtful: Fe3O4 tends to lose O2 -> FeO/Fe first',notes='No lab data. Lodders 2003: magnetite forms only at ~371 K (oxidation of metal).')
noz('magnetite','Fe3O4','magnetite')
w18('hematite','Fe2O3','hematite (SUPCRTBL)',src='D4',rho='',cong='doubtful: Fe2O3 -> Fe3O4 + O2 (1659 K in air; lower in vacuum)',notes='Decomposition likely precedes sublimation; no lab data.')
# ---------------- carbides ----------------
w18('silicon carbide','SiC','alpha-SiC',cong='INCONGRUENT in lab: Si lost, C/graphene residue (Yazdi et al. 2016)',sigma='1800',ssrc='N03 (Kozasa et al. 1996)')
noz('silicon carbide','SiC','SiC',cong='congruent assumed (actually incongruent)',rho='3.21')
add(material='silicon carbide',formula='SiC',phase='SiC',source_id='vL14a',source_kind='compilation (Lilov 1993; SiC2 vapour)',original_form='p_v = exp(-A/T + B) dyn cm^-2',original_coefficients='A=78462+-3923, B=37.8+-1.9',A10=37.8/LN10,B10_K=78462/LN10,T_range_K='1500-2000',vapour_species='SiC2',alpha='0.1 (adopted, arbitrary)',
    mu_unit_amu='40.10',rho_g_cm3='3.22',verification='[V-txt] arXiv:1410.3494',notes='In H2-H2O gas SiC oxidises actively (Yamamoto et al. 2023 LPSC #1129).')
w18('titanium carbide','TiC','TiC',sigma='1242',ssrc='N03 (Rhee 1970)',notes='No lab alpha.')
noz('titanium carbide','TiC','TiC',rho='4.93')
# ---------------- ice ----------------
I='water ice'
add(material=I,formula='H2O',phase='hexagonal ice',source_id='MK05',source_kind='laboratory compilation',original_form='p[Pa] = exp(9.550426 - 5723.265/T + 3.53068 ln T - 0.00728332 T) (eq.7)',original_coefficients='as given',T_range_K='>110',vapour_species='H2O',congruency='congruent',
    alpha='alpha(T) = 0.854/(1+exp[-c1(1/T - c2)]) + 0.146, c1=11.58e3 K, c2=4.99e-3 K^-1',alpha_conditions='vacuum (Gundlach et al. 2011 eq.20); ~1 below 194 K; Pound 1972: 0.9+-0.1 (188-213 K)',mu_unit_amu='18.015',rho_g_cm3='0.93',
    sigma_erg_cm2='109',sigma_source='Lee et al. 2018 Table A.2 (Ketcham & Hobbs 1969)',Tcond_K='182',Tcond_condition=L03+' (text)',verification='[V-txt] Murphy & Koop 2005 QJRMS 131, 1539',notes='Latent heat L(T) = 46782.5 + 35.8925T - 0.07414T^2 + 541.5 exp[-(T/123.75)^2] J/mol.')
add(material=I,formula='H2O',phase='ice',source_id='FS84/G11',source_kind='laboratory (Fanale & Salvail 1984, via Gundlach et al. 2011 Table 1)',original_form='p = a1 exp(-a2/T) Pa',original_coefficients='a1=3.56e12 Pa, a2=6141.667 K',A10=13.551,B10_K=2667.3,vapour_species='H2O',alpha='see MK05 row',mu_unit_amu='18.015',rho_g_cm3='0.93',verification='[V-img] arXiv:1101.2518',notes='Gundlach own fit (151.7-194.1 K): a1=3.23e12 Pa, a2=6134.6 K.')
add(material=I,formula='H2O',phase='ice',source_id='K11-ice',source_kind='compilation (Washburn 1928)',original_form='P_v = P0 exp(-mu m_u H/kT)',original_coefficients='P0=3.08e13 dyn cm^-2, H=2.83e10 erg/g',A10=math.log10(3.08e13),B10_K=18*mu*2.83e10/kB/LN10,mu_unit_amu='18',rho_g_cm3='1.0',verification='[V-txt] arXiv:1104.5627',notes='')
add(material=I,formula='H2O',phase='ice',source_id='W18-fit7',source_kind='compilation (Ackerman & Marley 2001 via Woitke et al. 2018 Table D.2)',original_form='p[dyn] = c0 exp[(c1 Tc + Tc^2/c2)/(Tc + c3)], Tc = T-273.15',original_coefficients='c0=6111.5, c1=23.036, c2=-333.7, c3=279.82',
    A10=13.300,B10_K=2623,T_range_K='110-200 (linearised here; max dev 0.044 dex)',vapour_species='H2O',mu_unit_amu='18.015',rho_g_cm3='0.93',verification='[V-txt] arXiv:1712.01010; linearisation derived',notes='')
for r in ROWS:
    if r['material']=='water ice' and r['P0_cm2s_at_1500K']: r['notes']=(r['notes']+' P0 column is at 1500 K; multiply by sqrt(10) for 150 K.').strip()
write(OUT); print(len(ROWS),'rows')
