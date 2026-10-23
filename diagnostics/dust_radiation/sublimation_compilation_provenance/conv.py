import math, json
from freeevap import AMU, parse_formula
LN10=math.log(10); kB=1.380649e-16; mu=1.66053907e-24
def m_of(f): return sum(AMU[e]*n for e,n in parse_formula(f).items())
# Nozawa+2003 Table 2: (solid, products {species:n per formula unit}, nu_key per f.u., A[K], B, sigma, a0[A])
NOZ=[('Fe',{'Fe':1},1,48418.0,16.5566,1800,1.411),('FeS',{'Fe':1,'S':1},1,93132.6,30.7771,380,1.932),
('Si',{'Si':1},1,53697.5,17.4349,800,1.684),('Ti',{'Ti':1},1,55890.2,16.6071,1510,1.615),('V',{'V':1},1,61539.4,17.8702,1697,1.490),
('Cr',{'Cr':1},1,46773.3,16.7596,1880,1.421),('Co',{'Co':1},1,50388.0,16.8372,1936,1.383),('Ni',{'Ni':1},1,50913.0,17.1559,1924,1.377),
('Cu',{'Cu':1},1,39795.5,14.9083,1300,1.412),('C',{'C':1},1,86472.6,19.0422,1400,1.281),('SiC',{'Si':1,'C':1},1,148934.,37.3825,1800,1.702),
('TiC',{'Ti':1,'C':1},1,164696.,37.2301,1242,1.689),('Al2O3',{'Al':2,'O':3},2,184788.,45.3543,690,1.718),
('MgSiO3',{'Mg':1,'SiO':1,'O':2},1,250129.,72.0015,400,2.319),('Mg2SiO4',{'Mg':2,'SiO':1,'O':3},2,186200.,52.4336,436,2.055),
('SiO2',{'SiO':1,'O':1},1,126028.,38.1507,605,2.080),('MgO',{'Mg':1,'O':1},1,119237.,33.1593,1100,1.646),
('Fe3O4',{'Fe':3,'O':4},3,132889.,39.1687,400,1.805),('FeO',{'Fe':1,'O':1},1,111290.,31.9850,580,1.682)]
AMU.update({'CO':58.933,'CU':63.546})
def mspec(s): return m_of(s)
noz=[]
for f,prod,nu,A,B,sig,a0 in NOZ:
    N=sum(prod.values()); mfu=sum(n*mspec(s) for s,n in prod.items())
    logC=0.5*math.log10(mfu)-sum((n/N)*math.log10(n*math.sqrt(mspec(s))) for s,n in prod.items())
    A10=nu*B/(N*LN10)+6+logC; B10=nu*A/(N*LN10)
    noz.append(dict(solid=f,products=prod,N=N,nu=nu,A=A,B=B,sigma=sig,a0=a0,mfu=mfu,A10=A10,B10=B10))
    print(f"NOZ {f:8s} N={N} mfu={mfu:7.2f} log10Peff[dyn]={A10:7.3f} - {B10:7.0f}/T   sigma={sig}")
json.dump(noz,open('nozawa_conv.json','w'),indent=1)
# van Lieshout+2014 Table 3: pv = exp(-A/T + B) dyn cm^-2
VL=[('Fe',55.845,1.0,48354,29.2,'1573-1973','F04'),('SiO',44.085,0.04,49520,32.5,'1275-1525','G13'),('Fe2SiO4',203.774,0.1,60377,37.7,'1373-1433','N94'),
('MgSiO3',100.389,0.1,68908,38.1,'1573-1923','M88,K91'),('Mg2SiO4',140.694,0.1,65308,34.1,'1673-2133','N94'),('SiO2',60.084,1.0,69444,33.1,'1833-1958','H90'),
('Al2O3',101.961,0.1,77365,39.3,'','L08'),('SiC',40.10,0.1,78462,37.8,'1500-2000','L93'),('C',12.011,0.1,93646,36.7,'2400-3000','Z73')]
for f,m,al,A,B,Tr,ref in VL:
    print(f"VL  {f:8s} log10P[dyn]={B/LN10:7.3f} - {A/LN10:7.0f}/T  alpha={al} mu={m} T={Tr}")
# Kobayashi+2011 Table 1: Pv = P0 exp(-mu m_u H/kT)
KO=[('olivine',3.3,169.1,3.21e10,6.72e14),('pyroxene',3.3,60.1,9.60e10,3.12e11),('obsidian',2.37,67.0,7.12e10,1.07e14),('carbon',1.95,12.0,7.27e11,4.31e16),('iron',7.86,55.8,2.97e10,5.00e4),('pure ice',1.0,18,2.83e10,3.08e13),('dirty ice',1.4,18,2.83e10,2.67e13)]
for n,rho,m,H,P0 in KO:
    th=m*mu*H/kB
    print(f"KOB {n:9s} log10P[dyn]={math.log10(P0):7.3f} - {th/LN10:7.0f}/T   (H*mu*m_u/k = {th:.0f} K)")
