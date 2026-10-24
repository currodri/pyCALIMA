from freeevap import *
import json
cases = [ # formula, src, T1, T2
 ('C','D2',1500,2500),('SiC','D2',1300,2300),('TiC','D2',1300,2300),('Fe','D2',1000,1800),('Cr','D2',1000,2000),('Ni','D2',1000,1700),
 ('FeS','D2',800,1400),('MnS','D2',1000,1800),('Na2S','D2',700,1300),('NaCl','D2',500,1000),('KCl','D2',500,1000),
 ('Al2O3','D2',1300,2300),('MgAl2O4','D2',1300,2300),('CaTiO3','D2',1300,2200),('TiO2','D2',1200,2100),
 ('Mg2SiO4','D2',1200,2100),('MgSiO3','D2',1100,1800),('Fe2SiO4','D2',1000,1450),('FeSiO3','D4',1000,1400),
 ('SiO2','D2',1100,1690),('SiO','D2',900,1500),('MgO','D2',1300,2300),('FeO','D2',1000,1600),('Fe2O3','D4',900,1500),('Fe3O4','D4',900,1500),
 ('Mg2SiO4','D4',1200,2100),('MgSiO3','D4',1100,1800),('Al2O3','D4',1300,2300),('MgAl2O4','D4',1300,2300),('TiO2','D4',1200,2100),('MgO','D4',1300,2300),('SiO2','D4',1100,1690),('FeS','D4',800,1400),('Fe','D4',1000,1800),('FeO','D4',1000,1600)]
out=[]
for f,s,T1,T2 in cases:
    r=fit(f,T1,T2,s); r.update(formula=f,src=s,T1=T1,T2=T2); out.append(r)
    print(f"{f:8s} {s} {T1}-{T2}K  log10Peff[dyn]= {r['A']:.3f} - {r['B']:.0f}/T  dev={r['maxdev_dex']:.3f}dex resid={r['maxresid']:.1e} m={r['mfu']:.2f} vap@{r['Tmid']:.0f}K {r['vap']}")
json.dump(out,open('freeevap_results.json','w'),indent=1)
