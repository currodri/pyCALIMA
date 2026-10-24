import math, json, csv
LN10=math.log(10); kB=1.380649e-16; mu=1.66053907e-24
TREF=1500.0
COLS=['material','formula','phase','source_id','source_kind','original_form','original_coefficients','A10','B10_K','T_range_K',
 'vapour_species','congruency','alpha','alpha_conditions','mu_unit_amu','rho_g_cm3','Bk_K','P0_cm2s_at_1500K','nu0_s_at_1500K_alpha1',
 'sigma_erg_cm2','sigma_source','Tcond_K','Tcond_condition','verification','notes']
ROWS=[]
def add(P0x=None, **kw):
    r={c:'' for c in COLS}; r.update(kw)
    if P0x is not None:
        r['P0_cm2s_at_1500K']=f'{P0x:.3e}'
        if r['rho_g_cm3']!='' and r['mu_unit_amu']!='':
            r['nu0_s_at_1500K_alpha1']=f'{(float(r["mu_unit_amu"])*mu/float(r["rho_g_cm3"]))**(2/3)*P0x:.3e}'
    if r['A10']!='' and r['B10_K']!='':
        A=float(r['A10']); B=float(r['B10_K']); r['A10']=f'{A:.3f}'; r['B10_K']=f'{B:.0f}'
        r['Bk_K']=f'{B*LN10:.0f}'
        if r['mu_unit_amu']!='' and P0x is None:
            m=float(r['mu_unit_amu']); P0=10**A/math.sqrt(2*math.pi*m*mu*kB*TREF); r['P0_cm2s_at_1500K']=f'{P0:.3e}'
            if r['rho_g_cm3']!='':
                rho=float(r['rho_g_cm3']); r['nu0_s_at_1500K_alpha1']=f'{(m*mu/rho)**(2/3)*P0:.3e}'
    ROWS.append(r)
def write(path):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in ROWS]
