"""Free (Langmuir) congruent evaporation into vacuum from Woitke et al. 2018 (A&A 614, A1) data.
Solid equilibrium from Table D.2 (fits 1,2,3,4,5,10) or Table D.4 (b-fit = -dGf/RT from atoms, 1 bar).
Gas species: atoms + all neutral fit-4 molecules of Table D.1 made only of the solid's elements.
Assumes same alpha for all vapour species, stoichiometric (congruent) fluxes, no ambient gas."""
import re, json, sys, math
import numpy as np
from scipy.optimize import fsolve
D = '/private/tmp/claude-502/-Users-currodri-Library-CloudStorage-OneDrive-TheUniversityofChicago-HotDust-Quasar/9b0de9ae-12ba-4775-919a-eb50ad854d56/scratchpad/'
L = open(D+'dl/1712.01010.txt').read().split('\n')
MOL = json.load(open(D+'work/w18_mol.json'))
AMU = {'H':1.008,'C':12.011,'N':14.007,'O':15.999,'F':18.998,'NA':22.990,'MG':24.305,'AL':26.982,'SI':28.085,'P':30.974,'S':32.06,'CL':35.45,'K':39.098,'CA':40.078,'TI':47.867,'V':50.942,'CR':51.996,'MN':54.938,'FE':55.845,'NI':58.693}
kB=1.380649e-16; mu=1.66053907e-24; R=8.314462618
def parse_formula(f):
    return {e.upper():int(n) if n else 1 for e,n in re.findall(r'([A-Z][a-z]?)(\d*)', f)}
# parse D.2
D2 = {}
for l in L[2570:2715]:
    m = re.match(r'^\s*(\d+)\s+(\S+)\[(s|l|s/l)\]\s+(.*?)\s+(\d\.\d\d)\s+(\d+)\s+(?:([<>]\s*\d+K)\s+)?(\S+)\s+(\S+)(?:\s+(\S+))?(?:\s+(\S+))?(?:\s+(\S+))?', l)
    if not m: continue
    if m.group(3)!='s': continue
    cs=[float(x) for x in m.group(8,9,10,11,12) if x and not x.startswith('±')]
    D2[m.group(2)] = dict(fit=int(m.group(6)), c=cs+[0.0]*(5-len(cs)), rho=float(m.group(5)), name=m.group(4).strip(), restr=m.group(7))
D4 = {}
for l in L[2755:2935]:
    m = re.match(r'^\s*(\d+)\s+([A-Z][A-Za-z0-9]*)\s+(\S+)\s+(-?\d+\.\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', l)
    if m: D4[m.group(2)] = dict(c=[float(x) for x in m.group(5,6,7,8,9)], name=m.group(3))
def lnkp(name, T):  # cgs
    a=MOL[name]['a']; n=sum(MOL[name]['comp'].values())
    return (1-n)*math.log(1e6) + a[0]/T + a[1]*math.log(T) + a[2] + a[3]*T + a[4]*T*T
def lnK_atoms_bar(formula, T, src):
    """ln prod (p_e/1bar)^n_e at solid-vapour equilibrium."""
    comp = parse_formula(formula); N=sum(comp.values())
    if src=='D4':
        b=D4[formula]['c']; return -(b[0]/T + b[1]*math.log(T) + b[2] + b[3]*T + b[4]*T*T)
    d=D2[formula]; c=d['c']; f=d['fit']
    if f==2: return (c[0]/T + c[1] + c[2]*T + c[3]*T**2 + c[4]*T**3)/(R*T)
    if f==5: return -(c[0]/T + c[1]*math.log(T) + c[2] + c[3]*T + c[4]*T*T)
    if f==1: return (c[0]/T + c[1] + c[2]*T + c[3]*T**2 + c[4]*T**3)*4.184/(R*T) + N*math.log(1.01325)
    if f in (3,4,10):
        if f==3: lnp = c[0]/T + c[1] + c[2]*T + c[3]*T**2 + c[4]*T**3
        if f==4: lnp = c[0] + c[1]/(T+c[2])
        if f==10: lnp = c[0]/T + c[1]
        mname = formula.upper()
        lnprod_cgs = lnp - (lnkp(mname,T) if N>1 else 0.0)
        return lnprod_cgs - N*math.log(1e6)
    raise ValueError(f)
def species(comp):
    els=set(comp); sp={e:{e:1} for e in els}
    for k,v in MOL.items():
        if set(v['comp'])<=els: sp[k]=v['comp']
    return sp
def solve(formula, T, src='D2', guess=None):
    comp=parse_formula(formula); els=sorted(comp); N=sum(comp.values())
    sp=species(comp); lnK=lnK_atoms_bar(formula,T,src)+N*math.log(1e6)  # cgs
    lkp={s:(0.0 if s in comp else lnkp(s,T)) for s in sp}
    msp={s:sum(AMU[e]*n for e,n in c.items()) for s,c in sp.items()}
    from scipy.special import logsumexp
    names=list(sp)
    def lnp(x):
        lp=dict(zip(els,x)); return {s:lkp[s]+sum(n*lp[e] for e,n in sp[s].items()) for s in names}
    def pres(x):
        return {s:math.exp(min(v,700)) for s,v in lnp(x).items()}
    def lnfl(x,e):
        l=lnp(x); terms=[l[s]-0.5*math.log(msp[s])+math.log(sp[s][e]) for s in names if sp[s].get(e,0)>0]
        return logsumexp(terms)-math.log(comp[e])
    def F(x):
        r=[sum(comp[e]*x[i] for i,e in enumerate(els))-lnK]
        r+= [lnfl(x,e)-lnfl(x,els[0]) for e in els[1:]]
        return r
    x0 = guess if guess is not None else [lnK/N]*len(els)
    x,info,ier,msg=fsolve(F,x0,full_output=True,xtol=1e-12)
    p=pres(x); e0=els[0]
    Je0=sum(sp[s].get(e0,0)*p[s]/math.sqrt(2*math.pi*msp[s]*mu*kB*T) for s in sp)
    Jfu=Je0/comp[e0]; mfu=sum(AMU[e]*n for e,n in comp.items())
    Peff=Jfu*math.sqrt(2*math.pi*mfu*mu*kB*T)
    resid=max(abs(r) for r in F(x))
    return dict(x=list(x),p=p,Jfu=Jfu,Peff=Peff,mfu=mfu,resid=resid)
def fit(formula, T1, T2, src='D2', n=41):
    Ts=np.linspace(T1,T2,n); g=None; lp=[]; res=[]
    for T in Ts:
        r=solve(formula,T,src,g); g=r['x']; lp.append(math.log10(r['Peff'])); res.append(r['resid'])
    Acoef=np.polyfit(1/Ts,lp,1)  # lp = slope/T + icpt
    B=-Acoef[0]; A=Acoef[1]; dev=np.max(np.abs(np.polyval(Acoef,1/Ts)-lp))
    rmid=solve(formula,0.5*(T1+T2),src)
    tot=sum(rmid['p'].values()); top=sorted(rmid['p'].items(),key=lambda kv:-kv[1])[:5]
    return dict(A=A,B=B,maxdev_dex=dev,maxresid=max(res),mfu=rmid['mfu'],vap=[(k,round(v/tot,3)) for k,v in top],Tmid=0.5*(T1+T2))
if __name__=='__main__':
    print(sorted(D2.keys())); print(len(D4))
