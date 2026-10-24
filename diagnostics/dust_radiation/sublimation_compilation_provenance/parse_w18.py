import re, json, sys
src = sys.argv[1]; out = sys.argv[2]
L = open(src).read().split('\n')
ELS = ['HE','LI','NA','MG','AL','SI','CL','CA','TI','CR','MN','FE','NI','ZR','H','C','N','O','F','P','S','K','V','W']
def parse(name):
    s = name
    def rec(s, i):
        comp = {}
        while i < len(s):
            if s[i] == '(':
                sub, i = rec(s, i+1)
                m = re.match(r'\d+', s[i:]); k = int(m.group()) if m else 1
                i += len(m.group()) if m else 0
                for e,v in sub.items(): comp[e] = comp.get(e,0)+v*k
            elif s[i] == ')':
                return comp, i+1
            else:
                for e in ELS:
                    if s.startswith(e, i):
                        i += len(e); m = re.match(r'\d+', s[i:]); k = int(m.group()) if m else 1
                        i += len(m.group()) if m else 0
                        comp[e] = comp.get(e,0)+k; break
                else:
                    raise ValueError(name)
        return comp, i
    c, i = rec(s, 0)
    return c
mol = {}
for l in L[1960:2570]:
    m = re.match(r'^\s*(\d+)\s+(\S+)\s+\((\d)\)\s+(\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', l)
    if not m: continue
    name, fit = m.group(2), int(m.group(4))
    if fit != 4 or '+' in name or name.endswith('-'): continue
    try: comp = parse(name)
    except ValueError: continue
    mol[name] = dict(comp=comp, a=[float(m.group(k)) for k in range(5,10)])
json.dump(mol, open(out,'w'), indent=0)
print(len(mol), 'neutral fit-4 molecules')
