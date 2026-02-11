def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

def sigmoid_function(k,x0,x):
    import numpy as np
    x = x / x0
    x0 = 1.
    return 1. / (1. + np.exp(-k*(x-x0)))