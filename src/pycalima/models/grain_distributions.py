"""
GRAIN SIZE DISTRIBUTIONS

This module contains the classes that help define different grain size distributions.
"""
# LIBRARIES
import numpy as np

from models.grain_size_config import build_lognormal_distribution, get_bin_by_rank


def _build_distribution_for(composition, bin_rank=0, is_pah=False):
    meta = get_bin_by_rank(composition=composition, bin_rank=bin_rank, is_pah=is_pah)
    return build_lognormal_distribution(meta["id"])

# CLASSES
class LogNormal_Distribution(object):

    def __init__(self,a0,amin,amax,sigma,grain_density):
        self.Nc = None
        self.a0 = a0
        self.amin = amin
        self.amax = amax
        self.sigma = sigma
        self.a = np.logspace(np.log(amin),np.log10(amax),1000)
        self.grain_density = grain_density
        self.sintegral = self._init_integral()
        self.grain_mass = 4./3. * np.pi * grain_density * a0**3.

    def _init_integral(self):
        y = (1.0/self.a) * np.exp(-(np.log(self.a/self.a0))**2/(2*self.sigma**2))
        return (4*np.pi*self.grain_density/3)*np.trapezoid(y,self.a)

    def n_density(self,mass_density,sizes):
        C = mass_density/self.sintegral
        dist = (C/sizes**4)*np.exp(-(np.log(sizes/self.a0))**2/(2*self.sigma**2))
        dist[sizes<self.amin] = 0.0
        dist[sizes>self.amax] = 0.0
        return dist
    
    def averaged_over(self,X,sizes):
        y = (1.0/(sizes**4)) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        N = np.trapezoid(y,sizes)
        
        x = (X/(sizes**4)) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        x[sizes<self.amin] = 0.0
        x[sizes>self.amax] = 0.0
        
        avg = (1/N) * np.trapezoid(x,sizes)
        
        return avg
    def averaged_over_mass(self,X,sizes):
        y = (1.0/sizes) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        N = np.trapezoid(y,sizes)
        
        x = (X/sizes) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        x[sizes<self.amin] = 0.0
        x[sizes>self.amax] = 0.0
        
        avg = (1/N) * np.trapezoid(x,sizes)
        
        return avg
    def averaged_over_column(self,X,sizes):
        y = (1.0/(sizes**2)) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        N = np.trapezoid(y,sizes)
        
        x = (X/(sizes**2)) * np.exp(-(np.log10(sizes/self.a0))**2/(2*self.sigma**2))
        x[sizes<self.amin] = 0.0
        x[sizes>self.amax] = 0.0
        
        avg = (1/N) * np.trapezoid(x,sizes)
        
        return avg
    def averaged_over_number(self,X,sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        norm = (1./sizes[mask]**4.) * np.exp(-(np.log10(sizes[mask]/self.a0))**2/(2*self.sigma**2))
        norm = np.trapezoid(norm,sizes[mask])
        y = (X[mask]/sizes[mask]**4.) * np.exp(-(np.log10(sizes[mask]/self.a0))**2/(2*self.sigma**2))
        y = np.trapezoid(y,sizes[mask])
        avg = y / norm     
        return avg

class Classical_LogNormal_Distribution(object):

    def __init__(self,a0,amin,amax,sigma,grain_density):
        self.Nc = None
        self.a0 = a0
        self.amin = amin
        self.amax = amax
        self.sigma = sigma
        self.a = np.logspace(np.log(amin),np.log10(amax),1000)
        self.grain_density = grain_density
        self.sintegral = self._init_integral()
        self.grain_mass = 4./3. * np.pi * grain_density * a0**3.

    def _init_integral(self):
        y = self.a**3. * np.exp(-(np.log(self.a/self.a0))**2/(2*self.sigma**2))
        return (4*np.pi*self.grain_density/3)*np.trapezoid(y,self.a)

    def n_density(self,mass_density,sizes):
        C = mass_density/self.sintegral
        dist = C*np.exp(-(np.log(sizes/self.a0))**2/(2*self.sigma**2))
        dist[sizes<self.amin] = 0.0
        dist[sizes>self.amax] = 0.0
        return dist

    def averaged_over_number(self,X,sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        norm = np.exp(-(np.log10(sizes[mask]/self.a0))**2/(2*self.sigma**2))
        norm = np.trapezoid(norm,sizes[mask])
        y = X[mask]* np.exp(-(np.log10(sizes[mask]/self.a0))**2/(2*self.sigma**2))
        y = np.trapezoid(y,sizes[mask])
        avg = y / norm     
        return avg

class Flat_Distribution(object):
    """Uniform (flat) grain size distribution: dn/da = const within [amin, amax].

    Accepts the same 5-parameter signature as LogNormal_Distribution so it can
    be passed as a drop-in distribution_class override; a0 and sigma are ignored.
    """

    def __init__(self, a0, amin, amax, sigma, grain_density):
        self.amin = amin
        self.amax = amax
        self.grain_density = grain_density
        self.sintegral = self._init_integral()
        a_mid = np.sqrt(amin * amax)
        self.grain_mass = 4./3. * np.pi * grain_density * a_mid**3.

    def _init_integral(self):
        # ∫ m(a) * (dn/da) * da for dn/da = const, m(a) = (4π/3)*ρ*a³
        return (4.*np.pi * self.grain_density / 3.) * (self.amax**4 - self.amin**4) / 4.

    def n_density(self, mass_density, sizes):
        C = mass_density / self.sintegral
        return np.where((sizes >= self.amin) & (sizes <= self.amax), C, 0.0)

    def averaged_over(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        return np.trapezoid(X[mask], sizes[mask]) / (self.amax - self.amin)

    def averaged_over_mass(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**3
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_column(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**2
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_number(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        return np.trapezoid(X[mask], sizes[mask]) / (self.amax - self.amin)


class PowerLaw_Distribution(object):
    """Pure power-law grain size distribution: dn/da ∝ a^{-alpha} within [amin, amax].

    Uses the same 5-parameter signature as LogNormal_Distribution so it can
    be passed as a drop-in distribution_class override.  a0 is ignored;
    sigma is reinterpreted as the power-law index alpha (dn/da ∝ a^{-alpha}).
    """

    def __init__(self, a0, amin, amax, powlaw_index, grain_density):
        self.amin = amin
        self.amax = amax
        self.powlaw_index = powlaw_index
        self.grain_density = grain_density
        self.sintegral = self._init_integral()
        # representative grain mass at geometric-mean size
        a_mid = np.sqrt(amin * amax)
        self.grain_mass = 4. / 3. * np.pi * grain_density * a_mid**3.

    def _init_integral(self):
        # sintegral = ∫ m(a) * dn/da * da  with  dn/da = C * a^(-alpha)
        # = C * (4π/3)*ρ * ∫ a^(3-alpha) da
        alpha = self.powlaw_index
        prefac = 4. * np.pi * self.grain_density / 3.
        if abs(alpha - 4.) < 1e-10:
            return prefac * np.log(self.amax / self.amin)
        exp = 4. - alpha
        # For very steep distributions (alpha >> 4) the integral is dominated
        # by amin; evaluate each term separately to avoid overflow.
        try:
            val = (self.amax**exp - self.amin**exp) / exp
        except OverflowError:
            # amin term dominates when alpha >> 4 (exp << 0)
            val = (-self.amin**exp) / exp
        return prefac * val

    def n_density(self, mass_density, sizes):
        C = mass_density / self.sintegral
        dist = C * sizes**(-self.powlaw_index)
        return np.where((sizes >= self.amin) & (sizes <= self.amax), dist, 0.0)

    def averaged_over(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**(-self.powlaw_index)
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_mass(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**(3. - self.powlaw_index)
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_column(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**(2. - self.powlaw_index)
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_number(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**(-self.powlaw_index)
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])


class PowerLaw_DualCutoff_Distribution(object):
    """Power law with exponential cutoffs at both ends.

    dn/da = C · a^{-alpha} · exp(-amin/a) · exp(-a/amax)

    Uses the same 5-parameter signature as LogNormal_Distribution so it can
    be passed as a drop-in distribution_class override; a0 is ignored and
    sigma is reinterpreted as the power-law index alpha.
    """

    def __init__(self, a0, amin, amax, powlaw_index, grain_density):
        self.amin = amin
        self.amax = amax
        self.powlaw_index = powlaw_index
        self.grain_density = grain_density
        self._a_grid = np.logspace(np.log10(amin), np.log10(amax), 600)
        self.sintegral = self._init_integral()
        self.grain_mass = 4. / 3. * np.pi * grain_density * np.sqrt(amin * amax)**3.

    def _shape(self, a):
        return a**(-self.powlaw_index) * np.exp(-self.amin / a) * np.exp(-a / self.amax)

    def _init_integral(self):
        # ∫ m(a) · dn/da · da = C · (4π/3)·ρ · ∫ a³ · shape(a) da
        prefac = 4. * np.pi * self.grain_density / 3.
        return prefac * np.trapezoid(self._a_grid**3 * self._shape(self._a_grid),
                                     self._a_grid)

    def n_density(self, mass_density, sizes):
        C = mass_density / self.sintegral
        return C * self._shape(sizes)

    def averaged_over(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_mass(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**3 * self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_column(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**2 * self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_number(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])


class PowerLaw_LowerCutoff_Distribution(object):
    """Power law with an exponential cutoff only at the lower end.

    dn/da = C · a^{-alpha} · exp(-amin/a)   for a <= amax, else 0

    Use for the first (smallest-a) bin of a composition: the distribution
    tapers smoothly to zero as a → 0, while the upper boundary is a hard
    truncation at amax. a0 is ignored; sigma is the power-law index alpha.
    """

    def __init__(self, a0, amin, amax, powlaw_index, grain_density):
        self.amin = amin
        self.amax = amax
        self.powlaw_index = powlaw_index
        self.grain_density = grain_density
        # Integrate from amin/100 so the lower exponential tail is captured
        self._a_grid = np.logspace(np.log10(amin * 0.01), np.log10(amax), 800)
        self.sintegral = self._init_integral()
        self.grain_mass = 4. / 3. * np.pi * grain_density * np.sqrt(amin * amax)**3.

    def _shape(self, a):
        return a**(-self.powlaw_index) * np.exp(-self.amin / a)

    def _init_integral(self):
        prefac = 4. * np.pi * self.grain_density / 3.
        return prefac * np.trapezoid(self._a_grid**3 * self._shape(self._a_grid),
                                     self._a_grid)

    def n_density(self, mass_density, sizes):
        C = mass_density / self.sintegral
        result = C * self._shape(sizes)
        result[sizes > self.amax] = 0.0
        return result

    def averaged_over(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_mass(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**3 * self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_column(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**2 * self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_number(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])


class PowerLaw_UpperCutoff_Distribution(object):
    """Power law with an exponential cutoff only at the upper end.

    dn/da = C · a^{-alpha} · exp(-a/amax)   for a >= amin, else 0

    Use for the last (largest-a) bin of a composition: the distribution
    tapers smoothly to zero as a → ∞, while the lower boundary is a hard
    truncation at amin. a0 is ignored; sigma is the power-law index alpha.
    """

    def __init__(self, a0, amin, amax, powlaw_index, grain_density):
        self.amin = amin
        self.amax = amax
        self.powlaw_index = powlaw_index
        self.grain_density = grain_density
        # Integrate out to amax*20 so the upper exponential tail is captured
        self._a_grid = np.logspace(np.log10(amin), np.log10(amax * 20.), 800)
        self.sintegral = self._init_integral()
        self.grain_mass = 4. / 3. * np.pi * grain_density * np.sqrt(amin * amax)**3.

    def _shape(self, a):
        return a**(-self.powlaw_index) * np.exp(-a / self.amax)

    def _init_integral(self):
        prefac = 4. * np.pi * self.grain_density / 3.
        return prefac * np.trapezoid(self._a_grid**3 * self._shape(self._a_grid),
                                     self._a_grid)

    def n_density(self, mass_density, sizes):
        C = mass_density / self.sintegral
        result = C * self._shape(sizes)
        result[sizes < self.amin] = 0.0
        return result

    def averaged_over(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_mass(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**3 * self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_column(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**2 * self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_number(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])


class Exponential_Distribution(object):
    """Pure exponential: dn/da = C * exp(-a/a_c), hard lower cutoff at amin.

    5-arg signature: a0 ignored; sigma is the exponential scale a_c in µm
    (converted to cm internally). Use for the last bin of each composition
    where the Zubko distribution is in its exponential rolloff regime.
    """

    def __init__(self, a0, amin, amax, sigma, grain_density):
        self.amin = amin            # cm
        self.amax = amax            # cm
        self.a_c  = sigma * 1e-4   # convert µm → cm
        self.grain_density = grain_density
        self._a_grid = np.logspace(np.log10(amin), np.log10(amax * 20.), 800)
        self.sintegral = self._init_integral()
        self.grain_mass = 4. / 3. * np.pi * grain_density * np.sqrt(amin * amax)**3.

    def _shape(self, a):
        return np.exp(-a / self.a_c)

    def _init_integral(self):
        prefac = 4. * np.pi * self.grain_density / 3.
        return prefac * np.trapezoid(self._a_grid**3 * self._shape(self._a_grid),
                                     self._a_grid)

    def n_density(self, mass_density, sizes):
        C = mass_density / self.sintegral
        result = C * self._shape(sizes)
        result[sizes < self.amin] = 0.0
        return result

    def averaged_over(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_mass(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**3 * self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_column(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = sizes[mask]**2 * self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])

    def averaged_over_number(self, X, sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        w = self._shape(sizes[mask])
        return np.trapezoid(X[mask] * w, sizes[mask]) / np.trapezoid(w, sizes[mask])


class PowerLaw_ExpCutoff_Distribution(object):

    def __init__(self,amin,amax,a_cutoff,powlaw_index,grain_density):
        self.Nc = None
        self.amin = amin
        self.amax = amax
        self.a_cutoff = a_cutoff
        self.powlaw_index = powlaw_index
        self.a = np.logspace(np.log10(amin),np.log10(amax),1000)
        self.grain_density = grain_density
        self.sintegral = self._init_integral()
        self.grain_mass = 4./3. * np.pi * grain_density * a_cutoff**3.

    def _init_integral(self):
        y = (self.a)**(3.-self.powlaw_index) * np.exp(-self.a/self.a_cutoff)
        return (4*np.pi*self.grain_density/3)*np.trapezoid(y,self.a)

    def n_density(self,mass_density,sizes):
        C = mass_density/self.sintegral
        dist = C * (sizes)**(-self.powlaw_index) * np.exp(-sizes/self.a_cutoff)
        dist[sizes<self.amin] = 0.0
        dist[sizes>self.amax] = 0.0
        return dist
    
    def averaged_over_number(self,X,sizes):
        mask = (sizes >= self.amin) & (sizes <= self.amax)
        norm = (sizes[mask]**(-self.powlaw_index)) * np.exp(-sizes[mask]/self.a_cutoff)
        norm = np.trapezoid(norm,sizes[mask])
        y = (X[mask]/sizes[mask]**(-self.powlaw_index)) * np.exp(-sizes[mask]/self.a_cutoff)
        y = np.trapezoid(y,sizes[mask])
        avg = y / norm
        return avg 
    

def plot_distribution(rho_gas,D_smallPAHs,D_largePAHs,D_smallC,D_largeC,D_smallSil,D_largeSil):
    """Create figure for the plotting of the full dust distribution.

    Args:
        rho_gas (float): Gas density in g/cm^3
        D_smallPAHs (float): Small PAHs mass fraction
        D_largePAHs (float): Large PAHs mass fraction
        D_small (float): Small carbonaceous fraction
        D_large (float): Large carbonaceous fraction
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    fig, axes = plt.subplots(1, 2, sharex=True,sharey=True, figsize=(8,5), dpi=300, facecolor='w', edgecolor='k')

    sizes = np.logspace(np.log10(1e-4),np.log10(1),1000)
    pah_bin0 = _build_distribution_for("graphite", bin_rank=0, is_pah=True)
    pah_bin1 = _build_distribution_for("graphite", bin_rank=1, is_pah=True)
    gra_bin0 = _build_distribution_for("graphite", bin_rank=0, is_pah=False)
    gra_bin1 = _build_distribution_for("graphite", bin_rank=1, is_pah=False)

    n_pah_bin0 = pah_bin0.n_density(rho_gas*D_smallPAHs,sizes)
    n_pah_bin1 = pah_bin1.n_density(rho_gas*D_largePAHs,sizes)
    n_gra_bin0 = gra_bin0.n_density(rho_gas*D_smallC,sizes)
    n_gra_bin1 = gra_bin1.n_density(rho_gas*D_largeC,sizes)

    n_tot = n_pah_bin0 + n_pah_bin1 + n_gra_bin0 + n_gra_bin1

    n_tot = (sizes**4)*n_tot

    axes[0].plot(sizes,(sizes**4)*n_pah_bin0,'--',color='blue',label='PAH bin 0',linewidth=2.5)
    axes[0].plot(sizes,(sizes**4)*n_pah_bin1,'--',color='royalblue',label='PAH bin 1',linewidth=2.5)
    axes[0].plot(sizes,(sizes**4)*n_gra_bin0,'-.',color='steelblue',label='Graphite bin 0',linewidth=2.5)
    axes[0].plot(sizes,(sizes**4)*n_gra_bin1,':',color='cornflowerblue',label='Graphite bin 1',linewidth=2.5)
    axes[0].plot(sizes,n_tot,'k-',label='Total C',linewidth=2.5)
    axes[0].set_ylabel(r'$a^4 n(a)$', fontsize=16)
    axes[0].set_xlabel(r'$a$ [$\mu$m]',fontsize=16)
    axes[1].set_xlabel(r'$a$ [$\mu$m]',fontsize=16)
    
    sil_bin0 = _build_distribution_for("silicate", bin_rank=0, is_pah=False)
    sil_bin1 = _build_distribution_for("silicate", bin_rank=1, is_pah=False)

    n_sil_bin0 = sil_bin0.n_density(rho_gas*D_smallSil,sizes)
    n_sil_bin1 = sil_bin1.n_density(rho_gas*D_largeSil,sizes)

    n_tot = n_sil_bin0 + n_sil_bin1

    n_tot = (sizes**4)*n_tot

    axes[1].plot(sizes,(sizes**4)*n_sil_bin0,'-.',color='saddlebrown',label='Silicate bin 0',linewidth=2.5)
    axes[1].plot(sizes,(sizes**4)*n_sil_bin1,':',color='sandybrown',label='Silicate bin 1',linewidth=2.5)
    axes[1].plot(sizes,n_tot,'k--',label='Total Sil',linewidth=2.5)
    
    axes[0].set_ylim([4e-30,3e-27])
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].tick_params(labelsize=14)
    axes[0].xaxis.set_ticks_position('both')
    axes[0].yaxis.set_ticks_position('both')
    axes[0].minorticks_on()
    axes[0].tick_params(which='both',axis="both",direction="in")
    axes[0].legend(loc='best',fontsize=14,frameon=False)

    axes[0].plot(sizes,1e-27*sizes**(.5),':',color='gray',linewidth=2)
    axes[0].text(0.1, 0.25, r'MRN ($n\propto a^{-3.5}$)',
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes[0].transAxes,fontsize=16,rotation=43)
    
    axes[1].set_ylim([4e-30,3e-27])
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].tick_params(labelsize=14)
    axes[1].xaxis.set_ticks_position('both')
    axes[1].yaxis.set_ticks_position('both')
    axes[1].minorticks_on()
    axes[1].tick_params(which='both',axis="both",direction="in")
    axes[1].legend(loc='best',fontsize=14,frameon=False)

    axes[1].plot(sizes,1e-27*sizes**(.5),':',color='gray',linewidth=2)
    axes[1].text(0.1, 0.25, r'MRN ($n\propto a^{-3.5}$)',
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes[1].transAxes,fontsize=16,rotation=43)

    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.1,right=0.99,hspace=0,wspace=0)

    return fig