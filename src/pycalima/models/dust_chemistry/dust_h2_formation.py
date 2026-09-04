"""
FORMATION OF MOLECULAR HYDROGEN ON DUST GRAINS

This module provides functions to calculate the rate of molecular hydrogen
(H2) formation on dust grains in interstellar environments. The formation
rate depends on factors such as the density of atomic hydrogen, the dust grain
surface area, temperature, and efficiency of the formation process.
This is all based on the model by Cazaux & Tielens (2002, 2004).

By: C. Rodriguez Montero (currodri@gmail.com)
"""
import numpy as np
import matplotlib.pyplot as plt

kB = 1.380649e-16       # erg/K
mH = 1.6735575e-24      # g
twopi = 2*np.pi
pi = np.pi

# ---------------------------
#  Parameters (Cazaux & Spaans 2004)
# ---------------------------
EH2 = np.array([540.0, 340.0])   # K
mu  = np.array([0.4,   0.3])
Es  = np.array([250.0, 200.0])
EHp = np.array([800.0, 650.0])
EHc = np.array([3e4,   3e4])

nuH2 = np.array([3e12, 2e12])    # s^-1
nuHc = np.array([2e13, 1e13])    # s^-1

Ns = 2e15   # sites cm^-2

# -------------------------------------------------------------------------
# 1) Sticking coefficient Hollenbach & McKee (1979)
# -------------------------------------------------------------------------
def h2_sticking_coef(Tgas, Td):
    return 1.0 / (1.0 + 0.4*np.sqrt((Tgas+Td)/100.0) +
                  0.2*(Tgas/100.0) + 0.08*(Tgas/100.0)**2)


# -------------------------------------------------------------------------
# 2) Desorption rate of H2
# -------------------------------------------------------------------------
def beta_h2(Tgas, Td, dust_index):
    return nuH2[dust_index] * np.exp(-EH2[dust_index] / Td)


# -------------------------------------------------------------------------
# 3) High-temperature correction
# -------------------------------------------------------------------------
def high_temp_correction(Tgas, Td, F, dust_index):
    a1 = nuHc[dust_index] / (2.0 * F)
    a2 = np.exp(-1.5 * EHc[dust_index] / Td)
    a3 = (1.0 + np.sqrt((EHc[dust_index] - Es[dust_index]) /
                        (EHp[dust_index] - Es[dust_index])))**2
    return 1.0 / (1.0 + a1 * a2 * a3)


# -------------------------------------------------------------------------
# 4) Ratio beta_hp / alpha_pc
# -------------------------------------------------------------------------
def beta_hp_over_alphapc(Tgas, Td, dust_index):
    a1 = (1.0 + np.sqrt((EHc[dust_index] - Es[dust_index]) /
                        (EHp[dust_index] - Es[dust_index])))**2
    a2 = np.exp(-Es[dust_index] / Td)
    return 0.25 * a1 * a2


# -------------------------------------------------------------------------
# 5) H flux onto grain surface
# -------------------------------------------------------------------------
def h_flux(nH, vH):
    return nH * vH / Ns


# -------------------------------------------------------------------------
# 6) Recombination efficiency
# -------------------------------------------------------------------------
def recombination_efficiency(Tgas, Td, nH, vH, F, dust_index):
    b2 = beta_h2(Tgas, Td, dust_index)
    a1 = 1.0 / (1.0 + (mu[dust_index] * F) / (2.0 * b2)
                + beta_hp_over_alphapc(Tgas, Td, dust_index))
    a2 = high_temp_correction(Tgas, Td, F, dust_index)
    return a1 * a2


# -------------------------------------------------------------------------
# Plotting helper: recombination efficiency vs grain size
# -------------------------------------------------------------------------
def plot_recombination_efficiency_vs_size(
    nH,
    Tgas,
    asizes,
    T_dust=None,
    dust_indices=(0, 1),
    figsize=(7, 5),
    savefile=None,
    show=True,
):
    """Plot recombination efficiency as a function of grain size.

    Parameters
    - nH : float
        Atomic hydrogen number density (cm^-3).
    - Tgas : float
        Gas temperature (K).
    - asizes : array_like
        Grain sizes in microns (1e-6 m). Can be scalar or 1D array.
    - T_dust : float or array_like, optional
        Dust temperature(s) in K. If a single value is given it will be
        broadcast to all sizes. If None (default) a constant 15 K is used.
    - dust_indices : tuple, optional
        Tuple/list of dust indices to plot (0 = carbonaceous, 1 = silicate).
    - figsize : tuple, optional
        Figure size passed to `plt.figure`.
    - savefile : str, optional
        If provided, the plot is saved to this path.
    - show : bool, optional
        If True (default) calls `plt.show()`; otherwise closes the figure.

    Notes
    - The recombination efficiency itself does not explicitly depend on grain
      size in this model; any variation with size must come through the
      provided `T_dust` (or other externally computed size-dependent dust
      properties). This helper will therefore plot the efficiency for the
      provided dust temperatures (one per size) so users can supply a
      physically-motivated `T_dust(asize)` if desired.

    Example
    >>> sizes = np.logspace(-3, 0, 20)  # 0.001 - 1.0 micron
    >>> plot_recombination_efficiency_vs_size(nH=30.0, Tgas=100.0, asizes=sizes)
    """
    asizes = np.atleast_1d(asizes).astype(float)

    # Prepare dust temperatures
    if T_dust is None:
        T_dust_arr = np.full_like(asizes, 15.0)
    else:
        T_dust_arr = np.atleast_1d(T_dust).astype(float)
        if T_dust_arr.size == 1:
            T_dust_arr = np.full_like(asizes, float(T_dust_arr))
        elif T_dust_arr.size != asizes.size:
            raise ValueError("`T_dust` must be scalar or same length as `asizes`")

    vH = np.sqrt(2 * kB * Tgas / mH)
    F = h_flux(nH, vH)

    plt.figure(figsize=figsize)

    for di in dust_indices:
        eff_vals = [recombination_efficiency(Tgas, Td_j, nH, vH, F, di)
                    for Td_j in T_dust_arr]
        label = 'Carbonaceous' if di == 0 else 'Silicate' if di == 1 else f'Dust {di}'
        plt.plot(asizes, eff_vals, marker='o', label=label)

    plt.xscale('log')
    plt.xlabel('Grain size (micron)')
    plt.ylabel('Recombination efficiency')
    plt.title(f'Recombination efficiency vs grain size (nH={nH:.2e}, Tgas={Tgas} K)')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()

    if savefile:
        plt.savefig(savefile, dpi=200, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


# -------------------------------------------------------------------------
# Plotting helper: recombination efficiency vs dust temperature
# -------------------------------------------------------------------------
def plot_recombination_efficiency_vs_Td(
    nH,
    Tgas,
    T_dust_vals,
    dust_indices=(0, 1),
    figsize=(7, 5),
    savefile=None,
    show=True,
):
    """Plot recombination efficiency as a function of dust temperature.

    Parameters
    - nH : float
        Atomic hydrogen number density (cm^-3).
    - Tgas : float
        Gas temperature (K).
    - T_dust_vals : array_like
        Dust temperatures in K (1D array) to evaluate the efficiency on.
    - dust_indices : tuple, optional
        Tuple/list of dust indices to plot (0 = carbonaceous, 1 = silicate).
    - figsize : tuple, optional
        Figure size passed to `plt.figure`.
    - savefile : str, optional
        If provided, the plot is saved to this path.
    - show : bool, optional
        If True (default) calls `plt.show()`; otherwise closes the figure.

    Example
    >>> Tds = np.linspace(5, 200, 50)
    >>> plot_recombination_efficiency_vs_Td(30.0, 100.0, Tds, dust_indices=(0,1))
    """
    T_dust_vals = np.atleast_1d(T_dust_vals).astype(float)

    vH = np.sqrt(2 * kB * Tgas / mH)
    F = h_flux(nH, vH)

    plt.figure(figsize=figsize)

    for di in dust_indices:
        eff_vals = [recombination_efficiency(Tgas, Td, nH, vH, F, di)
                    for Td in T_dust_vals]
        label = 'Carbonaceous' if di == 0 else 'Silicate' if di == 1 else f'Dust {di}'
        plt.plot(T_dust_vals, eff_vals, marker='o', label=label)

    plt.xlabel('Dust temperature (K)')
    plt.ylabel('Recombination efficiency')
    plt.title(f'Recombination efficiency vs dust temperature (nH={nH:.2e}, Tgas={Tgas} K)')
    plt.grid(True, ls='--', alpha=0.4)
    plt.legend()

    if savefile:
        plt.savefig(savefile, dpi=200, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


# -------------------------------------------------------------------------
# Convenience wrapper: only requires nH and Tgas
# -------------------------------------------------------------------------
def plot_recombination_efficiency(
    nH,
    Tgas,
    Td_min=5.0,
    Td_max=200.0,
    npoints=80,
    dust_indices=(0, 1),
    figsize=(7, 5),
    savefile=None,
    show=True,
):
    """Convenience plot of recombination efficiency vs dust temperature.

    This wrapper requires only `nH` and `Tgas`. It generates a dust
    temperature grid between `Td_min` and `Td_max` with `npoints` values
    and plots recombination efficiency for the requested grain compositions
    (default: carbonaceous and silicate).

    Parameters
    - nH, Tgas : floats
        Gas conditions (cm^-3 and K).
    - Td_min, Td_max : floats
        Range of dust temperatures (K) to evaluate.
    - npoints : int
        Number of temperature points.
    - dust_indices : tuple
        Dust types to compare (0 = carbonaceous, 1 = silicate).
    - figsize, savefile, show : as in other plotting helpers.

    Returns
    - None (shows or saves a plot).
    """
    Tds = np.linspace(float(Td_min), float(Td_max), int(npoints))

    # Use the more general plotting function defined earlier
    plot_recombination_efficiency_vs_Td(
        nH=nH,
        Tgas=Tgas,
        T_dust_vals=Tds,
        dust_indices=dust_indices,
        figsize=figsize,
        savefile=savefile,
        show=show,
    )


# -------------------------------------------------------------------------
# 7) TOTAL H2 formation rate for dust grains
# -------------------------------------------------------------------------
def h2_formation_rate(
    nH, Tgas,
    rho_dust, T_dust,
    asize, mgrain,
    dust_types
):
    """
    dust_types: array of 0 or 1
        0 = carbonaceous
        1 = silicate
    For each grain size j:
        rho_dust[j]  = g/cm3 (mass density per size bin)
        T_dust[j]    = K
        asize[j]     = micron
        mgrain[j]    = g (mass per grain)
    """
    vH = np.sqrt(2 * kB * Tgas / mH)
    F = h_flux(nH, vH)

    total = 0.0
    N = len(rho_dust)

    for j in range(N):
        dtype = dust_types[j]      # 0 or 1
        Td = T_dust[j]

        # Number density of grains * surface 2πa^2
        sdust = (rho_dust[j] / mgrain[j]) * twopi * (asize[j]*1e-4)**2

        R_H2 = sdust * recombination_efficiency(Tgas, Td, nH, vH, F, dtype)
        total += R_H2 * h2_sticking_coef(Tgas, Td)

    return 0.5 * nH * vH * total   # cm^-3 s^-1