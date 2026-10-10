"""One-zone collapse with Grackle's chemistry (Chiaki & Wise 2019 metal and dust
network), as a reference for ``pycalima.collapse``.

Runs under a gracklepy install, not pyCALIMA's environment (see
``make_grackle_tables.py`` for the build). Writes the run to an ``.npz``:

    python grackle_freefall.py OUT.npz --log-z -4 --pathway 2 --data-dir DIR \
        [--grain-growth] [--dust-free]

Saved: n_H, T, mu, f_H2 = 2 n(H2) / n_H, x_e = n_e / n_H, xi = sum_i n_i (m_e / m_i)^1/2 / n_e
over the positive ions (the ratio of ion to electron currents onto a neutral
grain; ``pycalima.collapse.charge``) and x(H2O) = n(H2O) / n_H.

Physics: Grackle's chemistry, cooling and dust, from n_H = 0.1 cm^-3, 300 K,
x_e = 1e-4 (Omukai et al. 2005) to 1e16 cm^-3, z = 0 CMB. ``--log-z`` is
[Z/H] with Z_sun = 0.02 (the convention of Chiaki et al. 2015); the metals are
those of injection pathway ``--pathway`` (1-4: the N07 13, 20, 25, 30 Msun
CCSNe with reverse shock at n_amb = 1 cm^-3). ``--dust-free`` switches the dust
module off (dust_chemistry = 0), leaving the gas-phase metals of the pathway:
setting the grain fields or h2_on_dust to zero does not remove the dust, which
Grackle then derives from the metal density.

The collapse is the Omukai et al. (2005) one: d rho / dt = rho / t_col with
t_col = t_ff / (1 - f(gamma))^1/2, f = 0 until the first temperature minimum,
and compressional heating (p / rho) d ln rho. gracklepy's ``evolve_freefall``
is not used: it applies f from the first step and heats at the free-fall rate
whatever f is, which overheats the gas by up to (1 - 0.95)^-1/2 = 4.5 (its
Z = 0 collapse reaches 4700 K and dissociates its H2 above 1e13 cm^-3).
"""

import argparse
import contextlib
import io
import os
import time
from collections import defaultdict

import numpy as np
from gracklepy import chemistry_data, setup_fluid_container
from gracklepy.utilities.evolve import add_to_data, calculate_collapse_factor
from gracklepy.utilities.physical_constants import (cm_per_mpc, gravitational_constant_cgs,
                                                    mass_hydrogen_cgs as MH, sec_per_Myr)

X_H = 0.76          # Grackle's hydrogen mass fraction
Z_SUN = 0.02        # Chiaki+15
ME_OVER_MH = 5.4462e-4
# positive ions and their mass numbers (HeIII, doubly charged and absent at these T, left out)
IONS = {"HII": 1, "HeII": 4, "H2II": 2, "HeHII": 5, "DII": 2, "HDII": 3, "CII": 12, "COII": 28, "OII": 16,
        "OHII": 17, "H2OII": 18, "H3OII": 19, "O2II": 32}


def chemistry(pathway, grain_growth, dust_free, data_dir):
    """The settings of Grackle's ``metal_dust_chemistry`` free-fall test, z = 0."""
    c = chemistry_data()
    for k, v in dict(use_grackle=1, with_radiative_cooling=1, primordial_chemistry=4, CaseBRecombination=1,
                     cie_cooling=2, h2_optical_depth_approximation=1, use_primordial_continuum_opacity=1,
                     h2_charge_exchange_rate=1, h2_h_cooling_rate=1, h2_cooling_rate=3, hd_reaction_rates=1,
                     metal_cooling=1, metal_chemistry=1, three_body_rate=4,
                     dust_chemistry=0 if dust_free else 2, photoelectric_heating=0,
                     dust_recombination_cooling=0 if dust_free else 1,
                     grain_growth=0 if dust_free else int(grain_growth), dust_species=0 if dust_free else 2,
                     multi_metals=0, metal_abundances=pathway, h2_on_dust=0 if dust_free else 1,
                     cmb_temperature_floor=1, UVbackground=0, comoving_coordinates=0, a_units=1.0, a_value=1.0,
                     density_units=MH, length_units=cm_per_mpc, time_units=sec_per_Myr).items():
        setattr(c, k, v)
    c.grackle_data_file = os.path.join(data_dir, "cloudy_metals_2008_3D.h5")
    c.set_velocity_units()
    return c


def collapse(fc, final_density, safety=0.01):
    """Omukai et al. (2005) collapse around Grackle's chemistry (see the module docstring)."""
    cd = fc.chemistry_data
    G = 4.0 * np.pi * gravitational_constant_cgs * cd.density_units * cd.time_units ** 2
    k_ff = np.sqrt(32.0 * G / (3.0 * np.pi))
    data, t, passed_min = defaultdict(list), 0.0, False
    while fc["density"][0] * cd.density_units < final_density:
        dt = safety * np.sqrt(3.0 * np.pi / (32.0 * G * fc["density"][0]))
        add_to_data(fc, data, extra={"time": t})
        T = data["temperature"]
        if not passed_min and len(T) > 3 and T[-1] > T[-2] and T[-2] < T[-3]:
            passed_min = True
        f = np.asarray(calculate_collapse_factor(data["pressure"], data["density"])).item() if passed_min else 0.0
        ratio = (fc["density"][0] ** -0.5 - 0.5 * k_ff * dt * np.sqrt(1.0 - f)) ** -2.0 / fc["density"][0]
        fc.calculate_pressure()
        fc["internal_energy"][0] += fc["pressure"][0] / fc["density"][0] * np.log(ratio)
        for field in fc.density_fields:
            fc[field] *= ratio
        fc.solve_chemistry(dt)
        t += dt
    for field in data:
        data[field] = np.squeeze(np.array(data[field]))
    return fc.finalize_data(data=data)


def run(log_z, pathway, data_dir, grain_growth=False, dust_free=False, n_end=1e16):
    """{n_H, T, mu, fH2, x_e, xi, x_H2O} along the collapse (see the module docstring)."""
    c = chemistry(pathway, grain_growth, dust_free, data_dir)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        fc = setup_fluid_container(c, density=0.1 / X_H * MH, temperature=300.0, state="neutral",
                                   metal_mass_fraction=10.0 ** log_z * Z_SUN, dust_to_gas_ratio=1e-20)
        xe = 1e-4 * fc["HI_density"]          # Omukai et al. (2005): residual ionisation
        fc["HII_density"][:] = xe
        fc["HI_density"][:] -= xe
        fc["e_density"][:] = xe
        d = collapse(fc, n_end / X_H * MH)
    g = {k: np.asarray(v) for k, v in d.items()}
    rho = g["density"]                              # g cm^-3: finalize_data returns cgs
    nH = rho * X_H / MH
    n_e = g["e_density"] / MH                      # Grackle stores the electrons as n_e m_H
    xi = sum(g[f"{k}_density"] / (A * MH) * np.sqrt(ME_OVER_MH / A) for k, A in IONS.items()
             if f"{k}_density" in g) / n_e
    return dict(nH=nH, T=g["temperature"], mu=g["mean_molecular_weight"],
                fH2=g["H2I_density"] / (g["HI_density"] + g["H2I_density"]), x_e=n_e / nH, xi=xi,
                x_H2O=g["H2OI_density"] / (18.0 * MH) / nH if "H2OI_density" in g else np.zeros_like(nH))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--log-z", type=float, required=True)
    ap.add_argument("--pathway", type=int, required=True)
    ap.add_argument("--data-dir", required=True, help="grackle_data_files/input of the Grackle checkout")
    ap.add_argument("--grain-growth", action="store_true")
    ap.add_argument("--dust-free", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    r = run(a.log_z, a.pathway, a.data_dir, a.grain_growth, a.dust_free)
    np.savez(a.out, **r)
    print(a.out, f"{time.time() - t0:.1f} s, {r['nH'].size} steps")


if __name__ == "__main__":
    main()
