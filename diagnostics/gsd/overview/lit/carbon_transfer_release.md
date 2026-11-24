# Literature and estimates: release of a physisorbed PAH from an a-C nano-grain under UV (2026-11-23)

Sonnet subagent, read-only. Read in relevant parts: Montillaud & Joblin 2014 (MJ14, Secs 2-4); Zacharia+04 (via fetch
summary); Lange+23 (arXiv 2303.07981, fetch summary of the full text); calima paper (arXiv 2602.21790) Secs 3.6-3.8.
Captions only: Rapacioli+05. Abstract/snippet: Tang & Hammer 2022, Björk+10, Omont & Bettinger 2025, Esmerian+.
Not found: Rapacioli+06, any measured PAH photodesorption yield from carbon, binding energies on amorphous or curved
carbon, any incorporation rate.

pahchem machinery: unimolecular kernels in rates.py (rrkm :50, inverse_laplace_arrhenius :91,
arrhenius_effective_temperature :102, rrk :117, thermal_rate :162); dos.py microcanonical_temperature :235; one-photon
IR cascade (emission.py cascade_photons :33). No van der Waals / cluster evaporation (no phase-space theory, no host +
guest heat capacity); the RRKM/RRK machinery could be reused with the host as a bath.

1. Binding energies
- PAH on graphite: 52 +- 5 meV per C (Zacharia+04, TPD, fitted); benzene 0.50, naphthalene 0.85-0.90, coronene
  1.4-1.5, ovalene 2.0-2.2 eV; prefactors 1e15-1e18 s^-1 (ovalene up to 5e21).
- PAH on graphene (DFT, Li+18 via Lange+23 Eq. 16): E_A/N_C = 0.046 + 0.021 N_H/N_C eV; coronene 1.4 eV, circumcoronene
  ~2.9 eV (agent's evaluation).
- Coronene dimer 94.9 kJ/mol (~0.98 eV), pyrene dimer 54.5 kJ/mol (Rapacioli+05 Figs 1-2); (C24H12)4 -294.5,
  (C54H18)4 -802.0 kJ/mol (MJ14 Sec 4.3). A PAH on a surface binds ~1.4-1.5 times the dimer energy.
- Per molecule at ~0.05 eV/C: ~1.2 eV (C24), 2.6 eV (C50), 5 eV (C100). Curved host (agent's estimate): a C50 loses
  ~0-30 % of its binding on a 1 nm host, nothing on >= 3 nm; amorphous roughness could raise it (no data).

2. Release physics
- MJ14 (phase-space theory, MD-calibrated, homo-molecular, gas phase): photostability "increases dramatically with the
  size of molecules ... and to a lesser extent with the number of molecules". Coronene clusters: tau ~ 1/G0, "quickly
  evaporated even in faint radiation fields"; (C54H18)N (~200-650 C) resistant until G0 ~ 1e3-1e4; (C96H24)12 no
  evaporation in 1e8 yr up to G0 3e7.
- Lange+23 (PAHs frozen on grains in discs): Arrhenius k = 2.5e17 exp(-E_A/kT_e) s^-1, T_e = T_m(1 - 0.2 E_A/E); host
  heat flow kappa 0.04 W/m/K over 3.34 A; thermalisation ~1e-10 s (multi-photon negligible); "PAHs with more than 54
  carbon atoms will stay frozen out" (their Sec 4.2 favours < 66 C: not fully consistent).
- Direct photodesorption of PAHs from carbon: no yield found.
- Agent's estimate (shared energy, crude a-C heat capacity, Arrhenius nu 2.5e17, E_b = 0.052 N_C eV, absorption 1e-17
  cm^2 per C, 1.3e8 photons cm^-2 s^-1 at G0 = 1): 1 nm host (335 C) peaks at 400-640 K for 6-13.6 eV photons: k(C24) 9
  to 2e7 s^-1, k(C50) 1e-20 to 1e-5 s^-1; 3 nm host (9000 C) ~105-137 K: k < 1e-29; 10 nm: zero. Release probability per
  photon collapses with host size (T_peak ~ N_host^-1/2) much faster than the photon rate grows: release is a 1 nm-host,
  single-photon, small-PAH phenomenon. Equilibrium thermal evaporation of C50 needs T_d >= ~400 K (G0 >~ 1e7).

3. Permanent incorporation: no rate found. Covalent dimerisation of dehydrogenated PAHs on graphene (Tang & Hammer 2022):
diffusion ~0.1 eV but H transfer >= 1.82 eV, other steps 3.5-5.4 eV: needs radicals (UV, cosmic rays). Burial by the
next accretion or coagulation event is the only quantifiable route.

4. Prior models: Lange+23 (perfect sticking + vdW release); calima paper Sec 3.8 (PAH freezing, psi = 1/(1+exp[(E-1)/0.1])
in collision energy, no G0 dependence, permanent except a cluster-only evaporation fit); Narayanan+26 (PAHs as the
smallest bin of the same network, no release).

5. Recommended form: S_eff = S0 / (1 + R_rel tau_inc), R_rel = photon rate on the host x P_rel(a, N_C, spectrum);
tau_inc ~ time to burial (order 2 Myr for a 1 nm host at n_H 100, agent's estimate). Indicative (orders of magnitude
uncertain): 1 nm host: C24 released at any G0 >~ 1e-7 (no sticking), C50 stays only below G0 ~ 1e-1 to 1e3; 3 nm and
10 nm hosts: both stick at all G0 below ~1e7.
