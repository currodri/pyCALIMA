"""
models/shiva — SHIVA model for PAH evolution.

Implements the Stochastic Heating and Ion–Victim Approach (SHIVA) as presented
in Murga et al. (2016, 2019, 2020), using:

  • Draine & Li 2001 (DL01)  — internal energy U(T) and microcanonical temperature
  • Draine & Li 2007 (DL07)  — UV/optical absorption cross-sections
  • Weingartner & Draine 2001a (WD01a) — photoionization yields and charge balance
  • Guhathakurta & Draine 1989 (GD89) / Pavlyuchenkov+2012 — P(T) distribution
  • Tielens 2005 Arrhenius law — dissociation rates at microcanonical temperature

Submodules
----------
dl01_internal_energy  — U(T, Nc) and T_micro(E, Nc) from DL01
dl07_crosssections    — analytical DL07 C_abs(E, Nc, Z)
wd01a_yields          — photoionization yields and recombination rates (WD01a)
gd89_heating          — steady-state P(T) distribution via GD89 forward recursion
shiva_charge          — 4-state charge-balance solver (<Z>, f_Z)
shiva_dissociation    — Arrhenius C₂H₂ loss rate integrated over P(T)
validate_murga2020    — reproduction of Murga+2020 Figures 3 & 4
"""
