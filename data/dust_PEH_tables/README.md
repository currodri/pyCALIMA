# Dust rate tables

Files:
- log10_Ts_Gra_0.0050_micron.dat : log10(T) array (length nT)
- log10_gammas_Gra_0.0050_micron.dat : log10(gamma) array (length n_gamma)
- dust_rates_heating_fix_G0_Gra_0.0050_micron.dat : nT rows x n_gamma columns, log10(heating [erg s^-1])
- dust_rates_cooling_fix_G0_Gra_0.0050_micron.dat : nT rows x n_gamma columns, log10(cooling [erg s^-1])

Rows correspond to increasing T (from Tmin to Tmax). Columns correspond to increasing gamma (from gamma_min to gamma_max).
Missing/invalid values are encoded as -1e30. Tables are plain whitespace-separated ASCII suitable for Fortran reading.
