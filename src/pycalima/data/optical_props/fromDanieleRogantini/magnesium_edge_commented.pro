;+
; NAME:
;   magnesium_edge_analysis
;
; PURPOSE:
;   Convert laboratory transmission measurements around the Mg K edge into
;   optical constants and dust extinction/absorption/scattering cross sections.
;   The final output is an AMOL/SPEX-compatible extinction profile.
;
; DESCRIPTION:
;   The workflow follows the laboratory-based X-ray dust modelling approach used
;   in Rogantini et al. (2018, A&A, 609, A22) and Rogantini et al. (2019, A&A,
;   630, A143). The script:
;
;     1. reads the selected laboratory sample;
;     2. applies pile-up and fluorescence self-absorption corrections;
;     3. derives the imaginary part of the refractive index, k;
;     4. reads the Kramers-Kronig output to recover f1 and f2;
;     5. converts f1 and f2 into n and k optical constants;
;     6. computes Q_ext, Q_abs, and Q_sca using the anomalous diffraction
;        theory approximation;
;     7. integrates the efficiencies over an MRN grain-size distribution;
;     8. prepares a SPEX/AMOL table in Mbarn.
;
; REQUIRED EXTERNAL ROUTINES / INPUTS:
;   - CONSTANTS()
;   - READCOL
;   - WRITECOL
;   - PILEUP
;   - FLUO_CORRECTION
;   - FITTING_DATA
;   - KKcalc output files containing f1 and f2
;
; NOTES:
;   - Grain radii are in micron unless explicitly converted to cm.
;   - Energies are in keV in the ADT calculation and in eV in the SPEX output.
;   - The MRN distribution is n(a) = A a^(-3.5).
;   - The current paths are hard-coded for the original magnesium-edge workflow.
;     Move them to a configuration block if this script is generalized.
;
; REFERENCES:
;   Rogantini et al. 2018, A&A, 609, A22
;   Rogantini et al. 2019, A&A, 630, A143
;-

;==============================================================================
; Compute the logarithmic MRN normalisation coefficient.
;
; The returned value is log10(A), where A is the coefficient used in
; n(a) = A a^(-q). Relevant references Mauche & Gorenstein 1986,  
; Draine (2003) and Hoffman & Draine 2016
;==============================================================================

FUNCTION calculate_mrn_log_norm, n_ref, k_ref, amin_um, amax_um

  COMPILE_OPT idl2

  IF (N_PARAMS() LT 4) THEN MESSAGE, $
    'Usage: logA = calculate_mrn_log_norm(n_ref, k_ref, amin_um, amax_um)'

  ; Reference energy used for the normalisation.
  energy_kev = 1.0D

  ; MRN parameters.
  q_size  = 3.5D
  n_radii = 100L

  ; Grain-size grid, logarithmically spaced in micron.
  radii_um = 10.D^(DINDGEN(n_radii) * $
             (ALOG10(amax_um) - ALOG10(amin_um)) / (n_radii - 1.D) + $
             ALOG10(amin_um))

  ; ADT extinction and absorption efficiencies at the reference energy.
  q_ext = DBLARR(n_radii)
  q_abs = DBLARR(n_radii)

  tan_beta  = k_ref / (n_ref - 1.D)
  beta      = ATAN(tan_beta)
  cos_beta  = COS(beta)
  cos_2beta = COS(2.D * beta)

  FOR j = 0L, n_radii - 1L DO BEGIN

    ; Size parameter: x = 2 pi a / lambda.
    ; The factor 1e-4 converts micron to cm, while 12.39852/E gives lambda in A.
    x_size = 2.D * !PI * radii_um[j] / (1.D-4 * 12.39852D / energy_kev)

    ; van de Hulst ADT phase-shift parameter.
    rho_hulst = 2.D * x_size * (n_ref - 1.D)

    q_ext[j] = 2.D - $
      4.D * EXP(-rho_hulst * tan_beta) * (cos_beta / rho_hulst) * $
      SIN(rho_hulst - beta) - $
      4.D * EXP(-rho_hulst * tan_beta) * (cos_beta / rho_hulst)^2.D * $
      COS(rho_hulst - 2.D * beta) + $
      4.D * (cos_beta / rho_hulst)^2.D * cos_2beta

    ; Absorption efficiency from the ADT approximation, as in the original
    ; Draine/Hoffman-Draine implementation used by the old script.
    rho_abs = 2.D * x_size * k_ref
    fac     = EXP(-rho_abs)
    fac2    = fac * fac

    q_abs[j] = 1.D + (fac2 + 0.5D * (fac2 - 1.D) / rho_abs) / rho_abs

  ENDFOR

  ; Integrate Q_abs * pi a^2 * a^(-q) da over the grain-size distribution.
  ; The original script used Q_abs for this normalisation, so this is preserved.
  integrand = DBLARR(n_radii)

  FOR i = 0L, n_radii - 2L DO BEGIN

    IF (i EQ 0L) THEN delta_um = radii_um[1] - radii_um[0] $
                  ELSE delta_um = radii_um[i] - radii_um[i - 1]

    radius_cm = radii_um[i] * 1.D-4
    delta_cm  = delta_um * 1.D-4

    integrand[i] = q_abs[i] * (!PI * radius_cm^2.D) * $
                   radius_cm^(-q_size) * delta_cm

  ENDFOR

  total_abs = TOTAL(integrand)

  ; Reference value used in the original script. This corresponds to the
  ; cross-section scaling adopted for the old silicate implementation.
  reference_cross_section = 5.83D-23
  n_hydrogen              = 1.D

  norm_a = reference_cross_section / (total_abs * n_hydrogen)

  PRINT, '**************************'
  PRINT, 'log10(A_MRN) = ', ALOG10(norm_a)
  PRINT, '**************************'

  RETURN, ALOG10(norm_a)

END


;==============================================================================
; Compute ADT cross sections integrated over an MRN grain-size distribution.
; Refernces van de Hulst 1957 and Hoffman & Draine (2016)
;
; INPUT:
;   energy_kev : energy grid in keV
;   n_real     : real part of refractive index
;   k_imag     : imaginary part of refractive index
;   radii_um   : grain-size grid in micron
;   log_norm_a : log10 of MRN normalisation coefficient
;
; OUTPUT:
;   ext_cm2_h  : extinction cross section in cm^2 H^-1
;   abs_cm2_h  : absorption cross section in cm^2 H^-1
;   sca_cm2_h  : scattering cross section in cm^2 H^-1
;==============================================================================

PRO compute_adt_cross_sections, energy_kev, n_real, k_imag, radii_um, log_norm_a, $
                                ext_cm2_h, abs_cm2_h, sca_cm2_h

  COMPILE_OPT idl2

  n_energy = N_ELEMENTS(energy_kev)
  n_radii  = N_ELEMENTS(radii_um)
  q_size   = 3.5D

  ext_cm2_h = DBLARR(n_energy)
  abs_cm2_h = DBLARR(n_energy)
  sca_cm2_h = DBLARR(n_energy)

  q_ext = DBLARR(n_energy, n_radii)
  q_abs = DBLARR(n_energy, n_radii)
  q_sca = DBLARR(n_energy, n_radii)

  ext_grid = DBLARR(n_energy, n_radii)
  abs_grid = DBLARR(n_energy, n_radii)
  sca_grid = DBLARR(n_energy, n_radii)

  ; For the current linear grid the radius step is constant, as in the original
  ; script. If a logarithmic grid is used, this should be replaced by local da.
  delta_cm = (radii_um[1] - radii_um[0]) * 1.D-4

  tan_beta  = k_imag / (n_real - 1.D)
  beta      = ATAN(tan_beta)
  cos_beta  = COS(beta)
  cos_2beta = COS(2.D * beta)

  FOR i = 0L, n_energy - 2L DO BEGIN

    FOR j = 0L, n_radii - 1L DO BEGIN

      radius_cm = radii_um[j] * 1.D-4

      ; x = 2 pi a / lambda. Here lambda is expressed consistently with the
      ; original ADT implementation.
      x_size = 2.D * !PI * radii_um[j] / (1.D-4 * 12.39852D / energy_kev[i])

      ; Extinction efficiency.
      rho_hulst = 2.D * x_size * (n_real[i] - 1.D)

      q_ext[i, j] = 2.D - $
        4.D * EXP(-rho_hulst * tan_beta[i]) * (cos_beta[i] / rho_hulst) * $
        SIN(rho_hulst - beta[i]) - $
        4.D * EXP(-rho_hulst * tan_beta[i]) * (cos_beta[i] / rho_hulst)^2.D * $
        COS(rho_hulst - 2.D * beta[i]) + $
        4.D * (cos_beta[i] / rho_hulst)^2.D * cos_2beta[i]

      ; Absorption efficiency.
      rho_abs = 2.D * x_size * k_imag[i]
      fac     = EXP(-rho_abs)
      fac2    = fac * fac

      q_abs[i, j] = 1.D + (fac2 + 0.5D * (fac2 - 1.D) / rho_abs) / rho_abs

      ; Scattering efficiency.
      q_sca[i, j] = q_ext[i, j] - q_abs[i, j]

      ; Convert efficiencies to cross-section contribution:
      ;   Q * geometric area * MRN weight * A * da.
      mrn_weight = radius_cm^(-q_size)
      geom_area  = !PI * radius_cm^2.D
      norm_a     = 10.D^log_norm_a

      ext_grid[i, j] = q_ext[i, j] * geom_area * mrn_weight * norm_a * delta_cm
      abs_grid[i, j] = q_abs[i, j] * geom_area * mrn_weight * norm_a * delta_cm
      sca_grid[i, j] = q_sca[i, j] * geom_area * mrn_weight * norm_a * delta_cm

    ENDFOR

    ext_cm2_h[i] = TOTAL(ext_grid[i, *])
    abs_cm2_h[i] = TOTAL(abs_grid[i, *])
    sca_cm2_h[i] = TOTAL(sca_grid[i, *])

  ENDFOR

END


;==============================================================================
; Main procedure for the Mg edge.
;==============================================================================

PRO magnesium_edge_analysis

  COMPILE_OPT idl2

  ;--------------------------------------------------------------------------
  ; Configuration.
  ;--------------------------------------------------------------------------

  cgs = CONSTANTS()

  base_dir  = '/home/danieler/magnesium/magnesium_data/'
  stage_dir = '/stage/danieler/magnesium/'

  sample_table = base_dir + 'magnesium_sample.txt'

  transmission_dir = base_dir  + 'transmission/'
  kkcalc_dir       = base_dir  + 'kkcalc/'
  henke_dir        = base_dir  + 'henke/'

  lab_const_dir = stage_dir + 'lab_optical_const/'
  adt_dir       = stage_dir + 'adt/'
  plot_dir      = stage_dir + 'adt/plot/'
  amol_dir      = stage_dir + 'amol/'

  ; MRN grain-size distribution.
  amin_um = 0.005D
  amax_um = 0.25D
  q_size  = 3.5D
  n_radii = 300L

  ; Magnesium abundance used to convert from cm^2 H^-1 to cm^2 Mg^-1.
  ; The value is the sum of Mg ion abundances in the SPEX hot model, consistent
  ; with the value adopted in the original script and close to Whittet.
  abundance_mg = 3.97D-5

  ;--------------------------------------------------------------------------
  ; Select the laboratory sample.
  ;--------------------------------------------------------------------------

  READ, sample_number, PROMPT='Sample number [1-12]: '

  READCOL, sample_table, compound_name, density, molecular_mass, $
           FORMAT='A,F,F', /SILENT

  sample_name = compound_name[sample_number - 1]
  rho         = density[sample_number - 1]
  mol_mass    = molecular_mass[sample_number - 1]

  sample_tag = STRTRIM(sample_name, 2)

  ;--------------------------------------------------------------------------
  ; Read and correct the laboratory transmission measurements.
  ;--------------------------------------------------------------------------

  ; These two routines are external to this script. They should return the
  ; corrected energy and flux arrays used by the transmission fit.
  PILEUP, sample_number, energy_kev_lab, flux, pileup
  FLUO_CORRECTION, sample_number, energy_kev_lab, flux

  ; Compute the sample transmission using Henke optical constants.
  transmission = FLTARR(1000)
  FITTING_DATA, sample_number, energy_kev_lab, transmission

  ; Save the corrected transmission curve.
  OPENW, lun, transmission_dir + sample_tag + '_transmission.txt', /GET_LUN
  PRINTF, lun, "compound's name = " + sample_tag
  PRINTF, lun, 'energy [eV]    transmission'

  FOR i = 0L, N_ELEMENTS(energy_kev_lab) - 1L DO BEGIN
    PRINTF, lun, energy_kev_lab[i] * 1000.D, transmission[i]
  ENDFOR

  CLOSE, lun
  FREE_LUN, lun

  ;--------------------------------------------------------------------------
  ; Derive k = Im(m - 1) from the transmission.
  ;--------------------------------------------------------------------------

  energy_ev_lab = energy_kev_lab * 1000.D

  ; Sample thickness in micron.
  thickness_um = 0.5D

  ; alpha is in micron^-1 because the thickness is in micron.
  alpha = -ALOG(transmission) / thickness_um

  ; lambda in cm and micron.
  lambda_cm     = cgs.hc / energy_ev_lab
  lambda_micron = lambda_cm * 1.D4

  ; Imaginary part of refractive index.
  k_lab = alpha * lambda_micron / (4.D * !PI)

  PLOT, energy_kev_lab, k_lab

  ; Save k for KKcalc.
  OPENW, lun, kkcalc_dir + sample_tag + '_kappa_KK.txt', /GET_LUN
  PRINTF, lun, 'energy    k=Im(m-1)'

  FOR i = 0L, N_ELEMENTS(energy_ev_lab) - 1L DO BEGIN
    PRINTF, lun, energy_ev_lab[i], k_lab[i]
  ENDFOR

  CLOSE, lun
  FREE_LUN, lun

  ; Convert k into f2 for KKcalc.
  lambda_cm = cgs.hc / energy_ev_lab
  f2_lab = (k_lab * 2.D * !PI * mol_mass) / $
           (cgs.Na * rho * cgs.re * lambda_cm^2.D)

  OPENW, lun, kkcalc_dir + sample_tag + '_f2_KK.txt', /GET_LUN
  PRINTF, lun, 'energy    f2'

  FOR i = 0L, N_ELEMENTS(energy_ev_lab) - 1L DO BEGIN
    PRINTF, lun, energy_ev_lab[i], f2_lab[i]
  ENDFOR

  CLOSE, lun
  FREE_LUN, lun

  ;--------------------------------------------------------------------------
  ; Read KKcalc output and recover n and k on the desired energy grid.
  ;--------------------------------------------------------------------------

  READCOL, kkcalc_dir + sample_tag + '_f1_KK.txt', kk_energy_ev, f1_kk, f2_kk, $
           FORMAT='D,D,D'

  ; Energy grid around the Mg K edge.
  energy_ev  = FINDGEN(4501) * 0.1D + 1100.D
  energy_kev = energy_ev / 1000.D

  f1 = INTERPOL(f1_kk, kk_energy_ev, energy_ev)
  f2 = INTERPOL(f2_kk, kk_energy_ev, energy_ev)

  lambda_cm = cgs.hc / energy_ev

  delta_m = cgs.Na * rho * cgs.re * lambda_cm^2.D * f1 / $
            (2.D * !PI * mol_mass)

  n_real = 1.D - delta_m
  k_imag = cgs.Na * rho * cgs.re * lambda_cm^2.D * f2 / $
           (2.D * !PI * mol_mass)

  ; Save the laboratory optical constants used for the ADT calculation.
  OPENW, lun, lab_const_dir + sample_tag + '_lab_optical_constants.txt', /GET_LUN
  PRINTF, lun, 'energy[ev]    n-1    k'

  FOR i = 0L, N_ELEMENTS(energy_ev) - 1L DO BEGIN
    PRINTF, lun, energy_ev[i], n_real[i] - 1.D, k_imag[i]
  ENDFOR

  CLOSE, lun
  FREE_LUN, lun

  refractive_index = DCOMPLEX(n_real, k_imag)

  ;--------------------------------------------------------------------------
  ; ADT calculation: extinction, absorption, and scattering.
  ;--------------------------------------------------------------------------

  ; Linear MRN grid, preserved from the original script.
  radii_um = DBLARR(n_radii)

  FOR i = 0L, n_radii - 1L DO BEGIN
    radii_um[i] = i / (n_radii - 1.D) * (amax_um - amin_um) + amin_um
  ENDFOR

  ; Estimate the MRN normalisation coefficient using Henke optical constants at
  ; the first available energy point.
  READCOL, henke_dir + sample_tag + '_optical_constants.txt', $
           henke_energy, henke_delta, henke_beta, /SILENT

  n_ref = 1.D - henke_delta[0]
  k_ref = henke_beta[0]

  log_norm_a = calculate_mrn_log_norm(n_ref, k_ref, amin_um, amax_um)

  ; Compute integrated cross sections in cm^2 H^-1.
  COMPUTE_ADT_CROSS_SECTIONS, energy_kev, n_real, k_imag, radii_um, log_norm_a, $
                              ext_cm2_h, abs_cm2_h, sca_cm2_h

  ; Save the raw cross sections.
  OPENW, lun, adt_dir + sample_tag + '_ext_sca_abs.txt', /GET_LUN
  PRINTF, lun, 'energy [keV], cross section [cm^2 H^-1]'
  PRINTF, lun, 'energy        extinction        scattering        absorption'

  FOR i = 0L, N_ELEMENTS(energy_kev) - 1L DO BEGIN
    PRINTF, lun, energy_kev[i], ext_cm2_h[i], sca_cm2_h[i], abs_cm2_h[i]
  ENDFOR

  CLOSE, lun
  FREE_LUN, lun

  ; Quick diagnostic plot.
  SET_PLOT, 'PS'
  DEVICE, FILENAME=plot_dir + sample_tag + '_ext_sca_abs.ps', $
          DECOMPOSED=1, COLOR=1

  PLOT, energy_kev, ext_cm2_h, XSTYLE=1
  OPLOT, energy_kev, abs_cm2_h, COLOR=250
  OPLOT, energy_kev, sca_cm2_h, LINESTYLE=1

  DEVICE, /CLOSE
  SET_PLOT, 'x'

  ;--------------------------------------------------------------------------
  ; Convert from cm^2 H^-1 to cm^2 Mg^-1 and remove the continuum slope.
  ;--------------------------------------------------------------------------

  ext_cm2_mg = ext_cm2_h / abundance_mg
  abs_cm2_mg = abs_cm2_h / abundance_mg
  sca_cm2_mg = sca_cm2_h / abundance_mg

  ; First continuum option:
  ;   - quadratic pre-edge fit
  ;   - cubic post-edge correction
  pre  = WHERE(energy_kev LE 1.25D)
  post = WHERE((energy_kev LE 1.549D) AND (energy_kev GT 1.35D))

  pre_par = POLY_FIT(energy_kev[pre], ext_cm2_mg[pre], 2)
  pre_fit = pre_par[0] + pre_par[1] * energy_kev + pre_par[2] * energy_kev^2.D

  ext_fit1 = ext_cm2_mg - pre_fit

  post_par = POLY_FIT(energy_kev[post], ext_fit1[post], 3)
  post_fit = post_par[0] + post_par[1] * energy_kev + $
             post_par[2] * energy_kev^2.D + post_par[3] * energy_kev^3.D

  correction = WHERE(energy_kev GE 1.4D)
  y0         = post_fit[correction[0]]

  reference_line = DBLARR(N_ELEMENTS(post_fit)) + y0
  diff           = reference_line - post_fit

  ext_fit1[correction] = ext_fit1[correction] + diff[correction]

  PLOT, energy_kev, ext_fit1, XSTYLE=1, YSTYLE=1, XRANGE=[1.1D, 1.549D]

  ; Second continuum option:
  ;   - quadratic pre-edge fit
  ;   - quadratic post-edge fit
  pre  = WHERE(energy_kev LE 1.25D)
  post = WHERE((energy_kev LE 1.548D) AND (energy_kev GT 1.4D))

  pre_par = POLY_FIT(energy_kev[pre], ext_cm2_mg[pre], 2)
  pre_fit = pre_par[0] + pre_par[1] * energy_kev + pre_par[2] * energy_kev^2.D

  post_par = POLY_FIT(energy_kev[post], ext_cm2_mg[post], 2)
  post_fit = post_par[0] + post_par[1] * energy_kev + $
             post_par[2] * energy_kev^2.D

  before_edge = WHERE(energy_kev LE 1.315D)
  after_edge  = WHERE(energy_kev GT 1.315D)

  front_fit = pre_fit[before_edge]
  back_fit  = post_fit[after_edge]

  offset  = back_fit[0] - front_fit[-1]
  fit_tot = [pre_fit[before_edge], post_fit[after_edge] - offset]

  ext_fit2 = ext_cm2_mg - fit_tot

  OPLOT, energy_kev, ext_fit2, LINESTYLE=1

  ; Third continuum option:
  ;   - quadratic pre-edge fit
  ;   - linear post-edge fit
  post_par = POLY_FIT(energy_kev[post], ext_cm2_mg[post], 1)
  post_fit = post_par[0] + post_par[1] * energy_kev

  back_fit = post_fit[after_edge]
  offset   = back_fit[0] - front_fit[-1]
  fit_tot  = [pre_fit[before_edge], post_fit[after_edge] - offset]

  ext_fit3 = ext_cm2_mg - fit_tot

  OPLOT, energy_kev, ext_fit3, LINESTYLE=3, COLOR=250

  READ, fit_choice, $
        PROMPT='Which fit represents better the profile? RED=1, DOT=2, RED_DOT=3 [1,2,3]: '

  IF (fit_choice EQ 1) THEN BEGIN
    ext_fit = ext_fit1
  ENDIF ELSE IF (fit_choice EQ 2) THEN BEGIN
    ext_fit = ext_fit2
  ENDIF ELSE IF (fit_choice EQ 3) THEN BEGIN
    ext_fit = ext_fit3
  ENDIF ELSE BEGIN
    MESSAGE, 'Invalid fit choice. Please select 1, 2, or 3.'
  ENDELSE

  ;--------------------------------------------------------------------------
  ; Prepare the SPEX/AMOL output.
  ;--------------------------------------------------------------------------

  ; cm^2 -> Mbarn. 1 cm^2 = 1e18 Mbarn.
  ext_spex_mbarn = ext_fit * 1.D18
  energy_spex_ev = energy_ev

  ; Energy range used in SPEX.
  keep = WHERE((energy_spex_ev GE 1100.D) AND (energy_spex_ev LE 1548.D))

  ext_spex_mbarn = ext_spex_mbarn[keep]
  energy_spex_ev = energy_spex_ev[keep]

  ; SPEX format: set the first point to zero.
  ext_spex_mbarn = ext_spex_mbarn - ext_spex_mbarn[0]

  output_spex = amol_dir + sample_tag + '_ext_mrn.mg.1s'

  PRINT, output_spex
  PRINT, ''
  PRINT, 'This is the end'
  PRINT, '               my beautiful friend'
  PRINT, ''

  WRITECOL, output_spex, energy_spex_ev, ext_spex_mbarn, FMT='(f,f)'

  STOP

END
